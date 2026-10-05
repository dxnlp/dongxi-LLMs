"""Independent CPU endpoint, enumeration, likelihood and gradient checks."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import torch

from dongxi_llms.critic_policy_lab import (
    EOS,RED,BLUE,PLAIN,FANCY,load_protocol,terminal_paths,state_prefixes,
    prompt_ids,pad_states,make_models,gae_targets,actor_loss,critic_loss,
    independent_quality,fit_rewards,oracle_tree,rollout,path_statistics,validate_protocol,
    run_arm,state_hash,allowed,
)


class CriticPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.protocol=load_protocol()
        cls.temporary=tempfile.TemporaryDirectory()
        small=dict(cls.protocol,reward_fit_updates=2)
        cls.bundles,cls.fits=fit_rewards(Path(__file__).resolve().parents[1],Path(cls.temporary.name)/"exports",small)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def tensors(self):
        rewards=torch.tensor([[0.,0.,2.],[0.,0.,0.]],dtype=torch.float64,requires_grad=True)
        values=torch.tensor([[.5,.7,.9],[.1,.2,.3]],dtype=torch.float64,requires_grad=True)
        mask=torch.tensor([[True,True,True],[True,True,False]])
        terminal=torch.tensor([[False,False,True],[False,False,False]])
        bootstrap=torch.tensor([9.,1.2],dtype=torch.float64,requires_grad=True)
        return rewards,values,mask,terminal,bootstrap

    def test_lambda_endpoints_and_nonterminal_bootstrap(self):
        tensors=self.tensors()
        monte=gae_targets(*tensors,gamma=.9,lam=1)
        torch.testing.assert_close(monte["targets"],torch.tensor([[1.62,1.8,2.],[.972,1.08,0.]],dtype=torch.float64))
        td=gae_targets(*tensors,gamma=.9,lam=0)
        torch.testing.assert_close(td["targets"],torch.tensor([[.63,.81,2.],[.18,1.08,0.]],dtype=torch.float64))
        self.assertFalse(any(v.requires_grad for v in monte.values()))

    def test_independent_closed_sum_matches_gae(self):
        rewards,values,mask,terminal,bootstrap=self.tensors()
        for lam in (0.,.3,1.):
            result=gae_targets(rewards,values,mask,terminal,bootstrap,gamma=.9,lam=lam)
            expected=torch.zeros_like(values)
            for row in range(2):
                length=int(mask[row].sum()); deltas=[]
                for t in range(length):
                    nxt=0. if terminal[row,t] else float((values[row,t+1] if t+1<length else bootstrap[row]).detach())
                    deltas.append(float(rewards[row,t].detach())+.9*nxt-float(values[row,t].detach()))
                for t in range(length):expected[row,t]=sum((.9*lam)**(u-t)*deltas[u] for u in range(t,length))
            torch.testing.assert_close(result["advantages"],expected)

    def test_cap_as_eos_is_an_observable_broken_variant(self):
        rewards,values,mask,terminal,bootstrap=self.tensors()
        correct=gae_targets(rewards,values,mask,terminal,bootstrap,lam=1)
        broken=terminal.clone();broken[1,1]=True
        wrong=gae_targets(rewards,values,mask,broken,bootstrap,lam=1)
        self.assertEqual(float(wrong["targets"][1,0]),0.)
        self.assertAlmostEqual(float(correct["targets"][1,0]),1.2)

    def test_padding_invariance_and_reject_action_after_eos(self):
        tensors=self.tensors(); result=gae_targets(*tensors)
        rewards,values,mask,terminal,bootstrap=tensors
        padded=gae_targets(torch.cat((rewards,torch.full((2,2),77.)),1),
                           torch.cat((values,torch.full((2,2),-99.)),1),
                           torch.cat((mask,torch.zeros(2,2,dtype=torch.bool)),1),
                           torch.cat((terminal,torch.zeros(2,2,dtype=torch.bool)),1),bootstrap)
        for key in result:torch.testing.assert_close(result[key],padded[key][:,:3])
        terminal[0,0]=True
        with self.assertRaisesRegex(ValueError,"after EOS"):gae_targets(rewards,values,mask,terminal,bootstrap)

    def test_policy_advantage_and_old_policy_are_detached(self):
        logp=torch.tensor([[-.2,-.4]],dtype=torch.float64,requires_grad=True)
        old=logp.detach().clone().requires_grad_(True)
        advantage=torch.tensor([[2.,-1.]],dtype=torch.float64,requires_grad=True)
        actor_loss(logp,old,advantage,torch.ones(1,2,dtype=torch.bool)).backward()
        torch.testing.assert_close(logp.grad,torch.tensor([[-2.,1.]],dtype=torch.float64))
        self.assertIsNone(old.grad);self.assertIsNone(advantage.grad)

    def test_critic_own_mse_detaches_targets(self):
        values=torch.tensor([[.2,.3]],requires_grad=True)
        targets=torch.tensor([[1.,2.]],requires_grad=True)
        critic_loss(values,targets,torch.tensor([[True,False]])).backward()
        torch.testing.assert_close(values.grad,torch.tensor([[-1.6,0.]]));self.assertIsNone(targets.grad)

    def test_finite_paths_and_quality_do_not_accept_truncation(self):
        self.assertEqual(len(terminal_paths()),14);self.assertEqual(len(state_prefixes()),15)
        red=self.protocol["items"][0]
        self.assertTrue(independent_quality(red,(RED,EOS)))
        self.assertTrue(independent_quality(red,(RED,FANCY,EOS)))
        for path in ((RED,), (BLUE,EOS),(RED,PLAIN,FANCY,EOS)):
            self.assertFalse(independent_quality(red,path))

    def test_inputs_do_not_use_source_identity_or_quality_reference(self):
        item=self.protocol["items"][0]
        original=prompt_ids(item,self.protocol)
        changed=dict(item,id="changed",source_group_id="changed",color="blue")
        self.assertEqual(original,prompt_ids(changed,self.protocol))
        self.assertEqual(len(set(tuple(prompt_ids(i,self.protocol)) for i in self.protocol["items"])),10)

    def test_exact_oracle_is_independent_terminal_path_enumeration(self):
        actor,_=make_models(2001,self.protocol);item=self.protocol["items"][0]
        values,_=oracle_tree(actor,[item],self.protocol,self.bundles["balanced"])
        total=0.;mass=0.
        for path in terminal_paths():
            probability=1.
            for t,action in enumerate(path):
                ids,mask=pad_states([item],[path[:t]],self.protocol)
                with torch.no_grad():p=actor(ids,mask)[0,list(allowed(t))].softmax(-1)
                probability*=float(p[allowed(t).index(action)])
            total+=probability*self.bundles["balanced"]["table"][(item["id"],path)]["proxy"];mass+=probability
        self.assertAlmostEqual(mass,1.,places=12);self.assertAlmostEqual(total,values[(item["id"],())],places=12)

    def test_actual_sampling_shift_support_eos_and_frozen_reward(self):
        actor,_=make_models(2002,self.protocol);reference=deepcopy(actor).requires_grad_(False)
        owners=[self.protocol["items"][0]]*64
        sampled=rollout(actor,owners,self.protocol,generator=torch.Generator().manual_seed(90))
        current,kl=path_statistics(actor,reference,owners,sampled["tokens"],sampled["mask"],self.protocol)
        torch.testing.assert_close(current*sampled["mask"],sampled["old_logp"],atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(kl,torch.zeros_like(kl),atol=1e-12,rtol=1e-12)
        stopped=sampled["tokens"][:,1]==EOS
        self.assertTrue(bool(stopped.any()));self.assertTrue(bool((~sampled["mask"][stopped,2]).all()))
        self.assertTrue(all(self.fits[a]["reload_exact"] for a in self.fits))
        for bundle in self.bundles.values():self.assertTrue(all(not p.requires_grad and p.grad is None for p in bundle["frozen"].model.parameters()))

    def test_actual_actor_and_learned_value_update_with_raw_ledger(self):
        result=run_arm(self.protocol,self.bundles["balanced"],2001,"learned",updates=2)
        self.assertNotEqual(result["actor_initial_sha256"],result["actor_final_sha256"])
        self.assertEqual(result["actor_initial_sha256"],result["reference_sha256"])
        self.assertEqual(result["critic_updates"],2)
        self.assertFalse(result["history"][0]["actor_loss_reaches_critic_parameters"])
        self.assertGreater(result["history"][0]["critic_gradient"],0.)
        self.assertNotEqual(result["critic_initial_sha256"],result["critic_final_sha256"])
        for field in ("first_actor_gradient_reach","first_critic_gradient_reach"):
            for key in ("embedding.weight","qkv.weight","head.weight"):
                self.assertGreater(result["history"][0][field][key],0.)
        self.assertEqual(len(result["training_ledger"]),2)
        self.assertEqual(result["panels"][0]["rows"],result["frozen_control"]["rows"])
        self.assertLess(max(h["alignment_max"] for h in result["history"]),1e-12)

    def test_actual_seed_repeat_and_different_optimizer_work_clocks(self):
        a=run_arm(self.protocol,self.bundles["balanced"],2003,"oracle",updates=1)
        b=run_arm(self.protocol,self.bundles["balanced"],2003,"oracle",updates=1)
        self.assertEqual(a["actor_final_sha256"],b["actor_final_sha256"])
        self.assertEqual(a["training_ledger"],b["training_ledger"])
        self.assertEqual(a["critic_updates"],0);self.assertEqual(a["actor_updates"],1)

    def test_neural_critic_loss_cannot_reach_actor_or_reward(self):
        actor,critic=make_models(2001,self.protocol)
        ids,mask=pad_states([self.protocol["items"][0]],[()],self.protocol)
        loss=critic_loss(critic(ids,mask)[:,None],torch.ones(1,1,dtype=torch.float64,requires_grad=True),torch.ones(1,1,dtype=torch.bool))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in actor.parameters()))
        self.assertGreater(float(critic.head.weight.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in self.bundles["balanced"]["frozen"].model.parameters()))

    def test_exact_delivered_quality_retains_caps_in_denominator(self):
        from dongxi_llms.critic_policy_lab import evaluate
        actor,_=make_models(2003,self.protocol)
        three=evaluate(actor,self.protocol["items"],self.protocol,self.bundles["balanced"],2003)
        four=evaluate(actor,self.protocol["items"],self.protocol,self.bundles["balanced"],2003,cap=4)
        for a,b in zip(three["exact_finite_diagnostics"],four["exact_finite_diagnostics"]):
            self.assertAlmostEqual(a["expected_delivered_quality"],b["expected_delivered_quality"])
            self.assertGreater(b["termination_probability"],a["termination_probability"])
            self.assertAlmostEqual(b["termination_probability"],1.)

    def test_pre_fit_actual_actor_encoding_rejects_whitespace_alias(self):
        changed=deepcopy(self.protocol)
        changed["items"][6]["prompt"]=changed["items"][0]["prompt"].replace(" ","  ")
        self.assertNotEqual(changed["items"][6]["prompt"],changed["items"][0]["prompt"])
        with self.assertRaisesRegex(ValueError,"input collision"):validate_protocol(changed)

    def test_pre_fit_vocabulary_rejects_duplicate_and_empty_tokens(self):
        for token in ("cup",""):
            changed=deepcopy(self.protocol);changed["actor_vocabulary"][7]=token
            with self.assertRaisesRegex(ValueError,"Unique nonempty"):validate_protocol(changed)


if __name__=="__main__":unittest.main()
