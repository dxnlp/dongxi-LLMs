"""Independent objective, split, boundary and generation controls; CPU only."""
from copy import deepcopy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from dongxi_llms import dpo_retention_lab as lab
from dongxi_llms.dpo_lab import dpo_loss


def manual_scores(model, data):
    """Independent aligned gather, not the implementation's sequence utility."""
    ids,attention,completion = data
    mask = attention[:,1:] & completion[:,1:]
    logits = model(ids[:,:-1])
    gathered = F.log_softmax(logits,-1).gather(-1,ids[:,1:,None]).squeeze(-1)
    return (gathered*mask).sum(-1)


class FixedToken(nn.Module):
    def __init__(self,c,token):
        super().__init__()
        self.cfg = lab.DecoderConfig(**c["model"])
        self.token_id = token

    def forward(self,ids):
        result = torch.zeros(*ids.shape,self.cfg.vocab)
        result[...,self.token_id] = 10
        return result


class RetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.c = lab.load_contract()

    def test_original_contract_and_independent_oracle(self):
        lab.validate_contract(self.c)
        self.assertEqual(self.c["seeds"],[1811,1812,1813])
        self.assertEqual(len(self.c["items"]),16)
        for a in range(4):
            for b in range(4):
                self.assertEqual(lab.oracle(dict(task="first",a=a,b=b)),6+a)
                self.assertEqual(lab.oracle(dict(task="parity",a=a,b=b)),14+((a+b)%2))
        for row in self.c["items"]:
            self.assertEqual(len(lab.prompt(row)),4)
            self.assertNotIn(row["id"],lab.prompt(row))

    def test_source_problem_and_encoding_leaks_rejected(self):
        for field in ("source_group","id"):
            c = deepcopy(self.c)
            c["items"][8][field] = c["items"][0][field]
            with self.assertRaises(ValueError): lab.validate_contract(c)
        c = deepcopy(self.c)
        c["items"][8].update(a=0,b=1,source_group="renamed-source")
        with self.assertRaises(ValueError): lab.validate_contract(c)

    def test_invalid_recipe_semantics_and_coefficients(self):
        for key,value in (("noise_indices",[0,1]),("conditions",["clean"]),
                          ("generation",dict(decoding="sampling",max_new_tokens=4,eos=2))):
            c = deepcopy(self.c); c[key] = value
            with self.assertRaises(ValueError): lab.validate_contract(c)
        for value in (-.1,float("nan")):
            c = deepcopy(self.c); c["arms"][0]["alpha"] = value
            with self.assertRaises(ValueError): lab.validate_contract(c)

    def test_fixed_noise_and_length_interventions(self):
        clean_c,clean_r,items = lab.pair_batches(self.c,"clean")
        noisy_c,noisy_r,_ = lab.pair_batches(self.c,"noisy")
        for i in range(4):
            self.assertTrue(torch.equal(noisy_c[0][i], (clean_r if i in [1,3] else clean_c)[0][i]))
            self.assertTrue(torch.equal(noisy_r[0][i], (clean_c if i in [1,3] else clean_r)[0][i]))
        for name,lengths in (("clean",(2,2)),("noisy",(2,2)),("chosen-longer",(4,2)),("matched-long",(4,4))):
            chosen,rejected,_ = lab.pair_batches(self.c,name)
            self.assertEqual(tuple(int(b[2].sum(1)[0]) for b in (chosen,rejected)),lengths)
            for i,item in enumerate(items):
                y = chosen[0][i][chosen[2][i]].tolist()
                self.assertEqual(y[-1],lab.EOS)
                if name in ("chosen-longer","matched-long"):
                    self.assertEqual(y,[lab.oracle(item),lab.STYLE,lab.STYLE,lab.EOS])

    def test_single_shift_eos_prompt_and_right_pad(self):
        model = lab.make_model(self.c,1811).double()
        items = self.c["items"][:2]
        data = lab.branch(items,[[6,2],[7,5,5,2]])
        actual = lab.scores(model,data)
        torch.testing.assert_close(actual,manual_scores(model,data),rtol=1e-13,atol=1e-13)
        self.assertEqual(data[2][0].tolist(),[False]*4+[True,True,False,False])
        shorter = lab.branch([items[0]],[[6,2]])
        torch.testing.assert_close(actual[:1],lab.scores(model,shorter),rtol=1e-12,atol=1e-12)
        invalid = (data[0],data[1].clone(),data[2])
        invalid[1][0,0] = False
        with self.assertRaises(ValueError): lab.scores(model,invalid)

    def test_alpha_gamma_zero_recovers_existing_loss_and_parameter_gradient_exactly(self):
        model = lab.make_model(self.c,1811).double()
        chosen,rejected,_ = lab.pair_batches(self.c,"chosen-longer")
        pc,pr = lab.scores(model,chosen),lab.scores(model,rejected)
        rc,rr = pc.detach().clone(),pr.detach().clone()
        original,_ = dpo_loss(pc,pr,rc,rr,.5)
        combined,_ = lab.combined_loss(pc,pr,rc,rr,chosen[2].sum(1),alpha=0,gamma=0,
                                       rehearsal_nll=torch.tensor(float("inf")),beta=.5)
        self.assertTrue(torch.equal(original,combined))
        params = tuple(model.parameters())
        first = torch.autograd.grad(original,params,retain_graph=True)
        second = torch.autograd.grad(combined,params)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(first,second)))

    def test_explicit_combined_parameter_gradients_and_detached_reference(self):
        model = lab.make_model(self.c,1812).double()
        chosen,rejected,_ = lab.pair_batches(self.c,"noisy")
        replay_items = [x for x in self.c["items"] if x["task"]=="parity" and x["split"]=="train"]
        replay = lab.branch(replay_items,[[lab.oracle(x),lab.EOS] for x in replay_items])
        pc,pr = lab.scores(model,chosen),lab.scores(model,rejected)
        rc,rr = pc.detach().clone().requires_grad_(),pr.detach().clone().requires_grad_()
        rn,_ = lab.response_nll(model,replay)
        actual,_ = lab.combined_loss(pc,pr,rc,rr,chosen[2].sum(1),alpha=.25,gamma=.25,rehearsal_nll=rn)
        independent_c,independent_r = manual_scores(model,chosen),manual_scores(model,rejected)
        margin = .5*(independent_c-independent_r-rc.detach()+rr.detach())
        independent_replay = -manual_scores(model,replay).sum()/replay[2].sum()
        expected = F.softplus(-margin).mean()+.25*(-independent_c.sum()/chosen[2].sum())+.25*independent_replay
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        params = tuple(model.parameters())
        ga = torch.autograd.grad(actual,params+(rc,rr),retain_graph=True,allow_unused=True)
        ge = torch.autograd.grad(expected,params)
        self.assertIsNone(ga[-1]); self.assertIsNone(ga[-2])
        for a,e in zip(ga[:-2],ge): torch.testing.assert_close(a,e,rtol=1e-11,atol=1e-12)

    def test_global_chosen_token_weighting_and_alpha_zero_with_rehearsal(self):
        pc = torch.tensor([-2.,-12.],dtype=torch.float64,requires_grad=True)
        zero = torch.zeros_like(pc); lengths = torch.tensor([2,4])
        loss,parts = lab.combined_loss(pc,zero,zero,zero,lengths,alpha=.4)
        self.assertAlmostEqual(float(parts["chosen_nll"].detach()),14/6)
        base,_ = dpo_loss(pc,zero,zero,zero,.5)
        g = torch.autograd.grad(loss-base,pc)[0]
        torch.testing.assert_close(g,torch.full_like(pc,-.4/6))
        replay = torch.tensor(3.,requires_grad=True,dtype=torch.float64)
        with_replay,_ = lab.combined_loss(pc,zero,zero,zero,lengths,alpha=0,gamma=.2,rehearsal_nll=replay)
        self.assertAlmostEqual(float((with_replay-base).detach()),.6)
        self.assertAlmostEqual(float(torch.autograd.grad(with_replay,replay)[0]),.2)

    def test_combined_loss_guard_and_empty_branch(self):
        p = torch.tensor([-2.,-3.]); n = torch.tensor([2,2])
        for kwargs in (dict(alpha=-1),dict(alpha=float("nan")),dict(beta=float("nan")),dict(beta=float("inf")),
                       dict(gamma=.1),dict(gamma=.1,rehearsal_nll=torch.ones(2))):
            with self.assertRaises(ValueError): lab.combined_loss(p,p,p,p,n,**kwargs)
        with self.assertRaises(ValueError): lab.combined_loss(p,p,p,p,n.float())
        with self.assertRaises(ValueError): lab.combined_loss(p*float("nan"),p,p,p,n)
        with self.assertRaises(ValueError): lab.branch([],[])
        with self.assertRaises(ValueError): lab.branch(self.c["items"][:1],[[6,2,5,2]])
        with self.assertRaises(ValueError): lab.branch(self.c["items"][:1],[[6,0,2]])

    def test_cache_identity_covers_weights_masks_template_and_dtype(self):
        model = lab.make_model(self.c,1811)
        chosen,rejected,_ = lab.pair_batches(self.c,"clean")
        with torch.no_grad(): rc,rr = lab.scores(model,chosen),lab.scores(model,rejected)
        base = lab.pair_cache_hash(model,self.c,chosen,rejected,rc,rr)
        modified = tuple(x.clone() for x in chosen); modified[2][0,4] = False
        self.assertNotEqual(base,lab.pair_cache_hash(model,self.c,modified,rejected,rc,rr))
        other = deepcopy(self.c); other["template"] += " changed"
        self.assertNotEqual(base,lab.pair_cache_hash(model,other,chosen,rejected,rc,rr))
        self.assertNotEqual(base,lab.pair_cache_hash(model,self.c,chosen,rejected,rc.double(),rr.double()))
        with torch.no_grad(): model.token.weight[0,0] += .01
        self.assertNotEqual(base,lab.pair_cache_hash(model,self.c,chosen,rejected,rc,rr))

    def test_sft_branch_enforces_same_padding_boundary(self):
        chosen,_,_ = lab.pair_batches(self.c,"clean")
        model = lab.make_model(self.c,1811)
        bad = tuple(x.clone() for x in chosen); bad[1][0,0] = False
        with self.assertRaises(ValueError): lab.response_nll(model,bad)
        empty = tuple(x.clone() for x in chosen); empty[2][0] = False
        with self.assertRaises(ValueError): lab.response_nll(model,empty)

    def test_input_pad_is_unscored_but_output_pad_still_competes(self):
        c = deepcopy(self.c); c["model"]["tied"] = False
        model = lab.make_model(c,1811)
        data = lab.branch(c["items"][:2],[[6,2],[7,5,5,2]])
        (-lab.scores(model,data).sum()).backward()
        self.assertEqual(float(model.token.weight.grad[lab.PAD].abs().sum()),0.)
        self.assertGreater(float(model.lm_head.weight.grad[lab.PAD].abs().sum()),0.)
        self.assertGreater(float(model.token.weight.grad[lab.FIRST].abs().sum()),0.)

    def test_actual_shared_model_fit_and_reference_cache(self):
        warm,_ = lab.warm_start(self.c,1811,updates=3)
        x = lab.fit_arm(warm,self.c,1811,"clean",self.c["arms"][3],updates=3)
        self.assertTrue(x["reference_unchanged"] and x["cache_equal"])
        self.assertFalse(x["reference_has_gradients"])
        self.assertNotEqual(x["initial_state_sha256"],x["final_state_sha256"])
        for prefix in ("token.weight","blocks.0.attn.q.weight","blocks.0.attn.v.weight","blocks.0.mlp.up.weight"):
            self.assertGreater(x["first_gradient_reach"][prefix],0.)
        self.assertEqual(x["costs"]["pair_score_response_tokens"],48)
        self.assertEqual(x["costs"]["chosen_aux_supervised_tokens"],24)
        self.assertEqual(x["costs"]["rehearsal_supervised_tokens"],24)
        self.assertEqual(len(x["final"]["rows"]),16)

    def test_seed_replay_and_no_heldout_model_selection(self):
        rng = torch.get_rng_state().clone()
        first,record = lab.warm_start(self.c,1811,updates=2)
        second,_ = lab.warm_start(self.c,1811,updates=2)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(lab.state_hash(first),lab.state_hash(second))
        changed = deepcopy(self.c); changed["items"][8]["b"] = 3
        other,_ = lab.warm_start(changed,1811,updates=2)
        original = lab.fit_arm(first,self.c,1811,"clean",self.c["arms"][0],updates=2)
        altered = lab.fit_arm(other,changed,1811,"clean",changed["arms"][0],updates=2)
        self.assertEqual(original["final_state_sha256"],altered["final_state_sha256"])
        self.assertEqual(original["history"],altered["history"])

    def test_actual_generation_eos_first_cap_and_cost_accounting(self):
        empty = lab.evaluate(FixedToken(self.c,lab.EOS),self.c,"fixed-eos")
        capped = lab.evaluate(FixedToken(self.c,lab.STYLE),self.c,"fixed-style")
        for row in empty["rows"]:
            self.assertEqual(row["token_ids"],[lab.EOS]); self.assertFalse(row["format_valid"])
            self.assertFalse(row["truncated"]); self.assertFalse(row["canonical_complete_success"])
            self.assertEqual(row["cost"]["generation_forwards"],1)
        for row in capped["rows"]:
            self.assertEqual(row["token_ids"],[lab.STYLE]*4); self.assertTrue(row["truncated"])
            self.assertEqual(row["cost"]["generation_tokens"],4)
        with self.assertRaises(ValueError): lab.generate(FixedToken(self.c,2),[1,3,6,7],cap=20)

    def test_complete_micro_matrix_is_marked_not_primary(self):
        result = lab.run_retention(self.c,seeds=[1811],updates=1,warmup_updates=1)
        self.assertFalse(result["primary_recipe"])
        self.assertTrue(result["all_predeclared_rows_retained"])
        self.assertEqual(len(result["runs"]),16)
        self.assertEqual({x["condition"] for x in result["runs"]},set(lab.CONDITIONS))
        self.assertEqual(torch.get_num_threads(),1)


if __name__ == "__main__":
    unittest.main()
