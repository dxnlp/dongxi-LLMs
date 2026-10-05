"""Original prefix/context/divergence/cost boundaries, not pretrained tests."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dongxi_llms.student_prefix_lab import (A,B,BOS,C,CAP,EOS,HINT,REVERSE,SEEDS,VOCAB,
    aligned_kl,attach_digest,collect_attempt,correct_body,distribution_controls,fit_arm,
    freeze_contract,grade,kl_probabilities,main,make_student,question,score_states,
    state_batch,state_digest,states_from_pool,teacher_distribution,topk_tail,
    validate_contract,validate_fixture,validate_probabilities)

ROOT=Path(__file__).resolve().parents[1]


class StudentPrefixTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.fixture=json.loads((ROOT/'fixtures/student-prefix/items.json').read_text())
        self.items=self.fixture['items'];self.contract=freeze_contract(self.fixture,'f'*64);self.item=self.items[0]

    def pool(self):
        return [collect_attempt(None,i,self.contract,campaign=SEEDS[0],phase='offline',update=0,
                 ordinal=o,arm='finite-offline') for i in self.items if i['split']=='train' for o in range(4)]

    def rehash(self,row):return attach_digest({k:v for k,v in row.items() if k!='payload_sha256'})

    def test_fixture_independent_groups_pairs_and_authored_refs(self):
        self.assertEqual(validate_fixture(self.fixture)['splits'],{'train':6,'test':3})
        for key,value in (('source_group',self.item['source_group']),('symbols',self.item['symbols']),('reference',['B','A'])):
            changed=deepcopy(self.fixture);changed['items'][6 if key!='reference' else 0][key]=value
            with self.assertRaises(ValueError):validate_fixture(changed)

    def test_rehashed_contract_permuted_vocab_and_recipe_rejected(self):
        for key,value in (('updates',31),('interface_sha256','0'*64),('gradient','occupancy gradient')):
            c=deepcopy(self.contract);c[key]=value
            from dongxi_llms.student_prefix_lab import canonical_hash
            c['identity']=canonical_hash({k:v for k,v in c.items() if k!='identity'})
            with self.assertRaises(ValueError):validate_contract(c,self.fixture)
        c=deepcopy(self.contract);c['vocabulary']['A']=6
        from dongxi_llms.student_prefix_lab import canonical_hash
        c['identity']=canonical_hash({k:v for k,v in c.items() if k!='identity'})
        with self.assertRaises(ValueError):validate_contract(c,self.fixture)
        own=freeze_contract(self.fixture,'e'*64);own['vocabulary']['A']=99
        self.assertEqual(VOCAB['A'],A)

    def test_teacher_context_changes_only_wrong_prefix_target(self):
        clean=teacher_distribution([A,B],[])
        hinted=teacher_distribution([A,B],[],hint=[B,A])
        self.assertTrue(torch.equal(clean,hinted));self.assertEqual(float(clean.sum()),1.)
        self.assertEqual(int(teacher_distribution([A,B],[C]).argmax()),C)
        self.assertEqual(int(teacher_distribution([A,B],[C],hint=[B,A]).argmax()),A)
        self.assertEqual(int(teacher_distribution([A,B],[C,A],hint=[B,A]).argmax()),EOS)
        self.assertTrue(bool((clean>0).all()))

    def test_teacher_rejects_post_eos_unknown_bool_and_bad_hint(self):
        for past in ([EOS],[99],[True],[A,B,C]):
            with self.assertRaises(ValueError):teacher_distribution([A,B],past)
        for hint in ([A,B],'BA',[True,A]):
            with self.assertRaises(ValueError):teacher_distribution([A,B],[],hint=hint)

    def test_full_mass_nonnegative_and_finite_validation(self):
        for p in (torch.tensor([]),torch.tensor([.2,.2]),torch.tensor([1.1,-.1]),torch.tensor([float('nan'),0.])):
            with self.assertRaises(ValueError):validate_probabilities(p)
        with self.assertRaises(ValueError):kl_probabilities(torch.tensor([1.,0.]),torch.tensor([1.,0.,0.]),'forward')

    def test_both_kl_directions_detach_teacher_and_align_ids(self):
        logits=torch.tensor([[.2,-.3,.6,.1,-.1,.4,.7,-.2]],dtype=torch.float64,requires_grad=True)
        q=teacher_distribution([A,B],[])[None].requires_grad_()
        grads=[]
        for direction in ('forward','reverse'):
            loss=aligned_kl(logits,q,direction).mean()
            grads.append(torch.autograd.grad(loss,logits,retain_graph=True)[0]);loss.backward(retain_graph=True)
            self.assertIsNone(q.grad)
        self.assertFalse(torch.allclose(*grads))
        changed=deepcopy(VOCAB);changed['A'],changed['B']=changed['B'],changed['A']
        with self.assertRaises(ValueError):aligned_kl(logits,q,'forward',teacher_vocab=changed)

    def test_forward_logits_gradient_exact_p_minus_q(self):
        logits=torch.zeros((1,8),dtype=torch.float64,requires_grad=True);q=teacher_distribution([A,B],[])[None]
        aligned_kl(logits,q,'forward').sum().backward()
        self.assertTrue(torch.allclose(logits.grad,logits.detach().softmax(-1)-q,atol=1e-15))

    def test_reverse_gradient_matches_independent_formula(self):
        z=torch.tensor([.1,.3,.6,-.4,.2,.7,-.2,.4],dtype=torch.float64,requires_grad=True);q=teacher_distribution([A,B],[])
        loss=aligned_kl(z,q,'reverse');actual=torch.autograd.grad(loss,z)[0];p=z.detach().softmax(-1)
        gap=p.log()-q.log();expected=p*(gap-(p*gap).sum())
        self.assertTrue(torch.allclose(actual,expected,atol=1e-15))

    def test_topk_tail_mass_decomposition_and_bound_both_directions(self):
        p=torch.tensor(self.fixture['tail_control']['p'],dtype=torch.float64);q=torch.tensor(self.fixture['tail_control']['q'],dtype=torch.float64)
        for direction in ('forward','reverse'):
            for k in (1,2,4,7,8):
                r=topk_tail(p,q,k,direction)
                self.assertAlmostEqual(float(r['student_bucket'].sum()),1.,places=14)
                self.assertAlmostEqual(float(r['teacher_bucket'].sum()),1.,places=14)
                self.assertAlmostEqual(float(r['full']),float(r['bucket']+r['lost_detail']),places=13)
                self.assertLessEqual(float(r['bucket']),float(r['full'])+1e-14)

    def test_equal_tail_total_hides_actual_detail_and_gradient(self):
        p=torch.tensor(self.fixture['tail_control']['p'],dtype=torch.float64,requires_grad=True);q=torch.tensor(self.fixture['tail_control']['q'],dtype=torch.float64)
        r=topk_tail(p,q,2,'forward');self.assertAlmostEqual(float(r['bucket'].detach()),0.,places=15)
        self.assertGreater(float(r['full'].detach()),.1)
        # Compare logit gradients, which respect the probability simplex.
        z=p.detach().log().requires_grad_();r=topk_tail(z.softmax(-1),q,2,'forward')
        full=torch.autograd.grad(r['full'],z,retain_graph=True)[0];bucket=torch.autograd.grad(r['bucket'],z)[0]
        self.assertTrue(torch.allclose(bucket,torch.zeros_like(bucket),atol=1e-15));self.assertGreater(float(full.abs().max()),.01)

    def test_full_k_recovers_loss_gradient_without_zero_tail_nan(self):
        z=torch.randn(8,dtype=torch.float64,requires_grad=True);q=teacher_distribution([A,B],[])
        for direction in ('forward','reverse'):
            r=topk_tail(z.softmax(-1),q,8,direction)
            g=torch.autograd.grad(r['bucket'],z,retain_graph=True)[0];full=torch.autograd.grad(r['full'],z,retain_graph=True)[0]
            self.assertTrue(bool(torch.isfinite(g).all()));self.assertTrue(torch.allclose(g,full,atol=1e-15))
        for k in (0,9,True):
            with self.assertRaises(ValueError):topk_tail(z.softmax(-1),q,k,'forward')

    def test_zero_support_control_explicit_infinity_not_numeric_json(self):
        c=distribution_controls(self.fixture)
        self.assertAlmostEqual(c['zero_support']['forward'],torch.tensor(2.).log().item(),places=6)
        self.assertEqual(c['zero_support']['reverse'],'positive-infinity')
        self.assertIsNone(c['teacher_target_gradients']);json.dumps(c,allow_nan=False)

    def test_zero_support_tail_detail_rejected_without_inf_minus_inf(self):
        pairs=([1.,0.],[0.,1.]),([.5,.5,0.],[1.,0.,0.]),([1.,0.],[1.,0.])
        for p,q in pairs:
            p=torch.tensor(p,dtype=torch.float64);q=torch.tensor(q,dtype=torch.float64)
            for direction in ('forward','reverse'):
                for k in (1,len(p)):
                    with self.assertRaisesRegex(ValueError,'Strictly positive'):
                        topk_tail(p,q,k,direction)
        # Full KL retains extended-real support semantics independently.
        self.assertTrue(torch.isinf(kl_probabilities(torch.tensor([1.,0.]),torch.tensor([0.,1.]),'forward')))

    def test_offline_is_finite_sampling_neural_prefix_is_actual_generation(self):
        finite=self.pool()[0];self.assertIn('finite',finite['provenance']);self.assertEqual(finite['cost']['attempted_forward_positions'],0)
        model=make_student(SEEDS[0]);r=collect_attempt(model,self.item,self.contract,campaign=SEEDS[0],phase='student-prefix',update=0,ordinal=0)
        self.assertIn('actual neural',r['provenance']);self.assertEqual(r['cost']['completed_forward_positions'],sum(4+t for t in range(len(r['token_ids']))))
        self.assertEqual(r['cost']['finite_teacher_queries'],0)

    def test_sampling_coordinates_pair_arms_without_forcing_future_states(self):
        model=make_student(SEEDS[0]);kw=dict(campaign=SEEDS[0],phase='student-prefix',update=0,ordinal=0)
        a=collect_attempt(model,self.item,self.contract,arm='forward/hint',**kw);b=collect_attempt(model,self.item,self.contract,arm='reverse/no-hint',**kw)
        self.assertEqual(a['attempt_seed'],b['attempt_seed']);self.assertEqual(a['token_ids'],b['token_ids']);self.assertNotEqual(a['sample_id'],b['sample_id'])
        e=collect_attempt(model,self.item,self.contract,campaign=SEEDS[0],phase='evaluation',update=0,ordinal=0)
        self.assertNotEqual(e['attempt_seed'],a['attempt_seed'])

    def test_evaluation_input_never_contains_privileged_hint_or_reference(self):
        class Spy:
            def __init__(self):self.inputs=[]
            def state_dict(self):return {'spy':torch.tensor([1.],dtype=torch.float64)}
            def __call__(self,ids):
                self.inputs.append(ids.tolist());z=torch.full((*ids.shape,8),-1000.,dtype=torch.float64);z[...,EOS]=0.;return z
        spy=Spy();changed=deepcopy(self.item);changed['teacher_hint']=['C','C'];changed['reference']=['C','C']
        r=collect_attempt(spy,changed,self.contract,campaign=SEEDS[0],phase='evaluation',update=0,ordinal=0)
        self.assertEqual(spy.inputs,[[question(self.item)]]);self.assertNotIn(HINT,spy.inputs[0][0]);self.assertFalse(r['teacher_context_in_student_input'])

    def test_eos_and_cap_states_not_post_terminal_and_no_eos_invention(self):
        pool=self.pool();states,audit=states_from_pool(pool,self.items,self.contract)
        self.assertEqual(len(states),sum(len(r['token_ids']) for r in pool))
        self.assertTrue(all(EOS not in s['input_ids'][4:] for s in states))
        capped=next((r for r in pool if r['stop']=='cap'),None)
        if capped:self.assertEqual(len([s for s in states if s['sample_id']==capped['sample_id']]),3)
        self.assertTrue(all(not g['actual_skipped'] for g in audit))

    def test_source_duplicate_payload_and_wrong_mapping_rejected(self):
        pool=self.pool()
        with self.assertRaises(ValueError):states_from_pool(pool+[pool[0]],self.items,self.contract)
        for key,value in (('source_group','other'),('interface_sha256','0'*64),('raw_response','changed'),('stop','unknown')):
            changed=deepcopy(pool);changed[0][key]=value;changed[0]=self.rehash(changed[0])
            with self.assertRaises(ValueError):states_from_pool(changed,self.items,self.contract)
        changed=deepcopy(pool);changed[0]['token_ids'][0]=C
        with self.assertRaises(ValueError):states_from_pool(changed,self.items,self.contract)

    def test_genuine_mixed_campaign_and_phase_cohorts_rejected(self):
        pool=self.pool()
        replacement=collect_attempt(None,self.item,self.contract,campaign=SEEDS[1],phase='offline',update=0,ordinal=0,arm='finite-offline')
        with self.assertRaisesRegex(ValueError,'Mixed'):
            states_from_pool([replacement]+pool[1:],self.items,self.contract)
        replacement=collect_attempt(make_student(SEEDS[0]),self.item,self.contract,campaign=SEEDS[0],phase='student-prefix',update=0,ordinal=0,arm='student-prefix/forward/hint')
        with self.assertRaisesRegex(ValueError,'Mixed'):
            states_from_pool([replacement]+pool[1:],self.items,self.contract)

    def test_genuine_mixed_student_update_arm_and_checkpoint_rejected(self):
        model=make_student(SEEDS[0]);arm='student-prefix/forward/hint'
        pool=[collect_attempt(model,i,self.contract,campaign=SEEDS[0],phase='student-prefix',update=0,ordinal=o,arm=arm)
              for i in self.items if i['split']=='train' for o in range(4)]
        for replacement_model,update,role in ((model,1,arm),(model,0,'student-prefix/reverse/hint'),(make_student(SEEDS[1]),0,arm)):
            replacement=collect_attempt(replacement_model,self.item,self.contract,campaign=SEEDS[0],phase='student-prefix',update=update,ordinal=0,arm=role)
            with self.assertRaisesRegex(ValueError,'Mixed'):
                states_from_pool([replacement]+pool[1:],self.items,self.contract)
        self.assertTrue(states_from_pool(pool,self.items,self.contract)[0])

    def test_uniform_wrong_offline_role_and_student_state_rejected(self):
        from dongxi_llms.student_prefix_lab import canonical_hash,derived_seed
        for key,value in (('update',1),('arm','initial'),('state_sha256','0'*64)):
            pool=self.pool()
            for row in pool:
                row[key]=value
                coordinate=[row['campaign'],row['phase'],row['update'],row['item_id'],row['ordinal']]
                row['attempt_seed']=derived_seed(*coordinate);row['sample_id']=canonical_hash([*coordinate,row['arm'],row['state_sha256']])
            with self.assertRaisesRegex(ValueError,'Offline cohort'):
                states_from_pool([self.rehash(r) for r in pool],self.items,self.contract)
        model=make_student(SEEDS[0]);arm='student-prefix/forward/hint'
        for role,state in (('initial','0'*64),(arm,'not-a-digest')):
            pool=[collect_attempt(model,i,self.contract,campaign=SEEDS[0],phase='student-prefix',update=0,ordinal=o,arm=role,checkpoint=state)
                  for i in self.items if i['split']=='train' for o in range(4)]
            with self.assertRaisesRegex(ValueError,'declared arm'):
                states_from_pool(pool,self.items,self.contract)

    def test_ordinary_error_skips_prompt_but_retains_all_attempt_cost(self):
        pool=self.pool();changed=deepcopy(pool);changed[0]['error']={'type':'RuntimeError','message':'scripted'};changed[0]['stop']='error';changed[0]=self.rehash(changed[0])
        states,audit=states_from_pool(changed,self.items,self.contract)
        self.assertEqual(sum(a['actual_skipped'] for a in audit),1);self.assertFalse(any(s['item_id']==self.item['id'] for s in states))
        self.assertEqual(len(changed),24);self.assertEqual(changed[0]['cost'],pool[0]['cost'])

    def test_partial_forward_failure_charges_attempted_not_completed_positions(self):
        class Broken:
            def __init__(self):self.calls=0
            def state_dict(self):return {'test':torch.tensor([1.],dtype=torch.float64)}
            def __call__(self,ids):
                self.calls+=1
                if self.calls==2:raise RuntimeError('scripted partial failure')
                z=torch.full((*ids.shape,8),-1000.,dtype=torch.float64);z[...,C]=0.;return z
        r=collect_attempt(Broken(),self.item,self.contract,campaign=SEEDS[0],phase='student-prefix',update=0,ordinal=0)
        self.assertEqual(r['token_ids'],[C]);self.assertEqual(r['stop'],'error')
        self.assertEqual(r['cost']['attempted_forward_positions'],9);self.assertEqual(r['cost']['completed_forward_positions'],4)

    def test_state_batch_padding_invariance_and_only_last_valid_loss(self):
        states,_=states_from_pool(self.pool(),self.items,self.contract);states=states[:3];model=make_student(SEEDS[0]);ids,last=state_batch(states)
        logits=model(ids);logits.retain_grad();q=torch.stack([teacher_distribution(s['input_ids'][2:4],s['input_ids'][4:],hint=s['input_ids'][2:4][::-1]) for s in states])
        loss=aligned_kl(logits[torch.arange(len(states)),last],q,'forward').mean();loss.backward()
        for i,position in enumerate(last):
            for t in range(ids.shape[1]):
                if t!=int(position):self.assertEqual(float(logits.grad[i,t].abs().sum()),0.)
        padded=torch.cat((ids,torch.zeros((len(states),1),dtype=torch.long)),-1)
        other=aligned_kl(model(padded)[torch.arange(len(states)),last],q,'forward').mean()
        self.assertAlmostEqual(float(loss.detach()),float(other.detach()),places=13)
        self.assertGreater(float(model.token.weight.grad.abs().sum()),0.)

    def test_same_init_actual_two_step_backbone_head_updates(self):
        initial=make_student(SEEDS[0]);offline=self.pool()
        with patch('dongxi_llms.student_prefix_lab.UPDATES',2):
            model,a=fit_arm(initial,self.items,self.contract,offline,campaign=SEEDS[0],source='offline',direction='forward',context='hint')
            _,b=fit_arm(initial,self.items,self.contract,offline,campaign=SEEDS[0],source='offline',direction='reverse',context='no-hint')
        self.assertEqual(a['initial_state_sha256'],b['initial_state_sha256']);self.assertNotEqual(state_digest(model),state_digest(initial))
        reach=a['history'][0]['first_gradient_reach']
        for piece in ('token','attn.q','mlp','lm_head'):self.assertTrue(any(piece in key and value>0 for key,value in reach.items()))
        self.assertEqual(a['collection_cost']['generated_actions'],0);self.assertGreater(a['teacher_target_queries'],0)

    def test_correct_last_symbol_is_not_correct_whole_response(self):
        row=self.pool()[0];row['token_ids']=[B,A,EOS];row['raw_response']='B A EOS';row['stop']='EOS';row['error']=None;row=self.rehash(row)
        g=grade(self.item,row);self.assertTrue(g['final_symbol_correct']);self.assertFalse(g['correct'])

    def test_cli_failure_and_existing_output_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'run';args=['--fixture',str(ROOT/'fixtures/student-prefix/items.json'),'--spec',str(ROOT/'experiments/specs/2026-10-04-student-prefix-distillation.md'),'--output',str(directory)]
            with patch('dongxi_llms.student_prefix_lab.run_reference',side_effect=RuntimeError('scripted fit failure')):
                with self.assertRaises(RuntimeError):main(args)
            self.assertTrue((directory/'failure.json').is_file())
            with self.assertRaises(SystemExit):main(args)


if __name__=='__main__':unittest.main()
