"""Independent CPU invariants for constructive controls, not pretrained evidence."""
from copy import deepcopy
import unittest

import torch

from dongxi_llms.reasoning_controls import (
    BOS, EOS, FIRST, LATER, NUMBER_START, SEEDS, UPDATES, LookupPolicy,
    answer_text, constant_baselines, decoder_control, decode, evaluate_policy, expected_success,
    frozen_copy, generate, load_fixtures, lookup_control, make_decoder,
    math_panel_audit, oracle_answer, path_log_probabilities, prompt_ids,
    state_hash, strict_path_reward, validate_items,
)


class ReasoningControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.control, cls.panel, cls.protocol = load_fixtures()

    def test_original_reference_and_split_contracts(self):
        for items in (self.control,self.panel):
            audit = validate_items(items)
            self.assertEqual(set(audit['by_split']),{'train','heldout-source','heldout-template','heldout-family'})
            train = [i for i in items if i['split']=='train']
            for item in items:
                self.assertTrue(answer_text(oracle_answer(item)))
                if item['split']=='heldout-family':
                    self.assertNotIn(item['family'],{i['family'] for i in train})
                if item['split']=='heldout-template':
                    self.assertNotIn(item['template_id'],{i['template_id'] for i in train})

    def test_same_problem_and_source_leaks_rejected(self):
        changed = deepcopy(self.control)
        changed[4]['source_group']=changed[0]['source_group']
        with self.assertRaisesRegex(ValueError,'Source group'):
            validate_items(changed)
        changed = deepcopy(self.control)
        changed[4]['problem']=deepcopy(changed[0]['problem'])
        with self.assertRaisesRegex(ValueError,'problem crosses'):
            validate_items(changed)

    def test_template_family_reference_tampering_rejected(self):
        for field,index,value in (('template_id',8,'predicate-symbolic'),
                                  ('family',12,'sum-positive'),('reference',0,'1')):
            changed=deepcopy(self.control)
            changed[index][field]=value
            with self.assertRaises(ValueError):
                validate_items(changed)

    def test_oracle_exact_algebra_and_multi_step_not_last_number(self):
        from fractions import Fraction
        self.assertEqual(oracle_answer({'family':'linear-algebra','problem':{'a':4,'b':3,'c':5}}),Fraction(1,2))
        self.assertEqual(oracle_answer({'family':'multi-step','problem':{'start':1,'receive':5,'groups':4}}),Fraction(3,2))
        with self.assertRaises(ValueError):
            oracle_answer({'family':'linear-algebra','problem':{'a':0,'b':3,'c':5}})

    def test_token_interface_has_no_reference_or_key_lookup(self):
        item=deepcopy(self.control[0])
        original=prompt_ids([item])
        item.update(reference='1',source_group='changed',id='changed',prompt='unrelated English')
        torch.testing.assert_close(original,prompt_ids([item]))
        with self.assertRaisesRegex(ValueError,'does not encode'):
            prompt_ids(self.panel[:1])

    def test_unsaturated_multiple_seed_decoder_and_rng_isolation(self):
        torch.manual_seed(87)
        before=torch.get_rng_state().clone()
        for seed in SEEDS:
            values=expected_success(make_decoder(seed),self.control[:4])
            self.assertTrue(bool(((values>0)&(values<.9)).all()))
        torch.testing.assert_close(before,torch.get_rng_state())

    def test_autoregressive_grammar_likelihood_alignment_and_eos(self):
        model=make_decoder(SEEDS[0])
        prompts=prompt_ids(self.control[:4]).repeat_interleave(16,0)
        responses,mask=generate(model,prompts,generator=torch.Generator().manual_seed(91))
        self.assertTrue(bool(torch.isin(responses[:,0],torch.tensor(FIRST)).all()))
        self.assertTrue(bool(torch.isin(responses[:,1],torch.tensor(LATER)).all()))
        self.assertTrue(bool(mask.all()))
        self.assertTrue(bool((responses[:,1]==EOS).any()))
        self.assertTrue(bool((responses[:,1]!=EOS).any()))
        measured=path_log_probabilities(model,prompts,responses)
        # Independent two-stage forward calculation, conditioned on actual first action.
        raw_first=model(prompts)[:,-1,list(FIRST)].log_softmax(-1)
        raw_second=model(torch.cat((prompts,responses[:,:1]),-1))[:,-1,list(LATER)].log_softmax(-1)
        first=raw_first.gather(-1,(responses[:,0]-NUMBER_START)[:,None]).squeeze(-1)
        second_index=torch.where(responses[:,1]==EOS,0,responses[:,1]-NUMBER_START+1)
        second=raw_second.gather(-1,second_index[:,None]).squeeze(-1)
        torch.testing.assert_close(measured,torch.stack((first,second),-1))

    def test_reward_and_grading_do_not_accept_cap_as_eos(self):
        self.assertEqual(strict_path_reward(torch.tensor([NUMBER_START,EOS]),torch.tensor([True,True]),0),1.)
        self.assertEqual(strict_path_reward(torch.tensor([NUMBER_START]),torch.tensor([True]),0),0.)
        self.assertEqual(strict_path_reward(torch.tensor([NUMBER_START,NUMBER_START]),torch.tensor([True,True]),0),0.)
        report=evaluate_policy(make_decoder(SEEDS[0]),self.control,checkpoint_id='original-random-test',seed=99,cap=1)
        self.assertTrue(all(r['truncated'] and not r['complete_path_success'] and not r['strict_reward'] for r in report['rows']))
        self.assertEqual(report['summaries']['train/greedy']['truncation'],1.)

    def test_exact_path_probability_independent_factorization(self):
        model=make_decoder(SEEDS[0])
        prompts=prompt_ids(self.control[:4])
        target=torch.tensor([int(oracle_answer(i))+NUMBER_START for i in self.control[:4]])
        p=model(prompts)[:,-1,list(FIRST)].softmax(-1).gather(-1,(target-NUMBER_START)[:,None]).squeeze(-1)
        after=model(torch.cat((prompts,target[:,None]),-1))[:,-1,list(LATER)].softmax(-1)[:,0]
        torch.testing.assert_close(expected_success(model,self.control[:4]),p*after)

    def test_actual_sampled_updates_reach_shared_layers_and_leave_reference_frozen(self):
        report=decoder_control(self.control,SEEDS[0],updates=4,group_size=16)
        self.assertNotEqual(report['initial_state_sha256'],report['final_state_sha256'])
        self.assertEqual(report['initial_state_sha256'],report['frozen_state_sha256'])
        first=report['history'][0]
        self.assertEqual(first['initial_log_ratio_max_abs'],0.)
        self.assertGreater(first['first_eos_output_gradient_abs_sum'],0.)
        reach=first['first_gradient_reach']
        for fragment in ('token.weight','attn.q.weight','attn.v.weight','mlp.up.weight','lm_head.weight'):
            self.assertGreater(sum(value for name,value in reach.items() if fragment in name),0.)
        self.assertTrue(all(torch.isfinite(torch.tensor(row['gradient_norm_before_clip'])) for row in report['history']))
        self.assertTrue(all(parameter.grad is None for parameter in frozen_copy(make_decoder(3)).parameters()))
        initial_tokens=[r['token_ids'] for r in report['initial']['rows']]
        self.assertEqual(initial_tokens,[r['token_ids'] for r in report['frozen']['rows']])

    def test_replay_is_seed_deterministic_not_selected_after_test(self):
        a=decoder_control(self.control,SEEDS[1],updates=2,group_size=4)
        b=decoder_control(self.control,SEEDS[1],updates=2,group_size=4)
        self.assertEqual(a['final_state_sha256'],b['final_state_sha256'])
        self.assertEqual(a['first_and_last_rollout_groups'],b['first_and_last_rollout_groups'])
        self.assertEqual(UPDATES,120)

    def test_lookup_unknown_keys_do_not_learn_from_test_references(self):
        results=lookup_control(self.control)
        torch.testing.assert_close(torch.tensor(results['unknown_key_probability']),torch.tensor([.5,.5]))
        for row in results['rows']:
            if row['split']=='train':
                self.assertGreater(row['final_correct_probability'],row['initial_correct_probability'])
            else:
                self.assertEqual(row['initial_correct_probability'],row['final_correct_probability'])

    def test_constant_baseline_exposes_heldout_class_imbalance(self):
        always_one=constant_baselines(self.control)[1]
        self.assertEqual(always_one['by_split'],{'train':.75,'heldout-source':1.,
                                              'heldout-template':1.,'heldout-family':.5})

    def test_full_support_generation_masks_post_eos_padding(self):
        model=make_decoder(SEEDS[0])
        prompts=prompt_ids(self.control[:4]).repeat_interleave(32,0)
        output,mask=generate(model,prompts,grammar=False,generator=torch.Generator().manual_seed(97))
        first_stop=output[:,0]==EOS
        self.assertTrue(bool(first_stop.any()))
        self.assertTrue(bool((~mask[first_stop,1]).all()))
        self.assertTrue(bool((output[first_stop,1]==0).all()))

    def test_math_parser_probe_and_pretrained_protocol_boundaries(self):
        audit=math_panel_audit(self.panel)
        self.assertTrue(all(r['correct'] and r['complete_success'] for r in audit['rows'] if r['variant']=='oracle'))
        self.assertTrue(all(not r['correct'] for r in audit['rows'] if r['variant']=='wrong'))
        self.assertTrue(all(not r['complete_success'] for r in audit['rows'] if r['variant'] in ('invalid','truncated')))
        self.assertTrue(all(r['correct'] and r['truncated'] for r in audit['rows'] if r['variant']=='truncated'))
        self.assertEqual(self.protocol['status'],'unexecuted-pretrained-protocol')
        self.assertTrue(all(r['status']=='unexecuted' for r in self.protocol['rows']))
        self.assertNotEqual(self.protocol['rows'][0]['checkpoint_id'],self.protocol['rows'][1]['checkpoint_id'])
        self.assertEqual(self.protocol['rows'][1]['checkpoint_id'],self.protocol['rows'][2]['checkpoint_id'])


if __name__ == '__main__':
    unittest.main()
