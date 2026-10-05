"""Independent selection identities, causal sampling and actual CPU work contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dongxi_llms.inference_selection_lab import (CAP, EOS, NUMBER_START, SUPPORT,
    budget_prefix, coordinate_seed, cost_totals, decode, eligible_answer,
    error_dependence, evaluate_pool, fit_policy, freeze_contract, generate_candidate,
    main, make_decoder, prompt_ids, select, selection_view, state_hash, validate_items)

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return json.loads((ROOT/'fixtures/inference-selection/items.json').read_text())


def controlled(ordinal, answer='0', *, tokens=2, score=-1., error=None):
    return dict(schema_version='dongxi-response-record-v1', contract_id='x', checkpoint_id='m',
        item_id='i', source_group='g',task='parity',split='test',sample_id=f'c{ordinal}',ordinal=ordinal,
        raw_response=answer+' <eos>',response_text=answer, token_ids=[NUMBER_START,EOS],
        generated_tokens=tokens,prompt_tokens=4,stop_reason='error' if error else 'eos',
        truncated=False,error=error,cost={'wall_seconds':.01,'generation_tokens':tokens,'scoring_tokens':tokens},
        rescored_log_probabilities=[score]*tokens)


class SelectionTests(unittest.TestCase):
    def test_projection_discards_gold_and_is_detached(self):
        r=controlled(0);r.update(reference='0',correct=True,task_success=True,problem={'a':0},
                                graded_status='CORRECT',rubric={'gold':'0'})
        v=selection_view(r)
        self.assertFalse({'reference','correct','task_success','problem','graded_status','rubric'} & v.keys())
        v['cost']['generation_tokens']=50
        self.assertEqual(r['cost']['generation_tokens'],2)

    def test_gold_injection_does_not_change_any_selector(self):
        pool=[controlled(0,'0',score=-2),controlled(1,'1',score=-.1),controlled(2,'0',score=-1)]
        injected=deepcopy(pool)
        for r in injected:r.update(reference='1',correct=r['response_text']=='1')
        for method in ('first','majority','unique_support','mean_logp','longest'):
            self.assertEqual(select(pool,method),select(injected,method))

    def test_frequency_and_unique_support_have_distinct_stable_ties(self):
        pool=[controlled(0,'1'),controlled(1,'0'),controlled(2,'0')]
        self.assertEqual(select(pool,'majority')['selected_sample_id'],'c1')
        self.assertEqual(select(pool,'unique_support')['selected_sample_id'],'c0')
        self.assertTrue(select(pool,'unique_support')['tie'])
        self.assertEqual(select(pool[:2],'majority')['selected_sample_id'],'c0')

    def test_independent_duplicate_strings_vote_but_duplicate_ids_fail(self):
        pool=[controlled(0,'1'),controlled(1,'0'),controlled(2,'0')]
        self.assertEqual(select(pool,'majority')['votes'],{'1':1,'0':2})
        pool[2]['sample_id']=pool[1]['sample_id']
        with self.assertRaises(ValueError):select(pool,'majority')

    def test_invalid_cap_error_and_empty_are_not_rescued(self):
        pool=[controlled(0,''),controlled(1,'0 1'),controlled(2,'0',error='scripted failure')]
        for row in pool:self.assertIsNone(eligible_answer(row))
        self.assertIsNone(select(pool,'majority')['selected_sample_id'])
        self.assertEqual(select(pool,'first')['selected_sample_id'],'c0')
        self.assertIsNone(select([],'first')['selected_sample_id'])
        row=controlled(3);row.update(stop_reason='max_tokens',truncated=True)
        self.assertIsNone(eligible_answer(row))

    def test_ranker_needs_actual_finite_eos_inclusive_scores(self):
        pool=[controlled(0,score=-2),controlled(1,'1',score=-.1)]
        self.assertEqual(select(pool,'mean_logp')['selected_sample_id'],'c1')
        for scores in ([],[float('nan'),-.1],[-.1]):
            pool[0]['rescored_log_probabilities']=scores
            with self.assertRaises(ValueError):select(pool,'mean_logp')

    def test_pool_mixed_contexts_ordinals_and_budget_types_fail(self):
        for key,value in (('item_id','j'),('checkpoint_id','n'),('contract_id','y'),('ordinal',0)):
            pool=[controlled(0),controlled(1)];pool[1][key]=value
            with self.assertRaises(ValueError):select(pool,'first')
        for args in ({},{'n':True},{'n':0},{'n':2,'tokens':3},{'tokens':2.5}):
            with self.assertRaises(ValueError):budget_prefix([controlled(0)],**args)

    def test_token_boundary_attempt_is_charged_but_not_eligible(self):
        pool=[controlled(0,tokens=2),controlled(1,tokens=3),controlled(2,tokens=1)]
        chosen,attempts,rejections=budget_prefix(pool,tokens=4)
        self.assertEqual([r['sample_id'] for r in chosen],['c0'])
        self.assertEqual([r['sample_id'] for r in attempts],['c0','c1'])
        self.assertEqual(rejections,[{'sample_id':'c1','reason':'whole-attempt-exceeds-remaining-budget',
                                      'generated_tokens':3,'remaining':2}])
        self.assertEqual(cost_totals(attempts)['generation_tokens']['known_total'],5)
        exact=budget_prefix(pool,tokens=5)
        self.assertEqual(len(exact[0]),2);self.assertEqual(exact[2],[])

    def test_attempt_prefix_and_unknown_replay_costs_are_explicit(self):
        pool=[controlled(i) for i in range(8)]
        for n in (1,2,4,8):self.assertEqual(budget_prefix(pool,n=n)[0],pool[:n])
        pool[0]['cost']={'wall_seconds':None,'generation_tokens':None,'scoring_tokens':None}
        self.assertEqual(cost_totals(pool[:1])['wall_seconds'],{'known_total':0,'unknown_rows':1})
        pool[0]['generated_tokens']=None
        with self.assertRaises(ValueError):budget_prefix(pool,tokens=3)

    def test_oracle_separated_from_wrong_nonoracle_selection(self):
        item=fixture()[0];pool=[controlled(0,'1'),controlled(1,'0'),controlled(2,'1')]
        for row in pool:row.update(item_id=item['id'],source_group=item['source_group'],split='train')
        result=evaluate_pool(item,pool)
        row=next(r for r in result['decisions'] if r['budget_kind']=='attempts' and r['budget']==4 and r['method']=='majority')
        self.assertTrue(row['oracle_any_complete']);self.assertFalse(row['selected_complete_success'])

    def test_error_covariance_has_known_population_values(self):
        pools=[{'slice':'train','source_group':f'g{i}','graded_candidates':[{'complete_success':v} for v in values]}
               for i,values in enumerate(([True,True],[True,False],[False,True],[False,False]))]
        d=error_dependence(pools)['all']
        self.assertEqual(d['ordinal_error_rates'],[.5,.5]);self.assertEqual(d['population_covariance'],[[.25,0.],[0.,.25]])
        self.assertEqual(d['correlation'],[[1.,0.],[0.,1.]])


class ActualSamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.items=fixture()
        self.contract=freeze_contract(self.items,dict(template_id='test-symbols',thinking_mode='none',
            decoding='test-sampling',stopping=[EOS],max_new_tokens=CAP))

    def test_fixture_source_problem_and_encoding_gates(self):
        audit=validate_items(self.items);self.assertEqual(audit['train_answers'],{'0':3,'1':3})
        copied=deepcopy(self.items);copied[6]['source_group']=copied[0]['source_group']
        with self.assertRaises(ValueError):validate_items(copied)
        copied=deepcopy(self.items);copied[6]['problem']=deepcopy(copied[0]['problem']);copied[6]['reference']='0'
        with self.assertRaises(ValueError):validate_items(copied)
        copied=deepcopy(self.items);copied[0]['reference']='1'
        with self.assertRaises(ValueError):validate_items(copied)

    def test_symbolic_alias_and_family_are_not_english_parsing(self):
        self.assertNotEqual(prompt_ids(self.items[6]),prompt_ids(self.items[10]))
        self.assertEqual(self.items[6]['source_group'],self.items[10]['source_group'])
        self.assertNotEqual(prompt_ids(self.items[0]),prompt_ids(self.items[14]))

    def test_coordinate_is_order_independent_and_prefix_reusable(self):
        self.assertEqual(coordinate_seed(3,'policy','i',0),coordinate_seed(3,'policy','i',0))
        self.assertNotEqual(coordinate_seed(3,'policy','i',0),coordinate_seed(3,'policy','i',1))
        self.assertNotEqual(coordinate_seed(3,'policy','i',0),coordinate_seed(3,'other','i',0))

    def test_actual_decoder_scores_and_work_are_recomputed(self):
        model=make_decoder(991).double().eval();before=state_hash(model)
        row=generate_candidate(model,self.items[0],self.contract,seed=991,policy_id='fixed',ordinal=0)
        self.assertIsNone(row['error']);self.assertEqual(before,state_hash(model))
        self.assertEqual(row['cost']['generation_tokens'],len(row['token_ids']))
        length=len(row['token_ids']);self.assertEqual(row['cost']['generation_forward_positions'],sum(4+i for i in range(length)))
        self.assertEqual(row['cost']['scoring_forward_positions'],4+length-1)
        self.assertTrue(torch.allclose(torch.tensor(row['selected_log_probabilities']),torch.tensor(row['rescored_log_probabilities'])))
        self.assertTrue(all(t in SUPPORT for t in row['token_ids']))
        if row['stop_reason']=='eos':self.assertEqual(row['token_ids'][-1],EOS)
        else:self.assertEqual(length,CAP)

    def test_sft_reaches_shared_backbone_and_head_not_heldout(self):
        a,b,e=fit_policy(self.items,997,updates=2)
        self.assertNotEqual(e['initial_state_sha256'],e['final_state_sha256'])
        self.assertTrue(all(i.startswith('train-') for i in e['train_item_ids']))
        reach=e['history'][0]['first_gradient_reach']
        for piece in ('token','q','mlp','lm_head'):
            self.assertTrue(any(piece in n and value>0 for n,value in reach.items()))
        self.assertTrue(all(not p.requires_grad for p in b.parameters()))
        changed=deepcopy(self.items);changed[6]['prompt']='Different English surface'
        a2,b2,e2=fit_policy(changed,997,updates=2)
        self.assertEqual(e,e2)

    def test_scripted_eos_not_forced_and_cap_is_retained(self):
        class Fixed:
            def __init__(self,token):self.token=token
            def __call__(self,ids):
                logits=torch.full((*ids.shape,10),-1000.,dtype=torch.float64);logits[...,self.token]=0.;return logits
        empty=generate_candidate(Fixed(EOS),self.items[0],self.contract,seed=1,policy_id='scripted-eos',ordinal=0)
        self.assertEqual(empty['token_ids'],[EOS]);self.assertEqual(empty['response_text'],'')
        repeat=generate_candidate(Fixed(NUMBER_START),self.items[0],self.contract,seed=1,policy_id='scripted-repeat',ordinal=0)
        self.assertEqual(repeat['token_ids'],[NUMBER_START]*3);self.assertTrue(repeat['truncated'])
        self.assertIsNone(eligible_answer(repeat))

    def test_forward_and_scoring_failures_preserve_partial_costs(self):
        class Broken:
            def __init__(self,fail):self.calls=0;self.fail=fail
            def __call__(self,ids):
                self.calls+=1
                if self.calls==self.fail:raise RuntimeError('scripted forward failure')
                logits=torch.full((*ids.shape,10),-1000.,dtype=torch.float64)
                logits[...,NUMBER_START if self.calls==1 else EOS]=0.;return logits
        partial=generate_candidate(Broken(2),self.items[0],self.contract,seed=1,policy_id='scripted-failure',ordinal=0)
        self.assertEqual(partial['token_ids'],[NUMBER_START]);self.assertEqual(partial['stop_reason'],'error')
        self.assertEqual(partial['cost']['attempted_generation_forward_positions'],9)
        self.assertEqual(partial['cost']['generation_forward_positions'],4)
        scorer=generate_candidate(Broken(3),self.items[0],self.contract,seed=1,policy_id='scripted-scorefail',ordinal=0)
        self.assertEqual(scorer['error_stage'],'likelihood_rescoring');self.assertEqual(scorer['cost']['scoring_tokens'],0)
        self.assertEqual(scorer['cost']['attempted_scoring_forward_positions'],5)

    def test_generation_is_causal_and_gold_blind(self):
        model=make_decoder(999).double().eval()
        changed=deepcopy(self.items[0]);changed['reference']='1';changed['prompt']='reference-looking decoration'
        a=generate_candidate(model,self.items[0],self.contract,seed=9,policy_id='m',ordinal=0)
        b=generate_candidate(model,changed,self.contract,seed=9,policy_id='m',ordinal=0)
        for key in ('token_ids','selected_log_probabilities','rescored_log_probabilities','attempt_seed'):
            self.assertEqual(a[key],b[key])

    def test_cli_existing_run_rejected_and_failure_journal_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination=Path(tmp)/'run'
            args=['--items',str(ROOT/'fixtures/inference-selection/items.json'),'--spec',str(ROOT/'experiments/specs/2026-10-04-inference-selection.md'),'--output',str(destination)]
            with patch('dongxi_llms.inference_selection_lab.run_experiment',side_effect=RuntimeError('scripted invocation failure')):
                with self.assertRaises(RuntimeError):main(args)
            self.assertTrue((destination/'failure.json').exists());self.assertTrue((destination/'events.jsonl').exists())
            with self.assertRaises(SystemExit):main(args)


if __name__=='__main__':unittest.main()
