"""Original causal sequence, provenance and printed-step boundary checks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dongxi_llms.response_distillation_lab import (ANS,BOS,CAP,DIGIT,EOS,INTERFACE,STEP,
    adapt_teacher,attach_digest,batch_examples,body_shape,canonical_hash,decode,derived_seed,
    fit_model,freeze_contract,generate,grade,main,make_model,prefix,select,state_digest,
    prepare_local_transfer,token_loss_sum,truth,validate_contract,validate_fixture)

ROOT=Path(__file__).resolve().parents[1]


def fixture():return json.loads((ROOT/'fixtures/response-distillation/items.json').read_text())


def authored(item,contract,ordinal=0,*,total=None,answer=None):
    expected_total,expected=truth(item)
    tokens=[STEP,DIGIT+(expected_total if total is None else total),ANS,DIGIT+(expected if answer is None else answer),EOS]
    state='a'*64;campaign=26011;phase='teacher-data';coordinate=[campaign,state,item['id'],phase,ordinal]
    return attach_digest(dict(contract_id=contract['identity'],interface_sha256=INTERFACE,
        teacher_or_student_state_sha256=state,campaign=campaign,item_id=item['id'],
        item_sha256=canonical_hash(item),source_group=item['source_group'],split=item['split'],
        phase=phase,ordinal=ordinal,sample_id=canonical_hash(coordinate),attempt_seed=derived_seed(*coordinate),
        prompt_ids=prefix(item),token_ids=tokens,raw_response=decode(tokens),
        selected_model_log_probabilities=[-1.]*5,stop='EOS',error=None,cost={'generated_actions':5}))


class ResponseDistillationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.fixture=fixture();self.items=self.fixture['items']
        self.contract=freeze_contract(self.fixture,'f'*64);self.item=self.items[0]

    def test_fixture_source_underlying_problem_and_encoding_leakage_fail(self):
        self.assertEqual(validate_fixture(self.fixture)['items'],18)
        for key,value in (('source_group',self.item['source_group']),('problem',self.item['problem'])):
            changed=deepcopy(self.fixture);changed['items'][6][key]=value
            with self.assertRaises(ValueError):validate_fixture(changed)
        changed=deepcopy(self.fixture);changed['items'][0]['reference']='1'
        with self.assertRaises(ValueError):validate_fixture(changed)

    def test_contract_rehashed_tokenizer_and_template_mismatch_rejected(self):
        for key,value in (('template','wrong serialization'),('interface_sha256','0'*64),('cap',6)):
            c=deepcopy(self.contract);c[key]=value;c['identity']=canonical_hash({k:v for k,v in c.items() if k!='identity'})
            with self.assertRaises(ValueError):validate_contract(c,self.fixture)
        with self.assertRaises(ValueError):freeze_contract(self.fixture,'not a digest')
        c=deepcopy(self.contract);c['vocabulary']['0'],c['vocabulary']['1']=c['vocabulary']['1'],c['vocabulary']['0']
        c['identity']=canonical_hash({k:v for k,v in c.items() if k!='identity'})
        with self.assertRaises(ValueError):adapt_teacher([],self.items,c,'a'*64)

    def test_prompt_excluded_response_eos_included_exactly_one_shift(self):
        e={'prompt_ids':prefix(self.item),'response_ids':[STEP,DIGIT,ANS,DIGIT,EOS]}
        batch=batch_examples([e]);self.assertEqual(batch['supervised_targets'],5)
        self.assertTrue(torch.equal(batch['labels'][0,:4],torch.tensor([-100]*4)))
        model=make_model(2);logits=model(batch['input_ids']);logits.retain_grad()
        loss,count=token_loss_sum(logits,batch['labels']);(loss/count).backward()
        self.assertEqual(count,5);self.assertTrue(torch.equal(logits.grad[0,:3],torch.zeros_like(logits.grad[0,:3])))
        self.assertGreater(float(logits.grad[0,3].abs().sum()),0.)
        self.assertGreater(float(logits.grad[0,7].abs().sum()),0.)
        self.assertEqual(float(logits.grad[0,8].abs().sum()),0.)

    def test_padding_ignored_and_unequal_target_counts_explicit(self):
        rows=[{'prompt_ids':prefix(self.item),'response_ids':[STEP,DIGIT,ANS,DIGIT,EOS]},
              {'prompt_ids':prefix(self.items[1]),'response_ids':[ANS,DIGIT+1,EOS]}]
        b=batch_examples(rows);self.assertEqual(b['supervised_targets'],8)
        self.assertEqual(b['labels'][1,-2:].tolist(),[-100,-100])
        model=make_model(3);actual,count=token_loss_sum(model(b['input_ids']),b['labels'])
        extended={'input_ids':torch.cat((b['input_ids'],torch.zeros((2,2),dtype=torch.long)),-1),
                  'labels':torch.cat((b['labels'],torch.full((2,2),-100,dtype=torch.long)),-1)}
        other,count2=token_loss_sum(model(extended['input_ids']),extended['labels'])
        self.assertTrue(torch.allclose(actual,other,atol=1e-12));self.assertEqual(count,count2)

    def test_no_truncation_missing_embedded_eos_and_specials_fail(self):
        for response in ([],[ANS,DIGIT],[ANS,EOS,DIGIT,EOS],[ANS,True,EOS]):
            with self.assertRaises(ValueError):batch_examples([{'prompt_ids':prefix(self.item),'response_ids':response}])
        with self.assertRaises(ValueError):batch_examples([{'prompt_ids':[BOS]*11,'response_ids':[ANS,DIGIT,EOS]}])
        self.assertIsNone(body_shape([STEP,DIGIT,BOS,DIGIT,EOS]))

    def test_unknown_boolean_negative_or_nonlist_decoding_fails_explicitly(self):
        for tokens in ([15],[-1],[True],None,'STEP'):
            with self.assertRaises(ValueError):decode(tokens)

    def test_first_format_eligible_preserves_semantic_wrong_teacher(self):
        wrong=authored(self.item,self.contract,0,total=1,answer=1);correct=authored(self.item,self.contract,1)
        adapted=adapt_teacher([wrong,correct],self.items,self.contract,'a'*64)
        self.assertEqual(adapted['full'][0]['response_ids'],wrong['token_ids'])
        self.assertEqual(adapted['answer_only'][0]['response_ids'],[ANS,DIGIT+1,EOS])
        self.assertEqual(adapted['answer_only'][0]['parent_sample_id'],wrong['sample_id'])
        self.assertFalse(grade(self.item,wrong)['answer_correct'])

    def test_wrong_step_right_answer_and_valid_step_wrong_answer_distinct(self):
        wrongstep=grade(self.item,authored(self.item,self.contract,total=1,answer=0))
        wronganswer=grade(self.item,authored(self.item,self.contract,total=0,answer=1))
        self.assertTrue(wrongstep['wrong_step_right_answer']);self.assertTrue(wronganswer['valid_step_wrong_answer'])
        self.assertFalse(wrongstep['joint_trace_correct']);self.assertFalse(wronganswer['joint_trace_correct'])

    def test_payload_provenance_phase_duplicate_and_test_source_rejected(self):
        original=authored(self.item,self.contract)
        corrupt=deepcopy(original);corrupt['token_ids'][1]=DIGIT+1
        with self.assertRaises(ValueError):adapt_teacher([corrupt],self.items,self.contract,'a'*64)
        for key,value in (('interface_sha256','0'*64),('phase','evaluation'),('raw_response','wrong'),('source_group','other')):
            corrupt=deepcopy(original);corrupt[key]=value;corrupt=attach_digest({k:v for k,v in corrupt.items() if k!='payload_sha256'})
            with self.assertRaises(ValueError):adapt_teacher([corrupt],self.items,self.contract,'a'*64)
        with self.assertRaises(ValueError):adapt_teacher([original,original],self.items,self.contract,'a'*64)
        with self.assertRaises(ValueError):adapt_teacher([authored(self.items[6],self.contract)],self.items,self.contract,'a'*64)

    def test_missing_eligible_source_keeps_coverage_no_gold_repair(self):
        row=authored(self.item,self.contract);row.update(stop='cap');row=attach_digest({k:v for k,v in row.items() if k!='payload_sha256'})
        result=adapt_teacher([row],self.items,self.contract,'a'*64)
        self.assertEqual(result['full'],[]);self.assertEqual(len(result['missing_train_ids']),6)
        self.assertIn('missing-natural-EOS',result['audit'][0]['reasons'])

    def test_gold_attached_to_payload_does_not_influence_selector(self):
        pool=[authored(self.item,self.contract,0,answer=1),authored(self.item,self.contract,1,answer=0)]
        injected=[attach_digest({**{k:v for k,v in r.items() if k!='payload_sha256'},'reference':'0','correct':i==1}) for i,r in enumerate(pool)]
        for method in ('first','majority','mean_logp'):self.assertEqual(select(pool,method),select(injected,method))
        self.assertEqual(select(pool,'majority'),pool[0]['sample_id'])

    def test_selector_context_likelihood_and_teacher_order_are_validated(self):
        a=authored(self.item,self.contract,0);b=authored(self.item,self.contract,1)
        with self.assertRaises(ValueError):adapt_teacher([b,a],self.items,self.contract,'a'*64)
        c=authored(self.items[1],self.contract,1)
        with self.assertRaises(ValueError):select([a,c],'first')
        c=deepcopy(b);c['selected_model_log_probabilities']=[-1.]
        c=attach_digest({k:v for k,v in c.items() if k!='payload_sha256'})
        with self.assertRaises(ValueError):select([a,c],'mean_logp')

    def test_same_initial_weights_and_actual_backbone_response_gradients(self):
        original=make_model(10);a=deepcopy(original);b=deepcopy(original)
        e={'item_id':self.item['id'],'prompt_ids':prefix(self.item),'response_ids':[STEP,DIGIT,ANS,DIGIT,EOS]}
        full=fit_model(a,[e],updates=2,lr=.015);short=fit_model(b,[dict(e,response_ids=[ANS,DIGIT,EOS])],updates=2,lr=.015)
        self.assertEqual(full['initial_state_sha256'],short['initial_state_sha256'])
        self.assertEqual(full['supervised_targets_per_update'],5);self.assertEqual(short['supervised_targets_per_update'],3)
        reach=full['history'][0]['first_gradient_reach']
        for piece in ('token','q','mlp','lm_head'):self.assertTrue(any(piece in n and v>0 for n,v in reach.items()))
        self.assertNotEqual(full['final_state_sha256'],full['initial_state_sha256'])

    def test_actual_teacher_sampling_coordinates_and_costs_no_gold(self):
        model=make_model(12,True);a=generate(model,self.item,self.contract,campaign=9,phase='teacher-data',ordinal=0)
        changed=deepcopy(self.item);changed['reference']='1'
        b=generate(model,changed,self.contract,campaign=9,phase='teacher-data',ordinal=0)
        self.assertEqual(a['token_ids'],b['token_ids']);self.assertEqual(a['selected_model_log_probabilities'],b['selected_model_log_probabilities'])
        self.assertEqual(a['cost']['completed_forward_positions'],sum(4+i for i in range(len(a['token_ids']))))
        self.assertEqual(a['cost']['generated_actions'],len(a['token_ids']))
        c=generate(model,self.item,self.contract,campaign=9,phase='evaluation',ordinal=0)
        self.assertNotEqual(a['attempt_seed'],c['attempt_seed'])

    def test_ordinary_failure_retains_partial_attempted_work(self):
        class Broken:
            def __init__(self):self.calls=0
            def state_dict(self):return {'test':torch.tensor([1.],dtype=torch.float64)}
            def __call__(self,ids):
                self.calls+=1
                if self.calls==2:raise RuntimeError('scripted teacher forward failure')
                logits=torch.full((*ids.shape,15),-1000.,dtype=torch.float64);logits[...,STEP]=0.;return logits
        row=generate(Broken(),self.item,self.contract,campaign=1,phase='scripted',ordinal=0)
        self.assertEqual(row['token_ids'],[STEP]);self.assertEqual(row['stop'],'error')
        self.assertEqual(row['cost']['attempted_forward_positions'],9);self.assertEqual(row['cost']['completed_forward_positions'],4)

    def test_cli_failure_and_overwrite_preserve_existing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run';args=['--fixture',str(ROOT/'fixtures/response-distillation/items.json'),'--spec',str(ROOT/'experiments/specs/2026-10-04-response-distillation.md'),'--output',str(output)]
            with patch('dongxi_llms.response_distillation_lab.run_reference',side_effect=RuntimeError('scripted failure')):
                with self.assertRaises(RuntimeError):main(args)
            self.assertTrue((output/'failure.json').exists())
            with self.assertRaises(SystemExit):main(args)

    def test_local_transfer_protocol_never_loads_or_executes_and_gates_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            teacher=Path(tmp)/'teacher';student=Path(tmp)/'student';teacher.mkdir();student.mkdir()
            # Runtime-generated temp fixtures, not real checkpoint or model evidence.
            for path in (teacher,student):
                (path/'config.json').write_text('{}');(path/'tokenizer.json').write_text('{}')
                (path/'dummy.safetensors').write_bytes(b'not a model; inspect filenames only')
            plan=prepare_local_transfer(teacher,student,['train-g'],['test-g'])
            self.assertFalse(plan['execution_permitted']);self.assertEqual(plan['teacher_model_loading'],'not performed')
            for kwargs in ({'execute':True},{'reserve_gib':20},{'teacher_cap':10000},{'teacher_attempts':True}):
                with self.assertRaises(ValueError):prepare_local_transfer(teacher,student,['train-g'],['test-g'],**kwargs)
            with self.assertRaises(ValueError):prepare_local_transfer(teacher,student,['same'],['same'])
            with self.assertRaises(ValueError):prepare_local_transfer('/not/a/local/model',student,['train'],['test'])


if __name__=='__main__':unittest.main()
