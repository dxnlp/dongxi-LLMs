import copy
import math
import unittest
import torch
from torch import nn
from dongxi_llms.evaluation_lab import pass_at_k, wilson_interval, exact_match, paired_bootstrap, contamination_groups, evaluation_fixture, slice_scores
from dongxi_llms.instruction_data_lab import encode_messages, collate, instruction_fixture, packed_visibility, mixture_exposure
from dongxi_llms.sft_lab import token_loss_sum, make_model, accumulated_gradients, LoRALinear, bounded_run


class EvaluationTests(unittest.TestCase):
    def test_pass_at_k_matches_combinations(self):
        for n in range(1, 15):
            for c in range(n+1):
                for k in range(1, n+1):
                    expected = 1-(math.comb(n-c,k) if n-c >= k else 0)/math.comb(n,k)
                    self.assertAlmostEqual(pass_at_k(n,c,k), expected)

    def test_intervals_and_normalization(self):
        lower, upper = wilson_interval(0,10)
        self.assertAlmostEqual(lower,0.)
        self.assertGreater(upper,.2)
        self.assertTrue(exact_match('  RED\n', ['red']))
        self.assertFalse(exact_match('1.0', ['1']))

    def test_paired_and_leakage(self):
        self.assertEqual(paired_bootstrap([0,1],[0,1])[0],0.)
        records = [dict(id='a',split='train',prompt='COPY red',answer='red'),
                   dict(id='b',split='test',prompt='copy red',answer='RED')]
        self.assertEqual(len(contamination_groups(records)),1)

    def test_fixture_exposes_hidden_rare_regression(self):
        rows=evaluation_fixture()
        self.assertGreater(sum(row['b'] for row in rows),sum(row['a'] for row in rows))
        scores={key:slice_scores([dict(slice=row['slice'],correct=row[key]) for row in rows])
                for key in ('a','b')}
        self.assertGreater(scores['b']['common']['accuracy'],scores['a']['common']['accuracy'])
        self.assertLess(scores['b']['rare-format']['accuracy'],scores['a']['rare-format']['accuracy'])


class InstructionTests(unittest.TestCase):
    def test_target_alignment_and_eos(self):
        example = instruction_fixture()[0]
        batch = collate([example])
        self.assertEqual(int((batch['labels'] != -100).sum()),2)
        self.assertEqual(int(batch['labels'][0,-1]),5)
        self.assertEqual(int(batch['labels'][0,-3]),-100)

    def test_segments_and_mixtures(self):
        mask = packed_visibility([0,0,1,1])
        self.assertFalse(bool(mask[3,0]))
        self.assertTrue(bool(mask[3,2]))
        self.assertFalse(bool(mask[2,3]))
        self.assertEqual(mixture_exposure([.5,.5],[10,90]),[.1,.9])


class SFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_masked_logits_have_zero_direct_gradient(self):
        batch = collate(instruction_fixture())
        logits = torch.randn(4,batch['input_ids'].shape[1],22, requires_grad=True)
        loss,count = token_loss_sum(logits,batch['labels'])
        (loss/count).backward()
        mask = batch['labels'][:,1:] != -100
        self.assertTrue(torch.equal(logits.grad[:,:-1][~mask], torch.zeros_like(logits.grad[:,:-1][~mask])))

    def test_accumulation_equals_joint_batch(self):
        examples = instruction_fixture()
        a,b = make_model(),make_model()
        joint = accumulated_gradients(a,[collate(examples)])
        split = accumulated_gradients(b,[collate(examples[:1]),collate(examples[1:])])
        for name in joint:
            torch.testing.assert_close(joint[name],split[name],atol=2e-6,rtol=2e-5)

    def test_lora_initial_equivalence_and_gradient_asymmetry(self):
        layer = LoRALinear(nn.Linear(6,5,bias=False), rank=2)
        x = torch.randn(3,6)
        torch.testing.assert_close(layer(x),layer.base(x),rtol=0,atol=0)
        layer(x).sum().backward()
        self.assertIsNone(layer.base.weight.grad)
        self.assertEqual(float(layer.a.grad.norm()),0.)
        self.assertGreater(float(layer.b.grad.norm()),0.)
        with torch.no_grad(): layer.b.add_(.1)
        torch.testing.assert_close(layer(x),nn.functional.linear(x,layer.merged_weight()))

    def test_bounded_learning_and_replay(self):
        _, result = bounded_run(40)
        self.assertLess(result['final_loss'],result['initial_loss'])
        self.assertTrue(result['state_forward_replay_equal'])


if __name__ == '__main__':
    unittest.main()
