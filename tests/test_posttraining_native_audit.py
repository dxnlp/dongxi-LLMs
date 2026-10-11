"""Independent CPU checks of the optional audit; no downloaded/native execution."""
from copy import deepcopy
import tempfile
import json
from pathlib import Path
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dongxi_llms import posttraining_native_audit as audit
from dongxi_llms import qwen_rlvr_lab as native
from dongxi_llms.grpo_lab import group_advantages


class NativeAuditTests(unittest.TestCase):
    def model_and_pool(self, rewards=(0., 1.)):
        with torch.random.fork_rng():
            torch.manual_seed(73)
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=16,hidden_size=16,intermediate_size=32,
                num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=8,
                max_position_embeddings=16,tie_word_embeddings=False)).eval().requires_grad_(False)
        prompts = torch.tensor([[1,3,4]])
        responses = torch.tensor([[5,2],[6,2]])
        inputs = torch.cat((prompts.repeat(2,1),responses),-1)[:,:-1]
        with torch.no_grad():
            logits = native.model_logits(model,inputs)[:,2:]
            old = logits.log_softmax(-1).gather(-1,responses[...,None]).squeeze(-1)
        pool = {'prompt_ids':prompts,'responses':responses,'mask':torch.ones(2,2,dtype=torch.bool),
                'old_logp':old,'rewards':list(rewards)}
        return model,pool

    def test_independent_reference_arithmetic_and_tier_denominators(self):
        items,protocol = audit.load_native_spec()
        self.assertEqual(len(items),12)
        self.assertEqual(protocol['group_size']*len(items),48)
        expected=[5,3,7,2,-5,-2,5,5,14,15,12,-15]
        self.assertEqual([i['expected'] for i in items],expected)
        self.assertEqual([audit.native_oracle(i) for i in items],expected)
        self.assertEqual({i['split'] for i in items},{'train','development'})
        self.assertEqual(sum(i['tier']=='signed' for i in items),4)

    def test_no_heldout_or_inferred_reference(self):
        items,protocol = audit.load_native_spec()
        wrong = deepcopy(items[0]);wrong['split']='heldout-source'
        with self.assertRaisesRegex(ValueError,'heldout'):
            audit.native_oracle(wrong)
        wrong = deepcopy(items[0]);wrong['a']=True
        with self.assertRaisesRegex(ValueError,'integer'):
            audit.native_oracle(wrong)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);changed=deepcopy(items);changed[0]['expected']=6
            protocol['items_sha256']=audit.identity(changed)
            (p/'posttraining_native_items.json').write_text(json.dumps(changed))
            (p/'posttraining_native_protocol.json').write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError,'oracle'):
                audit.load_native_spec(p)

    def test_head_gradient_matches_independent_softmax_outer_products(self):
        model,pool = self.model_and_pool()
        before=native.state_digest(model.state_dict())
        result,gradient = audit.head_task_gradient(model,pool)
        inputs=torch.cat((pool['prompt_ids'].repeat(2,1),pool['responses']),-1)[:,:-1]
        with torch.no_grad():
            hidden=model.model(input_ids=inputs,use_cache=False).last_hidden_state[:,2:]
            probabilities=native.model_logits(model,inputs)[:,2:].softmax(-1)
            # Inclusive two-response population std1/2 plus kernel epsilon.
            advantages=torch.tensor([-.5,.5])/(.5+1e-8)
            expected=torch.zeros_like(gradient)
            for row in range(2):
                for position in range(2):
                    score=probabilities[row,position].clone()
                    score[pool['responses'][row,position]]-=1
                    expected.add_(torch.outer(score,hidden[row,position])*advantages[row]/4)
        self.assertTrue(torch.allclose(gradient,expected,atol=1e-7,rtol=1e-5))
        self.assertGreater(result['task_gradient_norm'],0)
        self.assertEqual(result['valid_scored_response_tokens'],4)
        self.assertEqual(result['initial_log_ratio_max_abs'],0)
        self.assertEqual(native.state_digest(model.state_dict()),before)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_allconstant_groups_have_exact_zero_task_gradient(self):
        for rewards in ((0.,0.),(1.,1.)):
            model,pool=self.model_and_pool(rewards)
            result,gradient=audit.head_task_gradient(model,pool)
            self.assertEqual(result['task_gradient_norm'],0)
            self.assertEqual(result['advantage_norm'],0)
            self.assertTrue(torch.equal(gradient,torch.zeros_like(gradient)))

    def test_fresh_likelihood_mismatch_is_an_error(self):
        model,pool=self.model_and_pool();pool['old_logp']=pool['old_logp']+.01
        with self.assertRaisesRegex(ValueError,'likelihood mismatch'):
            audit.head_task_gradient(model,pool)

    def test_gate_needs_both_splits_and_aggregate_task_gradient(self):
        rows=[{'split':split,'strict_terminated_integer_reward':reward} for split in ('train','development') for reward in (True,False)]
        gradients=[{'split':split,'strict_successes':1,'requested_responses':2,'task_gradient_norm':.1} for split in ('train','development')]
        self.assertTrue(audit.audit_gate(rows,gradients,{'train':.1,'development':.1})['passes'])
        self.assertFalse(audit.audit_gate(rows,gradients,{'train':.1,'development':0})['passes'])
        self.assertFalse(audit.audit_gate(rows[:2],gradients,{'train':.1,'development':.1})['passes'])


if __name__=='__main__':
    unittest.main()
