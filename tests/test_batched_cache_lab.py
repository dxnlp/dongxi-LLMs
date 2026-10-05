"""Independent ragged position, cache, likelihood, stopping and resume checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch

from dongxi_llms.batched_cache_lab import (CONFIG,EOS,INTERFACE,PROMPTS,Generation,PolicySession,
    branch,compare,digest,forward,left_pad,load_state,make_model,model_hash,payload,save_state,
    score_branch,single_run,numeric_tree,tensor_tree_equal)


def independent_v2(value,tensor_bytes=None):
    """Independent accumulated byte stream for the unchanged framed contract."""
    chunks=[]
    def primitive(x):
        return json.dumps([type(x).__name__,x],sort_keys=True,allow_nan=False,separators=(",",":")).encode()
    def put(raw):chunks.extend((len(raw).to_bytes(8,"big"),raw))
    def walk(x):
        if isinstance(x,torch.Tensor):
            put(b"tensor");put(json.dumps([str(x.dtype),list(x.shape)],separators=(",",":")).encode())
            raw=tensor_bytes(x) if tensor_bytes is not None else x.detach().cpu().contiguous().numpy().tobytes()
            put(raw)
        elif isinstance(x,dict):
            put(b"dict");chunks.append(len(x).to_bytes(8,"big"))
            for key in sorted(x,key=primitive):walk(key);walk(x[key])
        elif isinstance(x,(list,tuple)):
            put(type(x).__name__.encode());chunks.append(len(x).to_bytes(8,"big"))
            for child in x:walk(child)
        else:put(b"scalar");put(primitive(x))
    walk(value)
    return hashlib.sha256(b"".join(chunks)).hexdigest()


class BatchedCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_identity_distinguishes_nested_dictionary_container_boundaries(self):
        a={"a":{"b":1},"c":2};b={"a":{"b":1,"c":2}}
        self.assertNotEqual(digest(a),digest(b))
        self.assertNotEqual(digest({"a":{},"b":{}}),digest({"a":{"b":{}}}))
        self.assertNotEqual(digest([[],[]]),digest([[]]))

    def test_identity_is_order_invariant_but_key_value_and_container_typed(self):
        self.assertEqual(digest({"a":1,"b":2}),digest({"b":2,"a":1}))
        self.assertNotEqual(digest({1:"x"}),digest({"1":"x"}))
        for a,b in (([1],(1,)),([1,2],[12]),(1,1.),(False,0),(None,"None"),({"a":1},{"a":"1"})):
            self.assertNotEqual(digest(a),digest(b))

    def test_identity_includes_tensor_dtype_shape_and_actual_bytes(self):
        tensor=torch.tensor([1.,2.],dtype=torch.float64)
        self.assertEqual(digest(tensor),digest(tensor.clone()))
        self.assertNotEqual(digest(tensor),digest(tensor.float()))
        self.assertNotEqual(digest(tensor),digest(tensor.reshape(1,2)))
        self.assertNotEqual(digest(tensor),digest(torch.tensor([1.,3.],dtype=torch.float64)))

    def test_tensor_byte_parity_original_numpy_dtypes_and_layouts(self):
        dtypes=(torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,
                torch.float16,torch.float32,torch.float64,torch.complex64,torch.complex128)
        for dtype in dtypes:
            base=torch.arange(12).reshape(3,4).to(dtype)
            layouts=(base[0,0],base[:0],base[:,0:0],base.reshape(-1),base,base.T,base[:,::2])
            for x in layouts:
                with self.subTest(dtype=dtype,shape=tuple(x.shape),stride=x.stride()):
                    self.assertEqual(digest(x),independent_v2(x))
                    nested={'tensor':x,'children':[x.clone(),(True,3,None)],'empty':{}}
                    self.assertEqual(digest(nested),independent_v2(nested))
                    self.assertEqual(digest(x),digest(x.contiguous()))

    def test_bf16_payload_matches_independently_packed_bits_in_all_layouts(self):
        # These are raw bit fixtures, not computed BF16 model outputs.
        bits=[0x0000,0x8000,0x3f80,0xc000,0x7f80,0xff80,0x7fc1,0x0001]
        integers=[b if b<32768 else b-65536 for b in bits]
        base=torch.tensor(integers,dtype=torch.int16).view(torch.bfloat16).reshape(2,4)
        def packed(x):
            words=x.detach().contiguous().reshape(-1).view(torch.int16).tolist()
            return b''.join(struct.pack('=H',word&0xffff) for word in words)
        self.assertEqual(packed(base),b''.join(struct.pack('=H',b) for b in bits))
        for x in (base[0,0],base[:0],base[:,0:0],base.reshape(-1),base,base.T,base[:,::2]):
            with self.subTest(shape=tuple(x.shape),stride=x.stride()):
                self.assertEqual(digest(x),independent_v2(x,packed))
                self.assertEqual(digest(x),digest(x.contiguous()))
                self.assertEqual(digest({'t':x,'rows':[x,(x,)]}),independent_v2({'t':x,'rows':[x,(x,)]},packed))

    def test_bf16_dtype_shape_bits_and_container_identity_do_not_alias(self):
        x=torch.tensor([1.,-2.],dtype=torch.bfloat16)
        changed=x.clone();changed.view(torch.int16)[0]^=1
        for other in (changed,x.reshape(1,2),x.view(torch.int16),x.float()):
            self.assertNotEqual(digest(x),digest(other))
        self.assertNotEqual(digest(torch.tensor(1.,dtype=torch.bfloat16)),digest(x[:1]))
        self.assertNotEqual(digest(torch.empty(0,dtype=torch.bfloat16)),digest(torch.empty((0,1),dtype=torch.bfloat16)))
        self.assertNotEqual(digest({'a':{'b':x},'c':1}),digest({'a':{'b':x,'c':1}}))
        self.assertNotEqual(digest([x]),digest((x,)))
        self.assertNotEqual(digest(torch.tensor(0.,dtype=torch.bfloat16)),digest(torch.tensor(-0.,dtype=torch.bfloat16)))

    def test_bf16_snapshot_witness_bit_tamper_rejected_before_restore(self):
        engine=Generation(make_model());state=engine.snapshot()
        state.pop('state_sha256');state['byte_witness']=torch.tensor([1.,2.],dtype=torch.bfloat16)
        state['state_sha256']=digest(state)
        state['byte_witness'].view(torch.int16)[0]^=1
        with self.assertRaisesRegex(ValueError,'Snapshot content digest'):
            Generation.restore(make_model(),state,expected_contract=state['contract'])

    def test_migration_projection_preserves_actions_versions_scores_and_tensor_bytes(self):
        a={"model_sha256":"old","contract":{"schema":"v1"},"token":3,"policy_version":2,"logp":-.4}
        b=dict(a,model_sha256="new",contract={"schema":"v2"})
        self.assertEqual(numeric_tree(a),numeric_tree(b))
        for key,value in (("token",4),("policy_version",3),("logp",-.5)):
            self.assertNotEqual(numeric_tree(a),numeric_tree(dict(b,**{key:value})))
        positive=torch.tensor([0.],dtype=torch.float64);negative=-positive
        self.assertTrue(tensor_tree_equal({"x":positive},{"x":positive.clone()}))
        self.assertFalse(tensor_tree_equal({"x":positive},{"x":negative}))
        self.assertFalse(tensor_tree_equal({"x":positive},{"x":positive.float()}))

    def test_old_identity_algorithm_is_rejected_before_tensor_loading(self):
        engine=Generation(make_model());s=engine.snapshot()
        with tempfile.TemporaryDirectory() as td:
            saved=save_state(s,Path(td)/"state.pt")
            old=dict(s["contract"],schema="ragged-generation-v1",identity_algorithm="v1")
            with patch("torch.load") as load:
                with self.assertRaisesRegex(ValueError,"obsolete"):
                    load_state(saved["path"],expected_file_sha256=saved["file_sha256"],expected_contract=old)
                load.assert_not_called()

    def test_unpadded_wrapper_matches_original_logits_and_gradients(self):
        model=make_model();other=deepcopy(model)
        ids,mask,pos=left_pad([[1,3,4],[1,5,6]])
        actual,_,_=forward(model,ids,mask,pos)
        expected=other(ids)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        actual.square().sum().backward();expected.square().sum().backward()
        for a,b in zip(model.parameters(),other.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=1e-12,rtol=1e-12)

    def test_ragged_left_positions_mask_ids_are_not_validity(self):
        ids,mask,pos=left_pad([[1,0,3],[1,4,5,6,7]])
        self.assertEqual(ids[0].tolist(),[0,0,1,0,3])
        self.assertEqual(mask[0].tolist(),[False,False,True,True,True])
        self.assertEqual(pos[0].tolist(),[0,0,0,1,2])
        model=make_model();actual,_,_=forward(model,ids,mask,pos)
        for k,row in enumerate(([1,0,3],[1,4,5,6,7])):
            torch.testing.assert_close(actual[k,-len(row):],model(torch.tensor([row]))[0],atol=1e-12,rtol=1e-12)

    def test_future_and_padding_interventions_preserve_legal_outputs(self):
        model=make_model();ids,mask,pos=left_pad(list(PROMPTS.values()))
        baseline,_,_=forward(model,ids,mask,pos)
        pads=ids.clone();pads[~mask]=11
        changed,_,_=forward(model,pads,mask,pos)
        torch.testing.assert_close(baseline[mask],changed[mask],atol=0,rtol=0)
        future=ids.clone();future[2,-1]=3
        alternate,_,_=forward(model,future,mask,pos)
        torch.testing.assert_close(baseline[2,:-1],alternate[2,:-1],atol=0,rtol=0)

    def test_cached_ragged_next_logits_and_compact_gqa_payload(self):
        model=make_model();ids,mask,pos=left_pad(list(PROMPTS.values()))
        _,cache,keymask=forward(model,ids,mask,pos)
        self.assertEqual(cache[0][0].shape,(3,2,6,4))
        self.assertEqual(payload(cache),2*CONFIG.layers*3*2*6*4*8)
        new=torch.tensor([[3],[4],[5]]);valid=torch.ones_like(new,dtype=torch.bool)
        logits,_,_=forward(model,new,valid,keymask.sum(-1)[:,None],cache=cache,key_mask=keymask)
        for i,(row,token) in enumerate(zip(PROMPTS.values(),new.flatten().tolist())):
            expected=model(torch.tensor([row+[token]]))[0,-1]
            torch.testing.assert_close(logits[i,0],expected,atol=1e-12,rtol=1e-12)

    def test_wrong_padded_positions_expose_real_error(self):
        model=make_model();ids,mask,pos=left_pad(list(PROMPTS.values()))
        a,_,_=forward(model,ids,mask,pos)
        # Global shifts alone cancel in ideal RoPE; use genuinely wrong positions,
        # giving every query the same zero position rather than row-relative order.
        b,_,_=forward(model,ids,mask,torch.zeros_like(pos))
        self.assertGreater(float((a[mask]-b[mask]).detach().abs().max()),1e-9)

    def test_all_four_greedy_paths_and_sampled_reorder(self):
        model=make_model();baseline=single_run(model,cached=False)
        for cached in (False,True):
            for batched in (False,True):
                a=Generation(model,cached=cached).finish() if batched else single_run(model,cached=cached)
                c=compare(baseline,a);self.assertTrue(c["ids_equal"] and c["stops_equal"])
                self.assertLess(c["selected_logp_max_error"],1e-10)
        a=Generation(model,greedy=False).finish()
        for b in (single_run(model,cached=False,greedy=False),Generation(model,dict(reversed(list(PROMPTS.items()))),greedy=False).finish()):
            c=compare(a,b);self.assertTrue(c["ids_equal"]);self.assertLess(c["selected_logp_max_error"],1e-10)

    def test_forced_stop_compacts_and_cap_is_not_eos_padding(self):
        run=Generation(make_model(),support="forced-stop").finish()
        self.assertEqual([r["token"] for r in run["records"]["short"]],[EOS])
        self.assertEqual(run["stops"],{"short":"eos","long":"eos","medium":"cap"})
        self.assertEqual([len(run["records"][k]) for k in PROMPTS],[1,4,2])
        rows=[e["active_rows"] for e in run["work"]["events"]]
        self.assertEqual(rows,[["short","medium","long"],["medium","long"],["medium"],["medium"]])
        self.assertTrue(all(r["behavior_logp"]==0 for r in run["records"]["short"]))
        self.assertLess(run["records"]["short"][0]["raw_logp"],0)

    def test_work_counting_is_rectangular_inputs_not_full_flops(self):
        a=Generation(make_model(),support="forced-stop").finish()["work"]
        self.assertEqual(a["prefill_calls"],1);self.assertEqual(a["decode_calls"],3)
        self.assertEqual(a["forwarded_positions"],18+2+1+1)
        self.assertEqual(a["valid_input_positions"],12+2+1+1)
        self.assertEqual(a["padding_positions"],6);self.assertEqual(a["useful_generated_tokens"],7)
        self.assertEqual(a["peak_kv_payload_bytes"],4608)

    def test_saved_cache_snapshot_exact_next_draw_and_whole_tail(self):
        engine=Generation(make_model(),greedy=False);engine.draw();engine.draw()
        state=engine.snapshot();contract=engine.contract();next_draw=engine.draw();expected=engine.finish()
        replay=Generation.restore(make_model(),state,expected_contract=contract)
        self.assertEqual(replay.draw(),next_draw);actual=replay.finish()
        self.assertEqual(actual,expected)
        rebuild=Generation.restore(make_model(),state,expected_contract=contract,rebuild=True).finish()
        c=compare(expected,rebuild);self.assertTrue(c["ids_equal"]);self.assertLess(c["selected_logp_max_error"],1e-10)

    def test_generation_stale_weights_and_cursor_rng_content_rejected(self):
        engine=Generation(make_model(),greedy=False);engine.draw();state=engine.snapshot()
        with torch.no_grad():engine.model.token.weight[1,0]+=.01
        with self.assertRaisesRegex(ValueError,"Stale"):engine.draw()
        for key in ("cursor","rng"):
            changed=deepcopy(state)
            if key=="cursor":changed[key]["short"]+=1
            else:changed[key]["short"][0]^=1
            with self.assertRaisesRegex(ValueError,"digest"):Generation.restore(make_model(),changed,expected_contract=state["contract"])

    def test_bound_version_and_cache_tensor_cannot_silently_change(self):
        engine=Generation(make_model());engine.version+=1
        with self.assertRaisesRegex(ValueError,"version"):engine.draw()
        engine=Generation(make_model());engine.cache[0][0][0,0,0,0]+=.01
        with self.assertRaisesRegex(ValueError,"Cache tensor"):engine.draw()
        engine=Generation(make_model());engine.logits[0,3]+=.1
        with self.assertRaisesRegex(ValueError,"Cache tensor"):engine.draw()

    def test_resealed_cursor_and_changed_interface_rejected_before_reuse(self):
        engine=Generation(make_model(),greedy=False);engine.draw();s=engine.snapshot()
        bad=deepcopy(s);bad["cursor"]["short"]+=1;bad.pop("state_sha256");bad["state_sha256"]=digest(bad)
        with self.assertRaisesRegex(ValueError,"cursor"):Generation.restore(make_model(),bad,expected_contract=s["contract"])
        with self.assertRaisesRegex(ValueError,"contract"):Generation.restore(make_model(),s,expected_contract=dict(s["contract"],interface={}))

    def test_disk_state_roundtrip_checks_contract_and_bytes_before_deserialization(self):
        engine=Generation(make_model(),greedy=False);engine.draw();s=engine.snapshot()
        with tempfile.TemporaryDirectory() as td:
            saved=save_state(s,Path(td)/"state.pt")
            loaded=load_state(saved["path"],expected_file_sha256=saved["file_sha256"],expected_contract=s["contract"])
            self.assertEqual(digest(s),digest(loaded))
            with self.assertRaises(FileExistsError):save_state(s,saved["path"])
            with patch("torch.load") as load:
                with self.assertRaisesRegex(ValueError,"Pre-load"):load_state(saved["path"],expected_file_sha256="0"*64,expected_contract=s["contract"])
                load.assert_not_called()
            with patch("torch.load") as load:
                with self.assertRaisesRegex(ValueError,"contract"):load_state(saved["path"],expected_file_sha256=saved["file_sha256"],expected_contract=dict(s["contract"],interface={}))
                load.assert_not_called()

    def test_completed_dpo_and_rlvr_recovery_exact_next_batch_state_and_adam(self):
        for objective in ("dpo","rlvr"):
            session=PolicySession(objective)
            for _ in range(3):session.update()
            s=session.snapshot();tail=[session.update() for _ in range(3)]
            replay=PolicySession(objective);replay.restore(s)
            self.assertEqual([replay.update() for _ in range(3)],tail)
            self.assertEqual(model_hash(replay.model),model_hash(session.model))
            self.assertEqual(digest(replay.optimizer.state_dict()),digest(session.optimizer.state_dict()))

    def test_post_collection_pending_rollout_resumes_without_resampling(self):
        for objective in ("dpo","rlvr"):
            session=PolicySession(objective)
            for _ in range(3):session.update()
            with self.assertRaises(InterruptedError):session.update(interrupt_after_collection=True)
            s=session.snapshot();self.assertIsNotNone(s["pending"])
            expected=session.update();replay=PolicySession(objective);replay.restore(s)
            with patch.object(replay,"collect",side_effect=AssertionError("must not collect again")):
                actual=replay.update()
            self.assertEqual(actual,expected)

    def test_actual_update_gradients_reference_freeze_and_support_likelihoods(self):
        for objective in ("dpo","rlvr"):
            session=PolicySession(objective);initial=model_hash(session.model)
            records=[session.update() for _ in range(3)]
            self.assertNotEqual(model_hash(session.model),initial)
            self.assertTrue(any(r["gradient_norm"]>0 for r in records))
            self.assertEqual(model_hash(session.reference),session.contract["reference_sha256"])
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in session.reference.parameters()))
            if objective=="rlvr":self.assertLess(max(r["alignment_error"] for r in records),1e-10)

    def test_omitted_optimizer_cursor_and_rollout_rng_are_broken_controls(self):
        for objective,omit in (("dpo","optimizer"),("dpo","cursor"),("rlvr","rng")):
            original=PolicySession(objective)
            for _ in range(3):original.update()
            s=original.snapshot();tail=[original.update() for _ in range(3)]
            broken=PolicySession(objective);broken.restore(s,omit=omit)
            self.assertNotEqual([broken.update() for _ in range(3)],tail)

    def test_pending_version_cursor_and_weights_cannot_mix(self):
        session=PolicySession("rlvr");session.collect();s=session.snapshot()
        for field in ("version","cursor_after_collection","policy_sha256"):
            bad=deepcopy(s);bad["pending"][field]="wrong" if field=="policy_sha256" else 99
            bad.pop("state_sha256");bad["state_sha256"]=digest(bad)
            with self.assertRaisesRegex(ValueError,"Pending"):PolicySession("rlvr").restore(bad)
        with torch.no_grad():session.model.token.weight[1,0]+=.1
        with self.assertRaisesRegex(ValueError,"Stale"):session.update()

    def test_changed_training_data_reference_and_completed_cursor_rejected(self):
        session=PolicySession("dpo");session.update();s=session.snapshot()
        for field in ("contract","reference","step"):
            bad=deepcopy(s)
            if field=="contract":bad[field]["targets"]=[4,3,4]
            elif field=="reference":bad[field]["token.weight"][0,0]+=.1
            else:bad[field]=5
            bad.pop("state_sha256");bad["state_sha256"]=digest(bad)
            with self.assertRaises(ValueError):PolicySession("dpo").restore(bad)
        session=PolicySession("dpo");session.targets[0]=4
        with self.assertRaisesRegex(ValueError,"bound training"):session.collect()

    def test_independent_dpo_shift_eos_mask_and_right_padding(self):
        model=make_model();prompts=list(PROMPTS.values());b=branch(prompts,[3,4,3])
        actual=score_branch(model,b)
        expected=[]
        for p,a in zip(prompts,[3,4,3]):
            full=p+[a,EOS];lp=model(torch.tensor([full[:-1]])).log_softmax(-1)
            expected.append(lp[0,len(p)-1,a]+lp[0,len(p),EOS])
        torch.testing.assert_close(actual,torch.stack(expected),atol=1e-12,rtol=1e-12)

    def test_invalid_geometry_and_caps_are_rejected(self):
        for rows in ([],[[]],[[1,-1]],[[True,3]]):
            with self.assertRaises(ValueError):left_pad(rows)
        with self.assertRaises(ValueError):Generation(make_model(),cap=7)
        with self.assertRaises(ValueError):Generation(make_model(),interface=dict(INTERFACE,template="other"))
        model=make_model();ids,mask,pos=left_pad([[1,3]])
        with self.assertRaises(ValueError):forward(model,ids,mask.float(),pos)


if __name__=="__main__":unittest.main()
