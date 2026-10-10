"""Bounded CPU arithmetic and existing mechanism references for Chapters 1–5.

These illustrative computations are separate from historical model experiments.
Run with the goal-local locked interpreter and PYTHONPATH=src; no model downloads.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from dongxi_llms.causal_attention_lab import attention_trace
from dongxi_llms.decoder_lab import (
    DecoderConfig, TinyDecoder, analytical_parameters, teaching_batch,
)
from dongxi_llms.embedding_gradient_lab import repeated_lookup_demo


def main() -> None:
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[4]
    result = {
        "scope": "Illustrative CPU arithmetic; no new capability or model-training evidence",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "cpu", "arithmetic_dtype": "float64 except declared controls"},
    }
    parameter_count = 1_000_000
    result["memory"] = {
        "parameters": parameter_count,
        "fp32_adamw_bytes": {
            "weights": 4 * parameter_count, "gradients": 4 * parameter_count,
            "first_moment": 4 * parameter_count, "second_moment": 4 * parameter_count,
            "total": 16 * parameter_count,
        },
        "bf16_weights_gradients_fp32_moments_no_master_bytes": 12 * parameter_count,
        "same_with_fp32_master_bytes": 16 * parameter_count,
    }
    a, b, c = [torch.tensor(v, dtype=torch.float32) for v in (1e8, -1e8, 1)]
    result["reduction_order"] = {"left": float((a+b)+c), "right": float(a+(b+c))}

    # Authored row values, with the existing repeated-lookup computation as a control.
    table = torch.tensor([[0., 0.], [2., 1.], [1., 0.], [1., 1.], [-1., 0.], [0., 1.]],
                         dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([2, 5, 3, 2])
    lookup = table[ids]
    lookup.sum().backward()
    result["embedding"] = {
        "ids": ids.tolist(), "lookup": lookup.tolist(), "gradient": table.grad.tolist(),
        "after_sgd_eta_0_1": (table.detach()-.1*table.grad).tolist(),
        "existing_repeated_lookup_reference": repeated_lookup_demo(),
    }
    changed_table = table.detach().clone().requires_grad_()
    changed_table[torch.tensor([2,5,3,5])].sum().backward()
    result["embedding"]["changed_final_id_gradient"] = changed_table.grad.tolist()

    # Three-row output-head microscope, separate from the six-row tokenizer example.
    h = torch.tensor([1., 0.], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([[2., 0.], [1., 0.], [-1., 0.]], dtype=torch.float64, requires_grad=True)
    z = h @ w.T
    p = z.softmax(-1)
    loss = F.cross_entropy(z[None], torch.tensor([1]))
    gw, gh = torch.autograd.grad(loss, (w, h))
    new_w = w.detach()-.1*gw
    new_z = h.detach() @ new_w.T
    result["output_head"] = {
        "h": h.tolist(), "w": w.tolist(), "z": z.tolist(), "p": p.tolist(),
        "loss": float(loss.detach()), "logit_gradient": (p.detach()-torch.tensor([0.,1.,0.])).tolist(),
        "weight_gradient": gw.tolist(), "hidden_gradient": gh.tolist(),
        "eta": .1, "updated_w": new_w.tolist(), "updated_logits": new_z.tolist(),
        "updated_p": new_z.softmax(-1).tolist(),
        "updated_loss": float(F.cross_entropy(new_z[None], torch.tensor([1]))),
        "changed_target_cat_gradient": (p.detach()-torch.tensor([1.,0.,0.])).tolist(),
    }
    observed = [.5, .25, .8]
    losses = [-math.log(v) for v in observed]
    result["chain_probability"] = {
        "conditional_probabilities": observed, "product": math.prod(observed),
        "sum_nll": sum(losses), "mean_nll": sum(losses)/3,
        "geometric_mean_target_probability": math.exp(-sum(losses)/3),
    }
    q = torch.tensor([.7,.3], dtype=torch.float64)
    entropy = float(-(q*q.log()).sum())
    result["entropy_kl"] = {"q": q.tolist(), "entropy": entropy, "candidates": []}
    for pp in ([.5,.5], [.7,.3], [.9,.1]):
        candidate = torch.tensor(pp, dtype=torch.float64)
        ce = float(-(q*candidate.log()).sum())
        kl = float((q*(q.log()-candidate.log())).sum())
        result["entropy_kl"]["candidates"].append({"p": pp, "cross_entropy": ce, "kl": kl})

    qk = torch.tensor([[1.,0.],[0.,1.],[1.,1.]], dtype=torch.float64)
    values = torch.tensor([[1.,2.],[3.,0.],[0.,4.]], dtype=torch.float64)
    trace = attention_trace(qk, qk, values)
    aa, vv = trace["weights"][1,:2], values[:2]
    go = torch.tensor([1.,0.], dtype=torch.float64)
    manual_score_gradient = aa * (vv @ go - trace["output"][1] @ go)
    scores = trace["scaled"][1,:2].clone().requires_grad_()
    objective = scores.softmax(-1) @ vv @ go
    score_gradient, = torch.autograd.grad(objective, (scores,))
    changed_values = values.clone(); changed_values[-1] = torch.tensor([100.,-100.])
    changed_trace = attention_trace(qk, qk, changed_values)
    result["attention"] = {
        "weights": trace["weights"].tolist(), "output": trace["output"].tolist(),
        "changed_final_value_output": changed_trace["output"].tolist(),
        "row_2_first_coordinate_score_gradient": score_gradient.tolist(),
        "manual_gradient_max_error": float((score_gradient-manual_score_gradient).abs().max()),
        "hard_lookup_control": [],
    }
    same_values = vv[0:1].expand(2,-1)
    same_scores = trace["scaled"][1,:2].clone().requires_grad_()
    same_objective = same_scores.softmax(-1) @ same_values @ go
    same_gradient, = torch.autograd.grad(same_objective, (same_scores,))
    result["attention"]["identical_values_score_gradient"] = same_gradient.tolist()
    for alpha in (1.,4.,16.):
        score = torch.tensor([alpha,0.,0.], dtype=torch.float64)/math.sqrt(3)
        result["attention"]["hard_lookup_control"].append({"alpha": alpha, "weights": score.softmax(-1).tolist()})
    result["attention"]["scaled_score_control"] = {
        "unscaled_scores": [8.,-8.], "unscaled_weights": torch.tensor([8.,-8.],dtype=torch.float64).softmax(-1).tolist(),
        "scaled_scores": [1.,-1.], "scaled_weights": torch.tensor([1.,-1.],dtype=torch.float64).softmax(-1).tolist(),
    }

    # Same initialization and assembly as the preserved Day 5 reference notebook.
    torch.manual_seed(505)
    model = TinyDecoder(DecoderConfig()).double()
    input_ids, labels = teaching_batch()
    with torch.no_grad():
        state = model.token(input_ids)+model.position(torch.arange(input_ids.shape[1]))[None]
        forward_trace = [{"stage": "token + position", "shape": list(state.shape), "sample_std": float(state.std())}]
        first_input = state.clone()
        for i, block in enumerate(model.blocks):
            state, _, _ = block(state)
            forward_trace.append({"stage": f"block {i}", "shape": list(state.shape), "sample_std": float(state.std())})
        normed = model.final_norm(state)
        logits = model.lm_head(normed)
        forward_trace.extend([
            {"stage": "final LayerNorm", "shape": list(normed.shape), "sample_std": float(normed.std())},
            {"stage": "vocabulary head", "shape": list(logits.shape), "sample_std": float(logits.std())},
        ])
        assembled_error = float((logits-model(input_ids)).abs().max())
        zero_block = copy.deepcopy(model.blocks[0])
        zero_block.attn.out.weight.zero_()
        zero_block.mlp.down.weight.zero_(); zero_block.mlp.down.bias.zero_()
        zero_output, _, _ = zero_block(first_input)
        zero_branch_error = float((zero_output-first_input).abs().max())
    norm_input = torch.tensor([10.,20.,30.,40.],dtype=torch.float64)
    norm_output = F.layer_norm(norm_input,[4])
    result["decoder"] = {
        "seed": 505, "construction": "float32 initialization followed by double(), matching Day 5 notebook",
        "forward_trace": forward_trace, "assembled_forward_max_error": assembled_error,
        "zero_branches_input_max_error": zero_branch_error,
        "layer_norm": {"input": norm_input.tolist(), "population_variance": float(norm_input.var(correction=0)),
                       "output": norm_output.tolist(),
                       "common_offset_max_error": float((norm_output-F.layer_norm(norm_input+100,[4])).abs().max())},
        "gelu": {"inputs": [-1.,0.,1.], "outputs": F.gelu(torch.tensor([-1.,0.,1.],dtype=torch.float64)).tolist()},
    }
    qwen = DecoderConfig(vocab=151936, width=1024, heads=16, kv_heads=8,
                         head_dim=128, layers=28, hidden=3072, modern=True, qk_norm=True)
    total = analytical_parameters(qwen)
    counts = {"tied_embedding_and_head": qwen.vocab*qwen.width,
              "attention_projections": qwen.layers*2*qwen.width*qwen.head_dim*(qwen.heads+qwen.kv_heads),
              "gated_mlps": qwen.layers*3*qwen.width*qwen.hidden,
              "block_rms_scales": qwen.layers*2*qwen.width,
              "qk_rms_scales": qwen.layers*2*qwen.head_dim,
              "final_rms_scale": qwen.width}
    result["qwen_configuration_accounting"] = {
        "unique_parameters": total, "component_parameters": counts,
        "component_sum_equals_analytical": sum(counts.values()) == total,
        "mlp_fraction": counts["gated_mlps"]/total,
        "bf16_raw_parameter_bytes": total*2,
        "untied_additional_parameters": qwen.vocab*qwen.width,
        "untied_additional_bf16_mib": qwen.vocab*qwen.width*2/2**20,
        "evidence_level": "Calculation from config and architecture shapes; no new Qwen model loaded",
    }
    snapshot = Path('/home/dongxi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca')
    if (snapshot/'config.json').exists() and (snapshot/'model.safetensors').exists():
        raw_config = (snapshot/'config.json').read_bytes()
        actual_config = json.loads(raw_config)
        for name, expected in {'hidden_size':1024,'num_attention_heads':16,'num_key_value_heads':8,
                               'head_dim':128,'intermediate_size':3072,'num_hidden_layers':28,
                               'vocab_size':151936,'tie_word_embeddings':True,'attention_bias':False}.items():
            if actual_config[name] != expected:
                raise ValueError(f'Pinned config changed at {name}')
        with (snapshot/'model.safetensors').open('rb') as stream:
            header_size = int.from_bytes(stream.read(8),'little')
            raw_header = stream.read(header_size)
        header = json.loads(raw_header)
        tensors = {k:v for k,v in header.items() if k != '__metadata__'}
        result['qwen_configuration_accounting']['pinned_source'] = {
            'model': 'Qwen/Qwen3-0.6B', 'revision': 'c1899de289a04d12100db370d81485cdf75e47ca',
            'config_url': 'https://huggingface.co/Qwen/Qwen3-0.6B/raw/c1899de289a04d12100db370d81485cdf75e47ca/config.json',
            'local_snapshot': str(snapshot), 'config_sha256': hashlib.sha256(raw_config).hexdigest(),
            'config': actual_config, 'header_size_bytes': header_size,
            'header_sha256': hashlib.sha256(raw_header).hexdigest(),
            'serialized_tensor_count': len(tensors),
            'serialized_elements_including_separate_head': sum(math.prod(v['shape']) for v in tensors.values()),
            'serialized_dtype_set': sorted(set(v['dtype'] for v in tensors.values())),
            'q_norm_shape': tensors['model.layers.0.self_attn.q_norm.weight']['shape'],
            'k_norm_shape': tensors['model.layers.0.self_attn.k_norm.weight']['shape'],
            'embedding_shape': tensors['model.embed_tokens.weight']['shape'],
            'separately_stored_head_shape': tensors['lm_head.weight']['shape'],
            'file_bytes': (snapshot/'model.safetensors').stat().st_size,
            'new_read_scope': 'JSON config and safetensors header only; no tensor payload or runtime aliasing inspection',
            'historical_runtime_tying_evidence': 'experiments/reports/2026-08-31-qwen3-embedding-inspection.md',
            'historical_runtime_tying_same_revision': True,
            'historical_runtime_tying_report_sha256': hashlib.sha256((root/'experiments/reports/2026-08-31-qwen3-embedding-inspection.md').read_bytes()).hexdigest(),
        }
    sources = ["src/dongxi_llms/causal_attention_lab.py", "src/dongxi_llms/decoder_lab.py",
               "src/dongxi_llms/embedding_gradient_lab.py", "notebooks/day-03/01_logits_softmax_nll.ipynb",
               "notebooks/day-03/03_distribution_learning.ipynb", "notebooks/day-05/06_assemble_decoder.ipynb"]
    result["source_sha256"] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources}
    destination = Path(__file__).with_name("depth-calculations.json")
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"destination": str(destination), "output_head": result["output_head"],
                      "entropy_kl": result["entropy_kl"], "attention": result["attention"],
                      "decoder": result["decoder"], "qwen": result["qwen_configuration_accounting"]}, indent=2))


if __name__ == "__main__":
    main()
