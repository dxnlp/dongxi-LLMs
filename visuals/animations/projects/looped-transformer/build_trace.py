"""CPU smoke evidence from the course's actual causal decoder block."""
from dataclasses import asdict, replace
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[3]
sys.path.insert(0, str(ROOT/'src'))
from dongxi_llms.decoder_lab import DecoderConfig, DecoderBlock, parameter_count, cost_estimate


def build_trace():
    torch.set_num_threads(1)
    torch.manual_seed(505)
    cfg = DecoderConfig()
    block = DecoderBlock(cfg).double()
    copies = nn.ModuleList([copy.deepcopy(block) for _ in range(3)])
    x = torch.randn(2,6,16,dtype=torch.float64)
    saved = {name:p.detach().clone() for name,p in block.named_parameters()}
    states = [x]
    caches = []
    for _ in range(3):
        state,cache,_ = block(states[-1])
        states.append(state)
        caches.append(cache)
    independent = x
    for layer in copies:
        independent = layer(independent)[0]
    output_error = float((states[-1]-independent).abs().max().detach())
    states[-1].square().mean().backward()
    independent.square().mean().backward()
    gradient_errors = []
    for shared,*others in zip(block.parameters(),*[c.parameters() for c in copies]):
        gradient_errors.append(float((shared.grad-sum(p.grad for p in others)).abs().max()))
    # Remove the output head from the course's analytic full-decoder estimate.
    one_block = replace(cfg,layers=1)
    whole = cost_estimate(one_block,batch=2,length=6)
    head_flops = 2*2*6*cfg.width*cfg.vocab
    per_use = whole['dense_forward_matmul_flops']-head_flops
    unique = parameter_count(nn.ModuleList([block]*3))
    changes = [float((b-a).detach().norm()) for a,b in zip(states,states[1:])]
    checks = {
        'shared_parameter_identity':len({id(p) for p in nn.ModuleList([block]*3).parameters()})==len(list(block.parameters())),
        'stored_weights_unchanged':all(torch.equal(p.detach(),saved[name]) for name,p in block.named_parameters()),
        'copy_parameter_count':parameter_count(copies)==3*unique,
        'forward_copy_equivalence':output_error<1e-10,
        'shared_gradient_sum':max(gradient_errors)<1e-10,
        'states_change':all(v>1e-8 for v in changes),
        'finite_states':all(bool(torch.isfinite(s).all()) for s in states),
        'matrix_work_accounting':per_use==53760,
    }
    trace = dict(task_id='ANIM-LOOP-001',seed=505,device='CPU',dtype='float64',
        torch_version=torch.__version__,config=asdict(cfg),shape=list(x.shape),
        parameters=unique,independent_copy_parameters=parameter_count(copies),
        parameter_shapes={n:list(p.shape) for n,p in block.named_parameters()},
        states=[s.detach().tolist() for s in states],
        heatmap_batch=0,heatmap_abs_max=max(float(s[0].detach().abs().max()) for s in states),
        state_change_norms=changes,forward_copy_max_error=output_error,
        gradient_sum_max_error=max(gradient_errors),
        per_application_dense_matmul_flops=per_use,
        ledger=[dict(applications=i,stored_parameters=unique,dense_matmul_flops=i*per_use) for i in range(4)],
        checks=checks,
        source_sha256=hashlib.sha256((ROOT/'src/dongxi_llms/decoder_lab.py').read_bytes()).hexdigest(),
        limitations=[
            'Untrained random-state mechanism; no quality improvement is measured.',
            'Counts describe the repeated block only, excluding embeddings and output head.',
            'Dense matrix arithmetic counts multiply-add as two FLOPs; normalization, softmax and nonlinear activation work are excluded.',
            'Heatmaps show batch 0 only with one fixed symmetric color scale.',
            'No adaptive routing, runtime benchmark, memory measurement, or cached generation is implemented.',
            'Weights are frozen during the animated forward pass; backward is a verification check, not an optimizer update.'
        ])
    assert all(checks.values()),checks
    return trace


if __name__=='__main__':
    trace=build_trace()
    (PROJECT/'trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    print(json.dumps({k:trace[k] for k in ('parameters','per_application_dense_matmul_flops','state_change_norms','forward_copy_max_error','gradient_sum_max_error','checks')},indent=2))
