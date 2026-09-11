"""Three distinct course blocks, reused in two passes; measured CPU fixture."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn

P=Path(__file__).resolve().parent
ROOT=P.parents[4]
sys.path.insert(0,str(ROOT/'src'))
from dongxi_llms.decoder_lab import DecoderConfig,DecoderBlock,parameter_count,cost_estimate

def build():
    torch.set_num_threads(1);torch.manual_seed(505)
    cfg=DecoderConfig()
    blocks=nn.ModuleList([DecoderBlock(cfg).double() for _ in range(3)])
    copies=nn.ModuleList([copy.deepcopy(blocks[i%3]) for i in range(6)])
    saved=[p.detach().clone() for p in blocks.parameters()]
    x=torch.randn(2,6,16,dtype=torch.float64)
    states=[x]
    for i in range(6):states.append(blocks[i%3](states[-1])[0])
    other=x
    for block in copies:other=block(other)[0]
    forward_error=float((states[-1]-other).abs().max().detach())
    states[-1].square().mean().backward();other.square().mean().backward()
    errors=[]
    for i in range(3):
        for p,a,b in zip(blocks[i].parameters(),copies[i].parameters(),copies[i+3].parameters()):
            errors.append(float((p.grad-a.grad-b.grad).abs().max()))
    counts=[parameter_count(b) for b in blocks]
    ids=[{id(p) for p in b.parameters()} for b in blocks]
    matmul=cost_estimate(cfg,batch=2,length=6)['dense_forward_matmul_flops']
    per_block=(matmul-2*2*6*cfg.width*cfg.vocab)//cfg.layers
    checks=dict(three_distinct_parameter_sets=all(ids[i].isdisjoint(ids[j]) for i in range(3) for j in range(i)),
        distinct_weight_values=all(not torch.equal(blocks[i].attn.q.weight,blocks[j].attn.q.weight) for i in range(3) for j in range(i)),
        shared_storage=parameter_count(nn.ModuleList([blocks[i%3] for i in range(6)]))==6480,
        independent_storage=parameter_count(copies)==12960,
        forward_copy_equivalence=forward_error<1e-10,
        paired_gradient_sum=max(errors)<1e-10,
        weights_unchanged=all(torch.equal(p.detach(),s) for p,s in zip(blocks.parameters(),saved)),
        states_change=all(float((b-a).norm().detach())>1e-8 for a,b in zip(states,states[1:])),
        finite_states=all(bool(torch.isfinite(s).all()) for s in states))
    assert all(checks.values()),checks
    return dict(seed=505,device='CPU',dtype='float64',torch_version=torch.__version__,
        source_sha256=hashlib.sha256((ROOT/'src/dongxi_llms/decoder_lab.py').read_bytes()).hexdigest(),
        shape=list(x.shape),states=[s.detach().tolist() for s in states],
        heatmap_abs_max=max(float(s[0].detach().abs().max()) for s in states),
        application_block_ids=['A','B','C','A','B','C'],
        block_parameter_counts=counts,shared_parameters=parameter_count(blocks),
        independent_parameters=parameter_count(copies),per_block_matmul_flops=per_block,
        six_application_matmul_flops=6*per_block,forward_copy_max_error=forward_error,
        gradient_sum_max_error=max(errors),checks=checks)

if __name__=='__main__':
    trace=build();(P/'trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    print(json.dumps({k:v for k,v in trace.items() if k not in ('states','shape','source_sha256')},indent=2))
