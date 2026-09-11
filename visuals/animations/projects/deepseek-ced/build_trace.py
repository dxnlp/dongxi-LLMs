"""Independent, untrained CPU microscopes; not DeepSeek model execution."""
import hashlib
import json
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def main():
    rng = np.random.default_rng(410)
    lower = [rng.normal(0, .3, (3, 4, 4)) for _ in range(2)]
    upper = [rng.normal(0, .3, (4, 4, 4)) for _ in range(3)]
    wg = rng.normal(0, .3, (4, 4))
    x = rng.normal(0, .5, (3, 4))

    def run(x):
        h = x.copy()
        for wq, wk, wv in lower:
            q, k, v = h @ wq, h @ wk, h @ wv
            scores = q @ k.T / 2
            scores[np.triu_indices(len(h), 1)] = -np.inf
            h = np.tanh(h + softmax(scores) @ v)
        enc = h.copy()
        bank = enc @ wg
        before = bank.copy()
        frames = []
        banks = []
        for wq, wl, wo, wf in upper:
            q, local = h @ wq, h @ wl
            banks.append(bank)
            out = []
            for t in range(len(h)):
                pool = np.concatenate([bank[:t+1], local[max(0,t-1):t+1]])
                out.append(softmax(q[t] @ pool.T / 2) @ pool)
            nxt = np.tanh(h + np.array(out) @ wo + np.tanh(h @ wf))
            frames.append(dict(input=h.tolist(),q=q.tolist(),local=local.tolist(),output=nxt.tolist(),reference_kv=(h @ wl).tolist(),ced_projection=(enc @ wl).tolist()))
            h = nxt
        return enc,bank,frames,h,all(b is bank for b in banks),np.array_equal(bank,before)

    enc, bank, frames, out, identity, unchanged = run(x)
    _, _, longer, out_long, _, _ = run(np.vstack([x, rng.normal(size=4)]))
    c = np.array([[1.,0.],[0.,1.],[-1.,0.]])
    q = np.array([[2.,0.],[0.,2.],[-2.,0.]])
    scores = q @ c.T / np.sqrt(2)
    weights = softmax(scores)
    outputs = weights @ c
    copies = softmax(q @ c.copy().T / np.sqrt(2)) @ c.copy()
    altered = c.copy(); altered[0,1] += 1
    checks = dict(shared_storage=identity,bank_unchanged=unchanged,
        causal_prefix=bool(np.allclose(out,out_long[:3],atol=1e-12)),
        distinct_layer_outputs=all(not np.allclose(f['input'],f['output']) for f in frames),
        weights_normalized=bool(np.allclose(weights.sum(axis=1),1)),
        equal_copies_agree=bool(np.array_equal(outputs,copies)),
        changed_value_breaks_equivalence=not np.allclose(outputs,weights@altered),
        predicted_favorites=weights.argmax(axis=1).tolist()==[0,1,2],
        finite=bool(np.isfinite(outputs).all() and np.isfinite(out).all()))
    assert all(checks.values()), checks
    data=dict(seed=410,evidence='Untrained float64 NumPy toy; no DeepSeek weights',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numpy=np.__version__,
        ced=dict(input=x.tolist(),encoder=enc.tolist(),bank=bank.tolist(),layers=frames,local_window=2),
        roles=dict(c=c.tolist(),q=q.tolist(),scores=scores.tolist(),weights=weights.tolist(),outputs=outputs.tolist()),checks=checks)
    (P/'trace.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(checks=checks,weights=weights.tolist(),outputs=outputs.tolist()),indent=2))

if __name__=='__main__':main()
