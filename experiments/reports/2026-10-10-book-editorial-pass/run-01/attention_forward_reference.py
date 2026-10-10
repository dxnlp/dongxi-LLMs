"""Regenerate Chapter 4's calculated fixture with the canonical attention trace.

Run from the repository root with PYTHONPATH=src and an isolated CPU interpreter.
This checks illustrative arithmetic; it does not train a model or modify evidence.
"""
import json
from pathlib import Path
import torch
from dongxi_llms.causal_attention_lab import attention_trace

q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
v = torch.tensor([[1., 2.], [3., 0.], [0., 4.]], dtype=torch.float64)
trace = attention_trace(q, q, v)
expected = json.loads(Path(__file__).with_name("attention-forward-calculation.json").read_text())
for name, actual in [("weights", trace["weights"]), ("outputs", trace["output"])]:
    torch.testing.assert_close(actual, torch.tensor(expected[name], dtype=torch.float64),
                               atol=1e-15, rtol=0)
changed = v.clone()
changed[-1] = torch.tensor([100., -100.], dtype=torch.float64)
intervention = attention_trace(q, q, changed)["output"]
torch.testing.assert_close(trace["output"][:2], intervention[:2], atol=0, rtol=0)
assert not torch.equal(trace["output"][-1], intervention[-1])
print(json.dumps({"weights": trace["weights"].tolist(),
                  "outputs": trace["output"].tolist(),
                  "causal_value_intervention": "passed",
                  "scope": "Calculated CPU teaching example, no empirical model claim"}))
