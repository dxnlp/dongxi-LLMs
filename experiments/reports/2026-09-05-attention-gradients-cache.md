# Attention gradients, scaling, and KV-cache verification

- Date: 2026-09-05; mode: smoke, no training, external data, or GPU computation.
- Pre-execution specification and computational source: commit `eb9b0bc`,
  `experiments/specs/2026-09-05-attention-gradients-cache.md`.
- Interpreter: `/home/dongxi/dgx-spark-dongxi/.venv/bin/python`;
  PyTorch `2.13.0+cu130`; CPU float64; scaling seed 40; cache seed 41.
- Reusable code: `src/dongxi_llms/attention_evidence.py`.
- Status: all prescribed criteria passed. No peak-RAM or throughput claim.

## Gradient evidence

The fixed head and target produce loss 2.030520960114874. Maximum absolute
manual/autograd error across Q/K/V/A/scores/X and all three projection matrices
is 2.220446049250313e-16. Central differences (epsilon 1e-6) across every
projection scalar give maximum error 1.8291579362283983e-10.

The forbidden score gradient and future input gradient are exactly zero in
this fixture. Earlier input rows have gradient norm 0.7815602268449181 despite
having no direct target. Undetached projection norms are W_Q=0.3517359856,
W_K=0.3517359856, W_V=0.8898004801.

With routing detached, W_Q/W_K have no gradient and W_V has nonzero gradient.
With values detached, W_V has no gradient and W_Q/W_K have nonzero gradients.
Forward loss is unchanged. These assertions concern the isolated computation;
shared or additional parameter paths can change that interpretation.

## Scaling evidence

Each width uses 256 independent queries and 16 allowed keys, IID normal
coordinates. Scaled/unscaled comparisons reuse identical draws at each width.

| Width | Raw std. | Scaled std. | Raw entropy | Scaled entropy | Raw Jacobian trace | Scaled Jacobian trace |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 2.84406205 | 1.00552778 | 1.24229237 | 2.35528114 | 0.54829739 | 0.86460626 |
| 64 | 7.98888486 | 0.99861061 | 0.42196612 | 2.36164855 | 0.22471650 | 0.86903067 |
| 512 | 22.28491835 | 0.98486356 | 0.11478712 | 2.35443417 | 0.06529190 | 0.86600885 |

Entropy uses natural logs. The Jacobian trace is 1−sum(a²), a local sensitivity
measure, not a training loss gradient. This tests the declared initialization
model and does not generalize the IID assumptions to learned representations.

## Cache evidence

Two residual attention layers, width four, six fixed input rows, and an arbitrary
five-class head. Norms, MLPs, positional encodings, and dropout are omitted.
Input replay is fixed, not free generation. All tested prefill lengths 1/2/4/6
pass the 1e-12 equivalence criterion; length six exercises prefill-only behavior.

For prefill length two, each of the four incremental/full-prefix maximum logit
errors is 0.0. Logical cache bytes grow 384 → 512 → 640 → 768. Final per-layer
K and V shapes are [6,4]. These are payload sizes, not peak allocation.

Changing the first input while keeping the second fixed leaves that second
position's first-layer key unchanged (max error 0.0), while its second-layer key
changes (max error 2.0491642070647273). Reusing the stale original prefix causes
max logit error 1.8069073123097426.

For each K or V projection, cached replay processes 12 layer-position rows;
full-prefix recomputation processes 40. These counts follow the loop geometry,
not a timer. They do not establish a 40/12 wall-clock speedup.

## Tests and notebook execution

The full repository suite passed: 26 unittest tests, process exit code 0.
Five new cases cover manual derivatives and finite differences, detach paths,
scaling, cache equivalence and stale prefixes, and a redundant-value example.
Three earlier tests cover forward masking and SDPA agreement.

Both new notebooks passed nbformat validation and fresh-kernel execution using
nbclient with `dgx-spark-native`, timeout 60 seconds per cell, and their parent
directory as execution working directory. Each executed 12 code cells without
error outputs. Reference outputs are kept out of learner notebooks so predictions
remain editable. Existing open learner notebook edits were not overwritten.

The committed canonical session 1 was also re-executed from an in-memory Git
snapshot: all 13 code cells passed. This validates all three reference notebooks
without changing either open learner notebook. Chapter, solution, and index
local-file links were checked and all resolved.

## Reproduction

Run from the repository root:

```bash
PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m unittest discover -s tests -v
```

Inspect `gradient_evidence()`, `scaling_evidence()`, and `cache_evidence()` in the
reusable module, or execute notebook sessions 2 and 3 in a fresh platform kernel.

## Limits

This package verifies mathematical mechanisms on small fixtures. It does not
train a model, identify semantic roles in real heads, assess learner mastery,
benchmark serving systems, or validate cache handling for positions, quantized
storage, beams, or modified model parameters. Those require further experiments.
