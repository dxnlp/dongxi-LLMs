# Identical rollouts expose different objective gradients

The three-seed CPU reference compares nine objectives on unchanged first-attempt
responses and unchanged shared-decoder weights. Analytical selected-log-probability
derivatives match autograd exactly. Different reductions and reward-scaling rules
produce different parameter gradients. This is a measured objective comparison,
not training, a capability gain or a full named-algorithm reproduction.

The [specification](../specs/2026-10-04-grpo-objective-controls.md) precedes collection.
[Final raw evidence](2026-10-04-grpo-objective-controls-final.json) retains every generated
path, component, strict-quality label, stop/cap, conditional behavior score,
gradient matrix, seed and retry. [Closing verification](2026-10-04-grpo-objective-controls-closing-verification.json)
binds current source/test hashes, eighteen tests, exact replay and fresh visual notebooks.
The original raw ledger and seventeen-test verification remain unchanged.
Historical G4/G8 and constructive-control reports remain separate and unchanged.

## What actually ran

Seeds 2401, 2402 and 2403 each initialize a float64 CPU TinyDecoder with one shared
attention/MLP block, width 16, four query heads, two KV heads and ten output IDs.
Five symbolic sum-positive prompts each receive eight first-attempt responses,
cap three. Every generated position conditions on EOS, 0 and 1. Immediate EOS,
repeated numerals and missing EOS are actual unforced outcomes. The grammar
provides syntax support, not the correct answer or a free stopping token.

The first-attempt comparison pools contain 79, 81 and 82 valid actions. The same
pool and parameters serve every objective for that seed. All nine initial
selected-action ratios equal one. All model state hashes stay unchanged.

The proxy has answer-only, complete-format and verbosity components with fixed
weights 1, 0.4 and 0.2. Independent quality requires the correct numeral followed
by EOS. Normalizing their weighted total is not the same as normalizing each
component before aggregation. No held-out accuracy is inferred from gradients.

For seed 2401, centered-reward parameter-gradient norms are 0.634216 with response
means, 0.484715 with token means and 0.319104 with a fixed-cap denominator. The
corresponding response-mean scalar loss is approximately zero. The nonzero
gradient demonstrates why centered rewards can cancel as numbers without their
shared parameter contributions canceling. Gradient size is not method quality.

Token-mean and fixed-cap directions agree because their numerators agree; the
fixed denominator changes scale. Response means can change direction. The saved
cosine matrix exposes that distinction, while all three seeds remain visible.
Selected-score derivative maximum error is zero across all 27 actual rows.

## A clipping fixture is not a model result

At ratio one, symmetric and asymmetric clipping agree in every measured pool.
A separately labeled constructed score fixture covers ratios 0.6, 0.9, 1.1,
1.3 and 1.6 for advantages +1 and −1. Raising only the upper clip boundary from
1.2 to 1.4 restores positive-advantage pressure at ratio 1.3 without moving the
negative-advantage lower boundary. Independent known derivatives, padding with
nonfinite inactive scores, constant components and detached old/reward paths
are tested. No optimizer or hidden KL term is part of this matrix.

## Selected samples are not the collection budget

Retry only unresolved prompts and stop after at most three groups each. The
first mixed independent-quality group is selected. Rejected groups stay in the
ledger and their generation cost stays in the denominator.

| Seed | Attempted groups | Attempted responses | All valid actions | Selected valid actions |
|---:|---:|---:|---:|---:|
| 2401 | 9 | 72 | 144 | 86 |
| 2402 | 9 | 72 | 142 | 86 |
| 2403 | 7 | 56 | 116 | 89 |

Each seed ultimately selects five groups, forty responses. In total, 200
responses and 402 actions were attempted; only 120 responses and 261 actions
were selected. The separate all-constant test exhausts the budget and selects
nothing. These particular successful seeds cannot guarantee qualification.

The uncached generation-position counts include finished rows still forwarded
inside a batch. Collection and objective rescoring add work and are not hidden
inside the generation count. These positions are not FLOPs, serving latency or
a hardware benchmark. Filtering is never substituted into the fixed objective
pool, so data-selection effects do not masquerade as loss effects.

## Reproduce and interpret

```bash
/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python -m unittest discover \
  -s tests -p test_grpo_objective_controls.py -v
/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  -m dongxi_llms.grpo_objective_controls --output /tmp/NEW-objective-controls.json
```

Both actual commands exit 0. Use a new report path; the collector rejects
overwriting prior evidence. The isolated Linux ARM64 environment has Python
3.12.14 and Torch 2.14.1+cpu, not a CUDA build. The full three-seed mechanism
computation records its actual short runtimes; those are not model-scale estimates.

The [Day 22 notebook](../../notebooks/day-22/03_objective_weighting_and_filtering.ipynb)
executes six complete reference cells and emits five figures in a fresh matching
CPU kernel. Root visual review found overlapping component labels, shortened
only the displayed labels and reran into a new verification directory. Exact
coefficient maps, actual gradient norms/cosines, constructed clipping derivatives
and actual retry costs remain labeled separately. No animation was rendered.

An eighteenth regression tests raw component amplitude rescaling separately from
rescaling a component's objective coefficient. Recollecting with the unchanged
module exactly reproduces all three campaigns apart from elapsed seconds.
Independent review repeats all eighteen tests, replays the full campaign and
executes another fresh six-cell/five-image notebook. This supplements the
original measurement rather than rewriting its earlier source identity.

The module and questions are original. Pinned upstream loss and advanced-objective
pathways motivate the comparisons, but are not imported as correctness oracles.
This evidence does not establish pretrained reasoning, unrestricted generation,
long-run optimization, Mac/GPU execution or learner mastery.
