# Chapter 5 Hands-On Pathway and Evidence

- Date: 2026-09-06.
- Learner request: separate hands-on notebooks for the architecture components,
  then build the entire sequence now so we can study it one session at a time.
- Delivered: seven baseline notebooks, three modern-architecture notebooks, and
  an optional fixed recurrent-depth notebook. Each has safe attempt cells,
  predictions, adjacent runnable solutions, explanations, and controlled changes.
- Canonical index: `notebooks/day-05/README.md`.
- Book placement: `book/labs/05-building-a-modern-decoder.md` and
  `book/solutions/05-decoder-notebook-solutions.md`; integrate into Chapter 5's
  full narrative during Day 7 synthesis.

## Mechanisms made inspectable

Lookup rows and their repeated-token gradients; positions versus causal
structure; per-head/vectorized attention; residual identity and cancellation;
LayerNorm versus wrong-axis normalization; pre/post-norm placement; positionwise
MLPs and affine collapse; full-model shapes, tying, causality, cache replay and
label shifting; bounded one-batch fitting; RMSNorm and SwiGLU ablations; RoPE
geometry and cache-offset failure; GQA, QK norm, parameter/FLOP/cache accounting;
and fixed repeated-block gradient accumulation.

The modern tiny decoder is not a Qwen checkpoint loader. Its configuration
mapping uses a pinned Qwen3 primary source and explicitly separates residual
width from concatenated query-head width. The recurrence notebook implements
fixed sharing only, not adaptive routing or a trained architecture comparison.

## Evidence and learning state

- Eleven fresh-kernel reference runs passed, totaling 93 executed code cells.
- 38 repository tests passed, including 12 new decoder tests.
- Fixed 160-step seed-505 CPU experiment: CE 2.774792432785034 to
  0.0007856183219701052, training-token accuracy 1.0. This verifies fitting the
  declared batch, not language capability or generalization.
- Specification/report: `experiments/{specs,reports}/2026-09-06-decoder-notebooks.md`.
- Learner has not yet completed the pathway. Day 4 practice stays deferred.
- Existing Day 3/4 learner edits were preserved and excluded from this work.
- Next: notebook 01, first lookup prediction and implementation.

## Reuse and portability

Math opportunity check expands CAND-ANIM-011/012/013 and captures 014–017 for
normalization, MLP gates, RoPE, and grouped KV accounting. Existing embedding and
attention packets remain canonical. All media production requires approval and
stays on Mac Studio. The looped-transformer article reminder remains X-LOOP-001.

CPU-only references require Torch, nbformat/nbclient for automated verification,
and a working Jupyter kernel for interactive study. Spark's kernel is
`dgx-spark-native`; Mac users should choose their local Python kernel. No Mac
execution is claimed. No notebook server was launched for this production task.
