# Day 6 — Modern architecture design

- Opened for material preparation: 2026-09-07.
- State: coherent chapter material and three reference notebooks ready; live
  Day 6 learning and independent mastery are not yet established.
- Canonical book: [Chapter 5, sections 5.11–5.22](../../book/chapters/05-building-a-modern-decoder.md#511-modernize-mechanisms-not-just-names).
- [Worked answers 13–24](../../book/solutions/05-decoder-notebook-solutions.md#day-6-modern-decoder--worked-conceptual-solutions).
- [Three notebook sessions](../../notebooks/day-06/README.md), already created
  and pushed in the preceding Chapter 5 notebook work.

## Learning route

1. RMSNorm: RMS scaling versus mean subtraction; shared learned scales;
   epsilon, constant vectors, and per-position statistics.
2. SwiGLU: two expanded branches, SiLU, coordinate-wise modulation, backward
   paths, and fair hidden-width/parameter comparisons.
3. RoPE: coordinate-pair rotation, relative-position dot-product identity,
   frequency choices, cached positions, and incorrect-offset failure.
4. GQA: distinct query heads with shared K/V, summed source gradients, compact
   cache payload versus temporary expansion, MHA/MQA endpoints.
5. Explicit Q/K norm variant: RMS per head, learned scales shared across heads,
   placement before RoPE, retained square-root scaling, and limits of invariance.
6. Compose the modern prototype; account for parameters, cache bytes, dense
   arithmetic, and memory categories; read pinned Qwen configuration fields.
7. Review accounting-only 50M/100M/150M candidates. Introduce fixed/variable/
   adaptive recurrence as a design axis, reserving the defense and trained
   comparison for Day 7 under a separate experiment contract.

## Evidence and boundaries

The three existing Day 6 notebooks passed a fresh readiness run: 43 code cells
and 13 figures, sources unchanged. Original mechanism evidence lives in
`experiments/reports/2026-09-06-decoder-notebooks.md`; latest chapter checks in
`experiments/reports/2026-09-07-day6-chapter-verification.md`.

The learner requested material ahead of guided study, not a claim of completing
the mechanisms. Day 5 understanding and deferred Day 4 practice remain recorded
separately. Next: begin notebook 01 by contrasting LayerNorm with RMSNorm on one
token vector. Use conceptual discussion and controlled examples, not arithmetic
quizzes. Record subsequent learner explanations in this directory.

## Portable reuse

User-led frontier discussion, 2026-09-10:
[DeepSeek V4.1 causal encoder-decoder and KV flow](deepseek-v41-causal-encoder-decoder.md).
Primary report and reference code inspected; first explanation and an inline
layer-dependency schematic prepared. Learner prediction and numerical verification
remain pending. This topic does not rewind or complete the separate Day 8 plan.

Canonical refinements and source decisions:
[modern-decoder-precision-and-evidence.md](modern-decoder-precision-and-evidence.md).
Reuse CAND-ANIM-014–017 for norm/gates/RoPE/GQA and CAND-ANIM-011 for recurrence;
the existing X-LOOP-001 article reminder remains active. All animation production
is approval-gated on Mac Studio; chapter creation is not rendering approval.
