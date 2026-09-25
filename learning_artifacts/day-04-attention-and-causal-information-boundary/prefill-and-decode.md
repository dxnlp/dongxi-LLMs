# Prefill and Decode: phase-focused public explanation

Discussion 2026-09-13; local article return 2026-09-14. Public derivative of
Chapter 4 §§4.10–4.11 and Chapter 5 §5.18, not a new serving lab or mastery claim.

## Learner direction and refined model

After the broader inference overview, the learner selected a standalone article
named “Prefill and Decode.” The mechanism is central. A Spark/Mac deployment
example should support it without becoming the topic. Use an everyday long
document request, short paragraphs, retained English AI terminology, and minimal
English static diagrams before selecting animation work.

- Known prompt positions can execute in parallel within each layer while causal
  visibility remains restricted. Layers still depend on earlier layers.
- The last prompt state predicts the first output token. Selecting y1 does not
  add its KV; feeding y1 through the model does. This prevents an off-by-one
  animation error at the boundary between the two phases.
- The same model weights participate in both phases. Projection shapes and
  batching affect weight reuse and arithmetic intensity; “compute versus
  bandwidth” is a conditional tendency, not an architectural law.
- Chunking changes iteration scheduling; disaggregation changes placement;
  prefix reuse skips already available compatible input computation.
- A handoff needs compatible KV and generation state. Layerwise transfer can
  overlap later computation but retains a tail and conversion/coordination cost.

## Evidence and boundaries

Package: `publications/x-articles/x-prefill-decode-001/`. `source-map.md` maps
primary references. `check.py` verifies an untrained float64 two-layer causal
attention fixture: cache length 4 before/after y1 selection, 5 after feeding y1;
cached continuation and two-token-chunk prefill match complete-prefix results.
Removing causality changes prior states after an append. This is a numerical
mechanism check, with no trained output, throughput or hardware comparison.

Seven static figures, a 5:2 cover and local Chinese HTML are ready for review.
No edits to the existing KV or inference articles. A full serving curriculum
remains deferred beyond v0.1; active Day 8 and practice backlog are unchanged.
Animation opportunities extend CAND-ANIM-009; no video rendering or X upload.
Latest learner direction, 2026-09-14: the Decode section now includes a static
token-feedback/KV-append illustration; all animations will be produced together
later after review. No animation production was started.

## Parallel-position clarification

The learner asked what “simultaneously” means and why position 2 need not wait
for position 1's output. The revised explanation separates incoming h1/h2,
input-derived Q/K/V, and attention outputs O1/O2: O2 reads Q2 and K1/V1,K2/V2,
with no O1 dependency. Batch execution is not literal simultaneous completion.
Layers retain their dependency through block outputs. New static figure in
article slot 02; numerical reverse-row evaluation agrees with batched attention.
Understanding was discussed and accepted for editorial revision; no independent
learner implementation or mastery assessment is claimed. Motion deferred.

## Example-led public adaptation

The learner found the technical draft dry and requested an approachable revision
like the prior KV article. Public prose now uses a report before a meeting,
a hypothetical “延期” then punctuation continuation, a long-speech comparison,
and an ongoing answer A sharing resources with incoming document B. Two stills
were adjusted; the recently clarified same-layer dependency is preserved.
Examples explain existing mechanisms without new model outputs or benchmarks.
All animation remains deferred for a later combined production pass.

## Approved animation return — 2026-09-14

The learner has now approved the combined production pass, superseding the
earlier deferrals above. Seven independent loops rendered in
`visuals/animations/projects/prefill-decode-article/`, with local MP4/GIF exports,
static fallbacks and article integration. Input-derived dependencies, two-step
per-layer KV growth, independent A/B state and the transfer tail are preserved.
Toy numerical and media/event checks pass; final frames inspected. No new
experiment or learner mastery claim, no browser playback verification, and no X
upload. Existing KV/inference articles and active Day 8 remain unchanged.

Next: review the animated article. Open empirical question: whether a
specific model/runtime/network makes phase separation worthwhile on the learner's
actual hardware. That would require a separately approved experiment, not an
inference from the third-party EXO demonstration.
