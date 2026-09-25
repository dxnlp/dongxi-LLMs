# From causal cache to inference systems

Discussion and public-article production, 2026-09-13. This is a source-grounded
concept map, not a completed serving lab or new learner mastery assessment.

## Learner direction

After the KV article, the learner requested a broader inference article. A short
generation primer should lead into latency, concurrency, memory and deployment
trade-offs, rather than spending the article on token selection. Use a project
document throughout, short paragraphs, accurate AI terms, and minimal figures.

## Refined distinctions

- TTFT includes more than prefill at client measurement boundaries. Token ITL,
  visible stream chunks and time to first answer text may differ.
- Continuous batching changes request membership; requests retain separate state.
- Prefix caching skips reusable input computation. Chunking schedules input work
  in pieces while retaining context. Phase separation changes resource placement
  and adds KV transfer; these are distinct, potentially composable mechanisms.
- Speculative verification concerns the target generation distribution, not truth.
  Correct exact acceptance/correction differs from an arbitrary quality filter.
- Weight and KV quantization affect different allocations. Calculated low-bit
  payload is not total runtime memory, quality or a speed measurement.
- Sharding distributes one model; replicas distribute independent request work.
  Communication, synchronization and load balance constrain gains.

## Evidence and reuse

Canonical basis: Chapter 4 §§4.10–4.11 and Chapter 5 §5.18. Detailed source links
and qualifications are preserved in
`publications/x-articles/x-inference-001/source-map.md`.
Only the ideal weight-payload arithmetic is calculated; all scheduling timelines
and draft-acceptance outcomes are schematic. No inference, quality or throughput
experiment was run. General serving material remains a bounded public extension,
not a silent addition to the v0.1 curriculum or a replacement for Day 8.

Next: learner reviews the local article and static figures; if motion is selected,
prioritize batching membership, P/D scheduling and speculative acceptance. Each
requires a precise storyboard/fixture before rendering on Mac.
