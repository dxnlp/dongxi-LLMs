# Visual plan — static review first

The article and concept figures are approved for local production. The following
motion treatments are proposals, not completed animations or model evidence.
Reuse the white canvas, normal Arial/Songti and stable geometry in
`visuals/animations/STYLE_GUIDE.md`. Existing KV videos remain unchanged.

| Slot | Static figure in this package | Proposed motion | Critical invariant |
|---|---|---|---|
| 01 | Request timeline, TTFT and ITL | First-token marker, then output markers; compare delay before/after start | First token follows prompt computation; network stream chunks may group tokens; widths not benchmarks. |
| 02 | Two batch membership snapshots | A finishes, B stays fixed, C joins after its input processing | Distinct request state; admission subject to resources. |
| 03 | Chunked schedule vs disaggregated workers | Interleave P/D chunks; separately move KV from prefill worker to decode worker | No discarded context, no zero-cost transfer, chunking and placement are different axes. |
| 04 | Draft/verify/correct | Accept a/b; reject c; remove dependent d; produce x | No semantic fact-check, no guaranteed fixed acceptance length; exact sampling rules separately validated before production. |
| 05 | 16-bit/4-bit weight payload bars | Change numeric representation while shrinking calculated payload | Same parameter count; metadata/KV excluded; no runtime speed or quality implication. |
| 06 | One sharded model vs two replicas | A traverses communicating shards; A/B separately use replicas | TP communication vs replica routing; no automatic speed multiplier. |
| 07 | Optimization map | Keep static | Better suited to reference than motion. |

Suggested first animation pass: 02 continuous batching, 03 phase scheduling,
04 speculation. These provide the largest new explanatory value relative to the
existing KV-cache assets. Latency metrics can reuse their visual grammar; a new
video is not needed for every section. Export MP4 and GIF only after the motion
scope and precision checks are approved; retain them locally with source/hash records.

Cover: original code-native 2000×800 design, single-line “LLM Inference”, abstract
parallel request tracks. This is a first concept, not a claim of final cover approval.
