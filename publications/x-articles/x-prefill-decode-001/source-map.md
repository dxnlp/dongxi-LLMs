# Source and evidence map

Checked 2026-09-13. Public prose contains no external links; this production
record retains attribution. Diagrams are independently drawn schematics, not
copied paper figures or measured performance plots.

| Claim / section | Primary basis | Qualification |
|---|---|---|
| Parallel known positions, causal invariance, cache offsets; §§2–3 | Local Chapter 4 §§4.10–4.11; [HF cache explanation](https://huggingface.co/docs/transformers/v4.50.0/en/cache_explanation) | One decoder-only causal model; per-layer KV. Attention layers remain sequential. |
| Last prompt position predicts y1; feeding y1 creates its KV | Local Chapter 3 next-token alignment; [HF generation](https://huggingface.co/docs/transformers/main_classes/text_generation) | Standard autoregressive generation; output selection is distinct from next forward. |
| TTFT versus ITL; §1 | [vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics/) | Client/server boundaries and stream chunks differ. No timing claim. |
| Projection shapes, weight reuse, history reads; §4 | Local Chapter 5 §5.18; [DistServe, OSDI 2024](https://arxiv.org/abs/2401.09670) | Compute/bandwidth are workload-dependent tendencies. Full attention assumed for history growth. |
| Chunking; §5 | [vLLM optimization](https://docs.vllm.ai/en/latest/configuration/optimization/) | Sketch is iteration scheduling; actual implementations may mix decode and prefill chunks. No default/version promise. |
| Separate workers, SLO and transfer tradeoffs; §5 | [DistServe](https://arxiv.org/abs/2401.09670); [vLLM disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/) | Goodput under latency SLOs differs from raw throughput. Neither universal improvement nor universal non-improvement claimed. |
| Prefix reuse; §5 | [vLLM APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | Exact compatible retained prefix, not semantic similarity; uncached suffix still needs computation. |
| Spark → M3 Ultra, layerwise overlap; §6 | [EXO first-party demonstration](https://blog.exolabs.net/nvidia-dgx-spark/) | Third-party demonstration, not our experiment or current general availability guarantee. No benchmark numbers reproduced. |

EXO's simplified unconditional compute/bandwidth labels are not repeated. Its
headline and table contain differing speedup figures; none are needed here.
No extrapolation to the learner's M1 Max hardware. A deployment guide would need
a fresh pinned-release and platform-support audit.

## Local checks and limits

`check.py` uses a deterministic two-layer causal attention fixture to compare
parallel prefill + cached continuation with full-prefix recomputation. It checks
the N → N+1 cache boundary and prompt/next-position output equality. This is a
tiny numerical sanity check, not a trained LLM, quality test or serving benchmark.
For the parallel-position illustration, it also computes attention row 2 before
row 1 and verifies equality with batched causal attention. Row outputs never
serve as same-layer K/V inputs. Norm/position and other block operations are
omitted from this dependency schematic, not from the general Transformer claim.

`metadata.json` records versions, dimensions and source/output hashes. `qa.json`
records structural and mechanism results. All figures use schematic token IDs;
no claim that Chinese words each map to one token. The selected “延期” token
and subsequent punctuation are explicitly hypothetical, not tokenizer/model
observations. The revised workload figure contrasts input/output volume only;
its line counts are illustrative. Scheduling and transfer widths encode order only; overlap
requires adequate compute time, link capacity and implementation support.

## Animation return — 2026-09-14

The same seven mechanisms are now independently authored Manim loops in
`visuals/animations/projects/prefill-decode-article/`. Durations, line counts and
packet travel are schematic. Decode shows two illustrative layers and two steps;
other block operations and vocabulary-head/selection details are compressed.
The scheduling transition compares two arrangements, without claiming an actual
request switched deployment mid-run. Handoff is a compatible-state illustration
of the cited third-party demonstration, not a local Spark/Mac measurement.

Event traces test state/order invariants; numerical checks test the toy attention
mechanism separately. Render hashes establish reproducibility, not experimental
performance. Preview contact sheets and final rendered frames were inspected.
HTML controls have DOM-stub unit tests; browser playback/layout remains unverified
because the earlier local-URL policy restriction was respected.
