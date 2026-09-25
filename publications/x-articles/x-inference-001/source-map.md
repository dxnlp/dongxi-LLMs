# Source and precision map

Primary sources checked 2026-09-13. Public copy has no hyperlinks by user
preference; this production record retains attribution. No external figure,
substantial code, or prose is copied. All diagrams are original schematics.

| Article claim | Primary source | Boundary used in draft |
|---|---|---|
| Fixed-weight generation, logits, prefill and append | Repository Chapters 3 and 4 §4.10; [Transformers generation](https://huggingface.co/docs/transformers/main_classes/text_generation) | Standard autoregressive text generation; inference is distinguished from reasoning. |
| KV payload and weight payload are different quantities | Repository Chapter 5 §5.18 | 8 billion × bits / 8 is calculated payload, not measured allocation. |
| TTFT, token latency, E2E and percentile reporting | [vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics/) | Client/server boundaries differ; network chunks are not necessarily tokens; no timings measured. |
| Requests can leave/join an active generation batch | [Transformers continuous batching](https://huggingface.co/docs/transformers/main/continuous_batching) | Resource constraints and new-request prefill still apply. No claim every scheduler instantly admits every waiting request. |
| Paged KV reduces allocation waste and enables sharing | [PagedAttention paper](https://arxiv.org/abs/2309.06180) | Does not eliminate attention reads or all fragmentation. No paper speedup reproduced or quoted. |
| Exact-prefix reuse skips matching prefill | [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | Actual compatible prefix and resident usable cache required; new suffix and output remain work. |
| Chunked prefill interleaves with decode | [vLLM optimization](https://docs.vllm.ai/en/latest/configuration/optimization/) | Scheduling chunks retain causal context; TTFT/ITL trade-off; no fixed chunk-size recommendation. |
| P/D separation isolates phases, transfers KV | [vLLM disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/) | Official page explicitly says separation does not improve throughput; draft frames separate latency tuning, transfer and balancing cost. Feature page labels it experimental. |
| Prefill often compute-heavy, small-batch decode bandwidth-limited | [vLLM optimization](https://docs.vllm.ai/en/latest/configuration/optimization/) plus repository Chapter 5 §5.18 | Workload-dependent tendency, not universal classification; no measured roofline. |
| Draft/verify/accept-prefix/reject-suffix | [Leviathan et al., ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html) | Exact speculative sampling requires correct acceptance and correction. Distribution preservation differs from identical sampled text. |
| Speculation depends on load and acceptance economics | [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/) | No universal speedup; implementation numerics may vary. Letter sequence and acceptance outcome are illustrative. |
| Quantization changes numerical representations | [Transformers quantization overview](https://huggingface.co/docs/transformers/main/quantization/overview), [vLLM quantization](https://docs.vllm.ai/en/latest/features/quantization/) | Precision, quality and hardware/kernel support; smaller payload is not a guaranteed speed gain. |
| KV quantization is separate from weight quantization | [vLLM quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) | Independent format/scaling/backend requirements; no blanket claim about FP8 latency. |
| TP shards layer tensors; PP partitions layers | [vLLM parallelism](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) | Communication and synchronization costs; no linear-scaling promise. |
| Replicas serve distinct requests, may internally use TP/PP | [vLLM data parallel deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/) | Simple dense-model explanation; MoE DP/EP hybrid implementation is explicitly not generalized. |

## Evidence classes

- **Derived/canonical:** causal cache dependency, per-weight bit accounting.
- **Source-reported mechanism:** scheduling, speculation, phase placement, parallelism.
- **Illustrative:** 50 万 project scenario, 2 s vs 0.3 s experience, A/B/C scheduling,
  accepted a/b and rejected c/d, diagram widths except weight-payload bars.
- **Calculated:** exactly 8,000,000,000 × 16/8 = 16,000,000,000 bytes and × 4/8 =
  4,000,000,000 bytes (decimal GB); no scales/alignment/unquantized tensors/KV.
- **Not done:** inference run, acceptance-rate test, quality evaluation, latency or
  throughput benchmark, multi-GPU deployment, browser/cloud upload unless later recorded.

Source pages are live documentation, not pinned claims about a locally installed
vLLM version. Commands/defaults and current feature compatibility are deliberately
excluded from the article. Before turning this into an executable lab, pin model,
tokenizer, engine revision, hardware, prompt/output lengths, cache state, load,
sampling and quality criteria. Do not reuse illustrative timings as results.
