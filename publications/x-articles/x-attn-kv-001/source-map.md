# Claim and visual provenance

Checked 2026-09-11. Repository provenance pins `d1374d1`; DeepSeek
citations pin `df42c109f1defefcbfcedbe7d905718a12266e40`.
Article bodies contain no links; provenance is maintained separately here.
On 2026-09-12, three typeset equation cards were added without changing the
attention mechanism or numerical claims. Original diagram numbers below refer
to the five diagram assets; upload slots now include the three equation cards.
Later on 2026-09-12, the user removed the experiment section in both languages.
The toy-result and projection-count entries below now document retained course
assets only; neither appears in the article upload plan.
The first static diagram was subsequently replaced by the user-approved
`ANIM-KV-001` prefill/decode GIF. Its source, event-order/geometry assertions,
render manifest and limitations are in `visuals/animations/projects/kv-prefill-decode/`.
This is a schematic animation, not a new model experiment or timing measurement.
User approval subsequently promoted append/edit and global sharing to
`ANIM-KV-003/004`, replacing article slots 3 and 6 with GIFs. Sources and checked
event records are in `visuals/animations/projects/kv-article-loops/`. The first
loop is unchanged. The original diagrams remain in the asset inventory for reuse.
The user then approved `ANIM-KV-005` for slot 5: memory growth and a separate
KV-head configuration comparison. `visuals/animations/projects/kv-memory-growth/`
retains calculated arithmetic, bar-ratio/geometry checks and render provenance.
Each grid column groups 1,024 positions; the layer/batch/feature axes are omitted
visually and included numerically. No measured GPU allocation is implied.
The user subsequently approved `ANIM-KV-006` to replace the code section with
one decode-step close-up. The current plan has seven inline assets: this loop
is slot 4, the memory equation/growth and global-sharing slots shift to 5/6/7.
`visuals/animations/projects/kv-decode-step/example.py` preserves the NumPy
implementation and exact toy fixture; source/media/timeline checks live beside it.
Both language versions are shortened and code-free; no model run is implied.

| Article claim / visual | Source | Evidence boundary |
|---|---|---|
| Q/K matching, V mixture, prefix invariance, first-layer qualification, one-token/chunk distinction | `book/chapters/04-attention-and-the-causal-information-boundary.md`, §§4.2, 4.10–4.11 | Derived for causal inference with compatible inputs/positions/weights/settings |
| Prefill/decode schedule; Figure 1 | Same chapter; `learning_artifacts/day-04-attention-and-causal-information-boundary/why-cache-keys-and-values.md` | Sentence pieces and predictions are illustrative, not tokenizer/model observations |
| Append versus edit; Figure 2 | Chapter 4 §4.10; Chapter 5 §5.15 | Dependency schematic; first-layer vectors need not change with preceding text |
| Zero recorded cached/full error; stale-prefix max error 1.8069073123097426; Figure 3 | `experiments/reports/2026-09-05-attention-gradients-cache.md`, Cache evidence; `src/dongxi_llms/attention_evidence.py` | Recorded CPU float64 toy with omitted MLP/norm/position/dropout; fixed replay, no new run |
| 12 vs 40 rows per projection | Same report; builder asserts `2*(2+1+1+1+1)` vs `2*(2+3+4+5+6)` | Count includes initial prefill; not FLOPs or latency |
| 192/384/96 MiB; Figure 4 | Chapter 5 §5.18; recomputed in `build.py` | Hypothetical logical separate-K/V tensor payload, not GPU measurement |
| GQA | Chapter 5 §5.16, [original paper](https://arxiv.org/abs/2305.13245) | Architecture definition, not a claim of measured quality or speed |
| Runtime incremental update | [Hugging Face cache explanation](https://huggingface.co/docs/transformers/cache_explanation), live checked 2026-09-11 | Refer only to update mechanics; masking derivation follows canonical course, not the docs' schematic equation |
| Cross-request exact-prefix identity | [vLLM prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/), live checked 2026-09-11 | Parent-block identity, token IDs and additional identifying state; avoid defaults/configuration claims |
| Chinese prefill / KV cache / prefix caching distinction and repeated-document example | [vLLM Automatic Prefix Caching: Example workloads and Limits](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/), live checked 2026-09-12; canonical Chapter 4 prefill/cache definitions | Official workload mentions annual reports and software manuals. Project-document questions are our illustrative adaptation, not official prompt text or an executed benchmark. Prefix hits skip matching-prefix prefill work; new-question prefill and answer decode remain. No speedup figure or default-setting claim. |
| DeepSeek global source / sharing / two roles; Figure 5 | `learning_artifacts/day-06-modern-architecture/deepseek-v41-causal-encoder-decoder.md`; pinned report §§2.2–2.3; reference model/config/kernel | Report and source-code evidence; original simplified diagrams, not model activations |
| Final 128-token decoder replay and approximate states | Pinned report §3.2.2; course topic above | Author-reported serving strategy; quality result not reproduced; not benchmarked by readable reference |

The source PDF used during the preceding verified architecture work has SHA256
`ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`.
The companion player links to its retained Figure 3 license and attribution.
No external paper figure is copied into the article's PNG upload package.

## Intentional omissions

Detailed PagedAttention allocation, quantization error, offloading, eviction,
beam/speculative state management, and DeepSeek performance bar charts belong
in later articles. Do not infer total-memory savings or wall-clock speedups from
the toy projection counts, payload arithmetic, or the small animations.
