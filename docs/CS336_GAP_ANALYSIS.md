# CS336 gap analysis — second pass

Date: 2026-10-18. Status: final planning analysis, discussed with the learner;
not an implementation goal until one is opened.
Local baseline: `7b340127` on `main`, after the completed
[CS336 upgrade](CS336_IMPROVEMENT_PLAN.md) of 2026-10-11.

The learner asked for a fresh comparison of [Stanford CS336, Spring 2026](https://cs336.stanford.edu/)
against the current fifteen chapters, to find what the book still lacks. This
pass reads all seventeen Spring 2026 lecture outlines (executable lecture
scripts and slide decks from `stanford-cs336/lectures`) and the five pinned
assignment handouts, then checks each topic against the chapter, appendix, lab
and companion sources with case-sensitive searches. Counts below are searches
of prose, not judgments of learner mastery. Nothing here changes the fifteen
chapters, the 28-day route, the learner's Day 9 position or D005.

## What the first pass already closed

The 2026-10-11 upgrade covered the five assignments at the mechanism level:
tokenizer contracts and bits per byte (A1), resource accounting, online
attention and checkpoint audits (A2), a frozen scaling fit with refusal (A3),
local corpus curation with exact/near duplicates and exposure (A4), and the
SFT/verifier/GRPO integration with finite-group audits (A5). Post-training
depth exceeds CS336's lectures 15–16 in derivations and controlled failures.
Lecture 1 (tokenization), lecture 3's mechanism half (pre-norm, RMSNorm,
SwiGLU, RoPE, GQA, QK norm) and lecture 2's accounting half are covered.

## Coverage by CS336 lecture

"Strong" means the mechanism, a derivation and an executable companion exist.
"Partial" means prose or accounting exists without the lecture's main
mechanism or evidence. "Absent" means no section treats the topic.

| # | CS336 lecture | Dongxi coverage | Where | Gap |
|---|---|---|---|---|
| 1 | Overview, tokenization | Strong | Ch2, cs336/05 | None material. |
| 2 | PyTorch, einops, FLOPs, memory, arithmetic intensity | Partial | Ch5.18, cs336/01, App. B | No einsum/einops notation; no model-FLOPs-utilization (MFU) measure against a device peak. |
| 3 | Architectures and hyperparameters | Partial | Ch5 Part B | Consensus hyperparameters (feed-forward ratio, aspect ratio, head count, vocabulary size), z-loss, logit soft-capping, dropout/weight-decay rationale, sliding-window and interleaved local/global attention: all absent. |
| 4 | Attention alternatives and mixture of experts | Absent | — | Linear attention and its recurrent form, gated delta nets, hybrid stacks, sparse attention, MoE routing, load balancing, expert parallelism, upcycling. Roadmap defers "comprehensive MoE" to v0.2. |
| 5 | GPUs and TPUs | Absent | Ch1.7 (unified memory only) | Streaming multiprocessors, HBM versus shared memory, low precision, fusion, recomputation, coalescing, tiling, wave quantization, the FlashAttention tiling derivation. |
| 6 | Kernels and Triton | Absent | one mention | Benchmarking/profiling discipline (`torch.profiler`, Nsight), compiled versus naive operations, a written kernel. Declared P2 in the first pass. |
| 7–8 | Parallelism | Absent | Ch6.6 (one data-parallel sentence) | Collective operations, DDP, ZeRO stages/FSDP, tensor, pipeline, sequence and expert parallelism, 3D-parallelism accounting. |
| 9, 11 | Scaling laws | Partial | Ch6 (one paragraph), cs336/03 | Chinchilla's three methods, data-repetition scaling, critical batch size, muP and scale-aware learning rates, WSD schedules, optimizer scaling; the `C ≈ 6ND` rule appears only in a solutions file. |
| 10 | Inference | Partial | Ch4.11, Ch14.7, Day 25 | Prefill versus decode arithmetic intensity, throughput/latency trade-off, MLA/CLA cache reduction, quantization, pruning, speculative sampling, continuous batching, paged attention. Speculative decoding has zero mentions. |
| 12 | Evaluation | Partial | Ch7 | Public benchmark families (exam, chat, agentic, pure reasoning, safety), realism/validity of benchmarks, large-scale train–test overlap. No public benchmark is run anywhere in the course. |
| 13 | Data I: sources, copyright | Partial | Ch6.2, Ch8.9 | Lineage of public corpora (Common Crawl, C4, Pile, Dolma, DCLM, FineWeb, Common Pile), licenses and lawsuits, model-based filtering. |
| 14 | Data II: filtering, dedup, mixing, synthetic | Partial | cs336/06, Ch8.11–8.14 | Language identification, PII and harmful-content classifiers, quality classifiers, MinHash+LSH at billion/trillion-token scale, mixture selection via scaling. Our curation runs only on an authored fixture, never on real web data. |
| 15 | SFT and RLHF | Strong | Ch8–11 | Dataset lineage (FLAN, Alpaca, OpenAssistant), style versus knowledge, safety SFT, midtraining, annotator economics and demographics, mode collapse under PPO. |
| 16 | RLVR | Strong | Ch12–14 | Published pipeline case studies (DeepSeek R1, Kimi K1.5, Qwen3 thinking-mode fusion), test-time scaling, agentic RL. MaxRL and GSPO are optional reading only. |
| 17 | Multimodality | Absent | — | CLIP/SigLIP, LLaVA, Qwen-VL, early fusion. Out of v0.1 scope. |

## Coverage by CS336 assignment

| Assignment element | Dongxi state | Gap |
|---|---|---|
| A1 trained ablations: remove RMSNorm, post-norm, NoPE, SwiGLU versus SiLU | Mechanism-level ablations in Ch5; "no trained comparison was launched" (BOOK.md) | No trained small-scale architecture ablation with learning curves. |
| A1 leaderboard: 45 B200 minutes on OpenWebText | Absent | No fixed-compute challenge with a frozen score and a results ledger. |
| A2 profiling: end-to-end benchmark, Nsight, memory profiler, mixed precision | Bounded GB10 phase timings in cs336/01 | No profiler trace reading, no memory timeline, no compiled-versus-eager comparison. |
| A2 FlashAttention-2 in Triton | Online-softmax CPU reference | No fused kernel; P2 by declaration. |
| A2 DDP, bucketed gradients, optimizer sharding, parallelism analysis | Absent | No executable distributed lesson; no communication/memory accounting exercise. |
| A3 IsoFLOP fit against a training API | Simulated surface plus tiny empirical matrix | No fit on independently trained multi-size runs; three-method comparison absent. |
| A4 Common Crawl WARC/WET, language ID, PII, toxicity, quality classifier, Gopher rules, MinHash+LSH | Authored HTML fixture with reason-coded filters and verified duplicate components | No real web sample, no trained classifier, no language or PII step, no LSH banding. |
| A5 prompting baselines on a public task, vLLM in the RL loop, learning-rate and prompt ablations, Dr. GRPO, RFT, MaxRL, difficulty reweighting, off-policy GSPO, own estimator | Native audit with zero usable signal; CPU symbolic chain; Dr. GRPO discussed; finite-group audit; importance ratios and clipping derived | No public-task baseline; no engine-backed rollout; MaxRL/GSPO not derived; no open "design your own estimator" assignment. |

## What to add, in priority order

Each item names the chapter it would extend and the smallest executable
companion that would make it evidence rather than prose. Priorities follow
the existing rules: post-training depth (D005) stays first, the route stays at
fifteen chapters and 28 days, Spark runs stay bounded, and nothing is labeled
measured before it runs.

### P1 — closes a real gap at low or moderate cost

1. **Scaling beyond one device (Ch6, new section; extension notebook).**
   Teach collectives, data parallel, ZeRO stages and tensor/pipeline
   parallelism, then execute DDP and optimizer-state sharding with
   `torch.distributed` on CPU (`gloo`, several processes, one machine) using
   the tiny story decoder. The decisive check is exact equivalence between
   multi-process DDP and the single-process accumulation of Ch6.6. Add an
   accounting exercise for memory and bytes communicated under each strategy.
   No multi-GPU hardware is needed for the mechanism; Spark results remain
   single-device.
2. **Inference mechanisms (Ch14 or Ch15, new section; extension notebook).**
   Prefill-versus-decode arithmetic intensity on the GB10, throughput versus
   latency, speculative sampling with an exact distribution-equivalence proof
   and a measured acceptance rate using two existing DongxiGPT checkpoints as
   draft and target, and weight-only int8 quantization with a measured
   held-out NLL change. Continuous batching and paged attention stay prose.
   This keeps serving out of v0.1 scope while covering lecture 10's mechanisms.
3. **Real web data through the curation pipeline (Ch6/Ch8; cs336/06 sibling).**
   Pass one small public Common Crawl WET/WARC sample through extraction,
   language identification, PII masking, a tiny trained quality classifier,
   Gopher-style rules and MinHash+LSH banding, with a retention ledger and the
   licensing boundary written beside it. Keep the authored fixture as the
   reference; label the web sample's results as a second observation.
4. **Scaling laws in practice (Ch6; cs336/03 extension).** Add the
   `C ≈ 6ND` derivation to the chapter, Chinchilla's three estimation methods
   on the existing simulated surface, data-repetition scaling connected to our
   repeated finite corpus, critical batch size and muP as sourced reading, and
   a WSD-schedule branch as a predeclared bounded Spark protocol that reuses the
   14,000-update story run. Label it unexecuted until it runs.
5. **Trained architecture ablations and a fixed-budget challenge (Ch5 Part C;
   Assignment A).** Run four bounded small-scale TinyStories ablations
   (no RMSNorm, post-norm, NoPE, SiLU MLP) at one budget with learning curves,
   then open a 30-minute GB10 challenge with frozen development NLL and a
   results ledger. This supplies the trained evidence Ch5.25 asks for and gives
   Assignment A a comparable end point.
6. **Architecture hyperparameters and stability (Ch5 Part B, new section).**
   Consensus ratios (feed-forward width, aspect ratio, heads, vocabulary),
   z-loss and logit soft-capping with the Ch3 softmax, dropout versus weight
   decay, and sliding-window/interleaved attention with its cache consequence.
   Use the pinned Qwen3 configuration already in Ch5.18 as the worked case.
7. **One public benchmark under the Ch7 contract (Ch7; Day 10 extension).**
   Freeze a small GSM8K or similar development subset, run the existing
   reasoning evaluation on Qwen3-0.6B, and write the contract, uncertainty and
   contamination audit around it. This mirrors A5's prompting baseline and
   gives the course one externally comparable number.

### P2 — valuable depth after P1

8. **Mixture of experts and attention alternatives (Ch5 frontier
   extension).** A CPU microscope for top-k routing, expert feed-forward
   layers, a load-balancing loss and router collapse, and a linear-attention
   recurrent-form microscope beside the existing recurrent-depth section.
   Keep it mechanism-level; comprehensive MoE stays in v0.2.
9. **How a GPU executes a decoder (new Appendix E).** Execution and memory
   model, low precision, fusion, recomputation, coalescing, tiling and wave
   quantization, with one measured GB10 matrix-size timing sweep. Connect the
   tiling story to the existing online-attention companion.
10. **Profiling discipline (Appendix D or Ch6.10).** Read one
    `torch.profiler` trace and one memory timeline from an existing bounded run;
    compare eager and `torch.compile` on one operation. A Triton kernel remains
    optional and platform-dependent.
11. **Post-training context (Ch8–10, Ch13).** Public SFT dataset lineage,
    style versus knowledge, safety SFT and midtraining; annotator economics;
    published pipeline case studies (R1, Kimi K1.5, Qwen3 stages); MaxRL and
    GSPO as derived exercises with a finite-group check; an open "design your
    own estimator" assignment at the end of Assignment E.
12. **Evaluation landscape (Ch7).** Families of public benchmarks, what each
    claims, and benchmark validity, written as a map rather than a leaderboard.
13. **Notation (Appendix B).** An einsum section with the shape conventions
    the chapters already use.

### P3 — record only

14. **Multimodality** and **production serving** remain v0.2 backlog items.

## Feasibility on the local platform

The learner asked whether these gaps can be addressed on the DGX Spark, a
single-GPU machine. A read-only check on 2026-10-18 in the native environment
(`/home/dongxi/dgx-spark-dongxi/.venv`) found Torch 2.13.0+cu130 with CUDA
available, Triton 3.7.1 importable, NCCL and gloo backends built, and Nsight
Systems and Nsight Compute installed under `/usr/local/cuda/bin`. The device
reports as NVIDIA GB10, compute capability 12.1, 48 streaming multiprocessors
and 121.7 GiB of unified memory. No job was launched for this check.

| Feasibility | Items | Notes |
|---|---|---|
| Fully addressable on one GB10 or on CPU | 2 inference, 4 scaling laws, 5 trained ablations and budget challenge, 6 hyperparameters, 7 public benchmark, 3 web data, 8 MoE/linear-attention microscopes, 9 GPU execution, 10 profiling, 11–13 context and notation | Single-device work by nature. The GB10's modest memory bandwidth relative to its compute makes decode-is-memory-bound and the roofline easier to observe than on a datacenter GPU. The device uses unified LPDDR memory, not HBM, so the hardware appendix must describe this device and cite the HBM/SRAM hierarchy as sourced reading. Triton on compute capability 12.1 is new; treat a first kernel as a feasibility check. The fixed-budget challenge uses a GB10 budget, not CS336's 45 B200 minutes. |
| Addressable as mechanism, not as measurement | 1 parallelism | DDP, ZeRO-style optimizer-state sharding and tensor-parallel slicing can execute on CPU with gloo across several processes on this machine, with exact equivalence checks against single-process accumulation. NCCL refuses two ranks on one GPU, so the GPU path does not help. Inter-device bandwidth, pipeline bubbles and 3D-parallel throughput cannot be measured here; they become accounting exercises and sourced reading, labeled unmeasured. |
| Not addressable here | multi-GPU speedups, expert parallelism for a real MoE, training at 7B parameters and above | Remain v0.2 backlog, as the roadmap already records. |

The only gap the Spark cannot close is the empirical half of parallelism.
Every other P1 and P2 item is single-device work, and inference, ablations and
scaling are arguably clearer on a bandwidth-limited single GPU because the
bottleneck is visible without distributed noise.

## Evidence boundaries

This analysis compares outlines and search results; it did not execute any
notebook, training run or benchmark. The platform check above imported
libraries and read device properties only. Lecture materials were read from the
public `stanford-cs336/lectures` repository and the pinned handouts listed in
the [first-pass plan](CS336_IMPROVEMENT_PLAN.md); no CS336 prose, code or
figures were copied. Any item above becomes course evidence only through its
own specification, bounded run and report.
