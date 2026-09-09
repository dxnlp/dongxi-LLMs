# Day 8 — Bounded pretraining recipe

Prepared 2026-09-09. This is a concrete educational CPU control plus explicit
gates for a later GPU candidate. It is not approval to launch a Day 9 campaign.
The pre-execution [material verification spec](2026-09-09-day8-material-verification.md)
governs the notebook smoke checks already requested by the material-building task.

## Question and hypothesis

Can the Chapter 5 modern decoder consume a documented text stream, optimize the
correct valid-token objective, evaluate a fixed holdout and resume the same
bounded update trajectory? Hypothesis: the complete state will replay the
uninterrupted CPU control exactly in the tested runtime; omitting optimizer or
stream state will not. Falling loss alone is not the success criterion.

## Fixed executable control

| Field | Contract |
|---|---|
| Implementation | `src/dongxi_llms/pretraining_lab.py`, reusing `decoder_lab.py`; exact hashes in report |
| Corpus | Original `TRAIN` (8 documents) and `VALID` (2) constants; SHA-256 over tuple representation; no external dataset |
| Split checks | Unique IDs and normalized exact cross-split text overlap; no near-duplicate claim |
| Tokenizer | UTF-8 bytes 0–255, EOS256, BOS257; immutable identity string in checkpoints |
| Windows | 16 positions, shift once, no cross-document context, reset positions/context at each window, right-pad labels with -100 |
| Decoder | Modern, width16, 2 blocks, query heads4, KV heads2, head dimension4, hidden32, vocab258, max length64; RMSNorm, RoPE, SwiGLU, QK norm, tied embedding/output |
| Precision/device | CPU float32 training, CPU float64 equality probes, one torch thread; no autocast/CUDA |
| Seed | 808 for model and independent shuffle generator |
| Batching | Microbatch1 × accumulation2 × ranks1; summed microbatch NLL divided by total valid targets of the update |
| Budget | 24 optimizer updates per branch; 768 processed positions ceiling; count actual valid labels and repeated presentations |
| Optimizer | AdamW, betas(.9,.999), epsilon1e-8, weight decay.01 on all unique trainable parameters, foreach=False |
| Rate | Warmup3 updates to .01; cosine decay to .001 at update24; exact pure function of completed updates |
| Stability | Finite loss check, finite accumulated gradient check, global clipping1 before step; fail/abort rather than silently skip |
| Validation | Fixed held-out windows, token-weighted full-split mean NLL; no update; mode restored afterward |
| Checkpoint | Trusted temporary local file at completed update12; weights, optimizer, stream/RNG, counters, model/data/tokenizer/schedule contract |
| Recovery | Compare through update24: full state vs uninterrupted; missing optimizer and missing cursor as separate controls |
| Execution bound | Fresh-kernel notebook cells timeout180s; no branch beyond24 updates; verifier is a finite script, no persistent server |
| Outputs | Executed temporary notebooks, source manifest, 10 PNG previews, tests, report; no retained model artifact needed |

## Acceptance and failure

All reference notebook assertions and regression tests must pass. Correct
accumulation must match full-batch gradients within maximum absolute error
1e-10 in float64; wrong weighting must differ by more than1e-4 for the fixture.
Manual AdamW must match the tested PyTorch variant within1e-12. Validation
grouping must agree within1e-12 in float64. Full recovery must have zero maximum
parameter difference and identical subsequent history under this local contract;
both omission controls must differ by more than1e-5.

Reject nonfinite updates, changed checkpoint contracts, split contamination,
failed kernels or violated resource boundaries. Inspect figures rather than
equating image output with readable explanation. A loss decrease is descriptive
evidence only, not proof of language capability. Track material readiness
separately from learner understanding.

## Day 9 Spark candidate — not yet executable or approved

Before a GPU candidate is launched, make a new specification that resolves:

1. Actual corpus source/revision, permitted use, filtering, deduplication,
   tokenizer revision and frozen validation IDs. The teaching fixture is not a
   replacement for this decision.
2. Model scale, context length, microbatch, accumulation, exact token exposure,
   wall-clock limit, optimizer/schedule and comparison invariant.
3. A verified CUDA BF16/FP32 policy, kernel path and finite-gradient smoke test.
4. Measured GPU allocated/reserved/peak memory and monitored host MemAvailable,
   preserving the platform 20–25 GiB reserve; no fit claim from parameter counts.
5. Timed throughput with explicit warmup, synchronization and logging boundaries.
6. Checkpoint location/cadence, storage budget, interruption/recovery drill and
   acceptable cross-device numerical differences if portability is tested.
7. A single named intervention, safety/quality criteria, and explicit approval.

Current state: no production corpus selected here, no GPU profile measured,
no large-run hyperparameter recommendation inferred, and no Day 9 run launched.
