# TinyStories pipeline verification — before execution

User authorized building and verifying the pipeline on 2026-09-13, not the
multi-hour learning run. Preserve existing uncommitted course work.

## Bounds and identity

- Reuse the pinned tokenizer and 66,638,848-parameter architecture in the
  [Day 9 design](../../learning_artifacts/day-09-pretraining-run-and-diagnosis/tokenizer-and-model-baseline.md).
- Download tokenizer assets only, never GPT-2 neural-network weights.
- First data preparation: original pinned TinyStories text files, at most
  1,024 training and 128 validation documents. This deterministic prefix is a
  pipeline fixture, not a representative training/evaluation sample.
- Record exact normalized overlap checks, per-file hashes, document lengths,
  complete-document boundaries and valid-target counts. No near-duplicate claim.
- Keep documents separate. Split long documents into nonoverlapping target
  windows, carrying the preceding input token but resetting attention/positions
  at each window; no text is silently truncated. Right padding uses EOS ID50256
  but only padded target positions are ignored. True EOS remains supervised.
- CPU tests: tiny shape SDPA/reference forward and gradient agreement, causality,
  shifting, token-weighted accumulation, fixed validation and exact recovery.
- GPU: eager (not compiled) PyTorch SDPA, FP32 parameters/AdamW state, BF16
  autocast, activation checkpointing; no retained decode KV cache.
- Full architecture smoke: microbatch1, accumulation1, context1024, at most
  3 optimizer updates per invocation; recovery probe resumes update2 to3.
  A separate tiny-shape CUDA test may compare reference/BF16 execution.
- Smoke-only optimizer: AdamW lr3e-4, betas(.9,.95), eps1e-8, decay.1 on matrix
  parameters only, clip1, one warmup update and cosine decay to3e-5 at update3.
  These are verification settings, NOT a selected multi-hour training recipe.
- Cap each GPU command at 15 minutes externally and 10 minutes internally;
  use platform cgroup wrapper with MemoryMax80G, no swap charged to the job;
  sample host available memory every0.2s, abort below25GiB. No concurrent model
  server. Record CUDA allocated/reserved peaks and cgroup peak separately.
- Checkpoint and small data outputs under ignored `outputs/day09-*` and
  `data/cache/day09-*`; refuse to overwrite an existing output directory.

## Acceptance

All CPU tests pass, including tiny FP64 forward/gradient agreement (1e-9 absolute
and relative tolerance) and uninterrupted-versus-resumed identity. Full-shape
GPU updates have finite loss, finite nonzero gradients and changed parameters;
memory reserve holds. Resume restores model, optimizer, RNG, shuffle/cursor,
schedule, token counter and immutable run/data contract; reject mismatches.
Record any GPU numerical resume difference, do not assume bitwise identity.
Include initial/final validation and fixed prompt samples, but do not interpret
three updates as evidence of coherent storytelling. Throughput from three
updates is a smoke observation, not a stable long-run speed forecast.

## Still outside this verification

Full-corpus audit/tokenization, representative frozen evaluation, final batch
and schedule selection, sustained throughput profiling, multi-hour training,
controlled learning ablation, commit/push, and animation production.
