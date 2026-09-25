# TinyStories pipeline: Spark smoke and recovery

Date: 2026-09-13. Mode: bounded implementation verification, **not a learning run**.
User authorized building and verifying the pipeline. The
[pre-execution specification](../specs/2026-09-13-tinystories-pipeline-smoke.md)
defines its limits. [Commands and data semantics](../../docs/TINYSTORIES_PIPELINE.md).
Machine-readable evidence: [JSON manifest](2026-09-13-tinystories-pipeline-smoke.json).

## What was built

- Pinned tokenizer-only download and bounded TinyStories streaming preparation;
  hashed memory-mapped token files, explicit boundaries, shifted windows,
  length statistics and exact cross-split overlap rejection.
- Separate SDPA modern decoder using the course reference's parameter layout.
  No decode KV retention or change to existing notebook implementations.
- Valid-target-weighted gradient accumulation, FP32-CE/BF16 CUDA autocast,
  FP32 AdamW state, clipping, finite checks and activation checkpointing.
- Fixed validation and story samples at initial, checkpoint and final states;
  valid-position activation summaries, update timing and memory telemetry.
- Atomic checkpoint saves and strict source/data/recipe/environment recovery
  contracts, with runnable checkpoint comparison and evidence collection.

## Data actually prepared

Original TinyStories repository revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`; GPT-2 tokenizer revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` (no neural weights downloaded).

| Split | Prefix documents | Valid next-token targets | Median targets/story | Maximum |
|---|---:|---:|---:|---:|
| Training | 1,024 | 234,872 | 196 | 972 |
| Validation | 128 | 21,818 | 170 | 301 |

No normalized exact duplicates within these prefixes or overlap across them
were found. No long story exceeded the 1,024-target window in this subset;
long-document handling was tested synthetically. The full corpus has **not**
been audited or tokenized. Prefix selection is not representative sampling;
near duplicates, memorization and training rights remain separate checks.

## Verification observations

Final source version: `outputs/day09-smoke-v2` and `outputs/day09-resume-v2`.
Both GPU commands exited **0**, as did checkpoint comparison. Earlier v1 passed
before adding intermediate observations/activation summaries; v2 was rerun and
is the canonical result. No failed GPU run was suppressed.

- CPU suite: **82 tests passed**, including 11 focused pipeline tests. FP64
  SDPA/reference logits and gradients pass 1e-9 tolerances. Unequal valid-token
  accumulation, causality, padding/EOS, corruption, and recovery tests pass.
- Full configured model: **66,638,848 parameters**, context1,024,
  microbatch1/accumulation1, checkpointing on, no compilation, automatic PyTorch
  SDPA dispatch (no claim that one named fused kernel was forced).
- Three updates consumed **557 valid target presentations**, within 3,072
  processed positions. Losses were10.938012,10.606697,10.123709 on different
  training examples, so their decline alone is not a controlled comparison.
- The same **two validation windows / 383 targets** changed from
  **10.903464 to10.182151** mean NLL. This is a tiny smoke metric, not a full
  validation result or evidence of coherent stories.
- Gradient norms before clipping:14.378839,6.676436,5.325219; updates and observed
  activations finite. The probed first Q projection changed at every update.
- Restoring update2 and executing update3 produced **bitwise-identical model,
  optimizer, stream, counters and CPU/CUDA RNG state** to uninterrupted execution.
  The final validation result also matched. This was a restart from a saved
  update boundary, not a forced power-loss or cross-device recovery test.

| Memory view | Uninterrupted | Resumed |
|---|---:|---:|
| Minimum sampled host MemAvailable | 114.182 GiB | 114.411 GiB |
| CUDA peak allocated | 1,798,136,832 bytes | 1,798,385,664 bytes |
| CUDA peak reserved | 2,090,860,544 bytes | 2,011,168,768 bytes |
| Cgroup peak | 3,302,199,296 bytes | 2,588,585,984 bytes |

Memory domains are separate and must not be added. Host sampling interval0.2s
can miss shorter transients. MemoryMax80G and MemorySwapMax0 were applied by
the platform wrapper; the monitor's25GiB reserve was never breached. Each
GPU command had a900s external timeout and600s internal deadline.

Update-only synchronized times in the uninterrupted v2 run were0.201678,
0.099423 and0.076997 seconds. Their valid-target rates are recorded but **not a
sustained performance benchmark**: only three updates, startup effects, varying
real lengths, substantial padding, and validation/checkpoint work excluded.

## Samples and interpretation

All six fixed greedy/sampled completions per observation are preserved in the
JSON evidence, not selected for attractiveness. After three updates, the rabbit
prompt's greedy continuation is16 periods; its sampled continuation is unrelated
word fragments. The result establishes working training machinery, **not learned
story coherence**. Generations were capped at16 new tokens; a cap is not EOS.

The host was GB10, driver580.173.02, Linux/aarch64, Python3.12.14,
PyTorch2.13.0+cu130. The platform lock hash and actual installed package versions
are captured; no dependencies were installed or changed. Code was based on
`ca2cfe1900a70971dd2cb8873104229916f1f981` with preserved local edits. Source
hashes, rather than that base commit alone, identify the code actually run.

## Remaining before a learning campaign

Prepare/audit intended training data, choose representative fixed evaluation,
freeze generation and checkpoint cadence, and profile a longer bounded workload
to select batch geometry and schedule. Only18.1% of processed positions in this
smoke were valid targets; improve efficiency without violating document boundaries
before treating measured positions/s as learning throughput. Multi-hour training,
story competence, a controlled recipe ablation and Day9 completion remain open.
No commit/push, animation production or persistent notebook/server was started.
