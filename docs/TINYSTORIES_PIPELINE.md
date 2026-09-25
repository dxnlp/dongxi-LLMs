# DongxiGPT Stories: preparation, smoke training and recovery

This is the executable companion to Chapter 6, not a GPT-2 reproduction. It
reuses GPT-2's tokenizer but trains our 66.64M modern decoder from random weights.
The transparent notebook decoder is unchanged. The separate training path uses
causal PyTorch SDPA without decode caches, FP32 parameters and AdamW state,
BF16 CUDA autocast, FP32 RoPE arithmetic and activation checkpointing.

## Runtime and scope

Run from the course repository on Spark. The verified interpreter is
`/home/dongxi/dgx-spark-dongxi/.venv/bin/python`; use the platform repository's
existing environment/lock. Required imports include PyTorch, NumPy, tokenizers,
huggingface_hub and requests. No installation or environment change is needed
on the verified host. Do not infer Mac readiness from these results.

Commands refuse to overwrite existing output directories/checkpoints. Choose
a fresh run name for reproduction. Data/checkpoints stay in ignored directories;
tracked reports retain manifests, source hashes, metrics and representative text.
Do not publish downloaded corpus text without reviewing its data terms.

## 1. Prepare the bounded fixture

```bash
PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/train_stories.py prepare \
  --output data/cache/day09-smoke-new --train-limit 1024 --valid-limit 128 --length 1024
```

The exact HF repository revisions are constants in `stories_data.py`, also
recorded in each manifest. Only tokenizer.json is downloaded for the tokenizer;
it embeds the vocab/merges and is hashed. No pretrained neural weights are used.
Public text streams stop at complete document boundaries. The fixture is the
first N documents, not a random or representative sample. Newlines are normalized
to LF by the line-oriented reader; the framing newline before each delimiter is
removed, with remaining text/whitespace preserved. Document-framed normalized
text digests are not presented as hashes of the complete upstream raw files.

The manifest records length quantiles, target counts, within-split duplicate
counts, and binary/window-file hashes. Normalized exact train/validation overlap
aborts preparation. Near duplicates and memorization are not checked. If the
full source contains exact overlaps, define and record a filtering policy rather
than bypassing the check. Incomplete preparation has no final manifest and must
not be used for training.

Each story is `[EOS-as-BOS, story tokens, EOS]`. Targets are shifted exactly once.
Only padded target positions are -100; real EOS contributes loss. Long stories
are split into target windows without dropping tokens, with preceding-token
overlap as input and context/positions reset per window. This preserves targets
but loses earlier context at boundaries; it is an explicit baseline policy, not
continuous whole-document training. No unrelated documents share attention.

Full preparation is supported by `--train-limit 0 --valid-limit 0`, but was **not
run in this verification**. It downloads/tokenizes the full streams and requires
separate resource planning. Do not begin a multi-hour run on the smoke prefix.

## 2. Verify mechanisms on CPU

```bash
PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m unittest discover -s tests -v
```

Tests cover SDPA/reference logits and gradients, causal exclusion, tied weights,
checkpointed gradients, shifted windows/padding/EOS, overlap/corruption rejection,
unequal-length token-weighted accumulation, validation grouping and exact recovery.

## 3. Bounded Spark smoke

First inspect `nvidia-smi` and `free -h`: do not run alongside a large model
server. The platform cgroup scope and our 0.2-second host monitor are separate
guardrails, not additive memory measurements.

```bash
PYTHONPATH=src MEMORY_MAX=80G OMP_NUM_THREADS=4 \
  /home/dongxi/dgx-spark-dongxi/scripts/run_memory_safe.sh timeout 900 \
  /home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/train_stories.py train \
  --data data/cache/day09-smoke-new --output outputs/day09-smoke-new \
  --total 3 --warmup 1 --valid-windows 2 --sample-tokens 16 --max-seconds 600
```

The defaults are smoke settings, not a tuned learning recipe: microbatch1,
accumulation1, AdamW betas(.9,.95), eps1e-8, matrix-only decay.1, clip1,
peak lr3e-4, floor3e-5, fixed three-update warmup/cosine horizon. CUDA uses BF16
autocast; CPU uses FP32. The monitor interrupts below25GiB host MemAvailable or
at the internal deadline. External timeout protects against stalled Python.

Artifacts include `run.json`, update metrics, initial/intermediate/final
validation and prompt completions, per-layer activation RMS/maxima, atomic
checkpoint files and sampled host/CUDA/cgroup memory summaries. Sampling uses
fixed prompts, greedy and temperature.8 modes, no top-k/p, and local seed909.
Generation recomputes the prefix; this simple correctness path is not a cached
serving benchmark. Validation uses the first `--valid-windows` windows and labels
whether the complete prepared split was evaluated. Three updates and two windows
do not establish language competence or representative held-out performance.

## 4. Recovery without changing the schedule

```bash
PYTHONPATH=src MEMORY_MAX=80G OMP_NUM_THREADS=4 \
  /home/dongxi/dgx-spark-dongxi/scripts/run_memory_safe.sh timeout 900 \
  /home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/train_stories.py train \
  --data data/cache/day09-smoke-new --output outputs/day09-resume-new \
  --total 3 --resume outputs/day09-smoke-new/update-000002.pt \
  --valid-windows 2 --sample-tokens 16 --max-seconds 600

/home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/verify_stories_checkpoints.py \
  outputs/day09-smoke-new/update-000003.pt outputs/day09-resume-new/update-000003.pt \
  --output outputs/day09-recovery-new.json
```

Restore includes optimizer, shuffle generator/order/cursor, CPU/CUDA RNG,
completed updates and valid-target presentations. Changes to data hashes,
architecture, recipe, training implementation, Torch version or device contract
are rejected. This strict policy favors reproducibility; cross-device migration
or a changed schedule needs an explicit migration design. Bitwise replay is a
tested local observation, not guaranteed across hardware/framework versions.

## Before the learning run

Execution update: the learner has now authorized learning run01. Its full-data
preparation uses `scripts/prepare_stories_full.py`: complete SHA256-verified
raw files, validated EOF handling, within-split deduplication and validation-first
exclusion from training. The guarded launcher and frozen four-hour recipe are
recorded in `experiments/specs/2026-09-13-tinystories-learning-01.md` and
`experiments/reports/2026-09-13-tinystories-learning-launch.md`. For this run,
partial validation is a fixed seeded selection, no longer the first N windows.
Graceful time-budget stops retain the latest completed checkpoint and
`completion.json`; schedule completion must not be inferred from process exit.

Use the [read-only training observatory](TRAINING_MONITOR.md) to inspect existing
smoke outputs or follow future logs. The viewer does not start training, change
the recipe or replace the trainer's safety monitor.

Prepare/audit the intended corpus, select representative development and final
evaluation splits, freeze longer story panels, profile multiple microbatches,
and settle the effective batch, token horizon, schedule and storage budget.
Smoke padding utilization is low; length-aware batching or document-safe packing
is a later efficiency improvement requiring its own correctness checks. The
current path intentionally favors simple, correct document boundaries.
Make a new bounded learning specification; do not silently extend the finished
three-update cosine schedule or extrapolate its timing to a four-hour claim.
