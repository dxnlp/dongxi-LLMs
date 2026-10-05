# Pinned Base acquisition and tokenizer sizing

The learner approved downloading the exact public Qwen3-0.6B-Base snapshot
and measuring its tokenizer on CPU. This resolves availability and encoded
work requirements only. It does not authorize weight deserialization, model
inference, training, installations, services, Git writes or another checkpoint.

## Acquisition contract

Acquire `Qwen/Qwen3-0.6B-Base` at
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, using the existing platform
environment and explicit ordinary Hub cache. Disable implicit authentication;
the repository is public. Permit only the ten files in the previously retained
[upstream manifest](../reports/2026-10-05-checkpoint-inspection/upstream-manifest.json).
Expected payload is 1,203,641,805 bytes, including 1,192,135,096 weight bytes.
Verify every file's size, SHA-256 and upstream Git blob or LFS digest.
Hash weights as streamed file bytes, never tensor objects. Retain an exclusive
receipt, source bindings, elapsed time and sampled host memory/disk availability.
Payload bytes are not measured network traffic or peak memory.

## CPU measurement contract

Use only `AutoTokenizer`, local files, no remote code, hidden CUDA and
`USE_TORCH=0`, `USE_TF=0`. Follow-up after the retained first failure: this
installed Transformers version ignores the Torch flag. The successful retry
adds a process-local availability-probe exclusion and an independent import
audit hook; it changes no installed dependency. Verify actual tokenizer/config
bytes against the acquisition receipt before and after measurement.
Reuse the exact native `encode_record` and
`generation_upper` function definitions through audited AST extraction, avoiding
the runner's top-level Torch import. Bind their complete source file hash.
Use the original generator's 240 training and 60 development records and the
course template. Reproduce all three raw split hashes; do not tokenize or
evaluate the publication test. Reject overlength or incompatible prefixes.

For the unchanged 20-update recipe, select the first 80 records of the
seed-1212 shuffled training order. Record per-example IDs, token sequences,
labels, prompt lengths and shifted valid assistant targets. Sum training targets
and two full development panels. Size the two first-eight development generation
panels with 64 new tokens per attempt and verify prompt plus 64 fits 256.
Reserve conservative uncached generation geometry as an upper bound, not actual
cached positions. Preserve interface fingerprints and distinct EOS/message-end
IDs. Do not claim GPU memory fit, snapshot envelopes, recovery tensor sizes or
actual training outcomes from these measurements.

## Verification and handoff

An independent review must reconcile the schedule, masks and totals against the
native source. Save acquisition and sizing separately; preserve failed attempts.
Keep the learner on Day 9, all original eighteen package criteria/statuses and
all 45 unexecuted campaign outcomes unchanged. Update the operational next step
to the remaining snapshot/I/O declarations and separate native-profile approval.
