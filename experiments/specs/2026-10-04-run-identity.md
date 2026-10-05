# Run Identity and Checkpoint Interface Verification

Specification recorded before the new CPU measurements,2026-10-04.
Work package: DXI-01. Baseline:`ac203ef`; code changes remain local.
No pretrained model, GPU job, remote download or external publishing is approved.

## Questions and hypotheses

Can a same-size tokenizer permutation pass shape checks while violating the
meaning of checkpoint rows? Predict yes; the new interface check must reject it
before model loading or training. Can identical token mappings hide different
encoding rules? Predict yes; capture the serialized encoding pipeline, not only
the vocabulary.

A recorded upstream revision is metadata. Actual local artifact hashes are
distinct evidence. The run record must retain partial identity if loading fails.
Stable objective/data/interface settings must not include volatile invocation
time; otherwise exact-resume checks could reject a legitimate restart.

## CPU acceptance

- Fingerprint vocabulary, serialized encoding rules, special IDs, chat template
  and declared stops. Backend padding/truncation runtime state is excluded from
  encoding identity and is documented separately.
- Reject same-size permuted IDs, different merge/normalization rules, changed
  special/stop/template semantics, incompatible declared tokenizer revision and
  tampered fingerprint fields.
- Legitimate local HF tokenizer save/load passes. An explicit legacy migration
  validates actual saved tokenizer bytes and records that old revision lineage
  was not proven; default downstream loading fails closed without this decision.
- Capture source/input/checkpoint hashes, Git/dirty identity, Python/packages,
  selected environment-lock hash, command, device/driver metadata and graph/
  objective/generation settings without reading credential environment values.
- Common journal atomically retains each completed identity stage and failure.
  A malformed/error path must not overwrite an existing unrelated run.
- Verify a small local HF checkpoint save/reload on CPU and an actual local PEFT
  adapter merge/reload with declared parent/interface identity. These tiny random
  models establish serialization/compatibility only, not Qwen training.
- Optional SFT/DPO/RLVR runners expose and share these contracts; CLI/file tests
  and source review do not count as Spark integration evidence.
- New notebook uses original categorical/tokenizer fixtures, a meaningful
  identity perturbation, adjacent solutions and an explanatory figure.

## Evidence boundaries

CPU tests use original fixtures and tiny randomly initialized models. No new
Qwen run is performed. Environment-lock hashing identifies bytes; it does not
by itself prove every installed package was resolved from that lock. Cooperative
resource checks are not hard deadlines or continuous memory measurements.
Actual Spark validation remains a separate approval-gated check.
