# Actual DPO snapshot paths under one persistent artifact ledger

This protocol is declared before measuring this new integration. Earlier shared
snapshot and DPO recovery/work-budget reports remain immutable historical runs.
No acquisition, GPU job, installation, service, platform write or Git action is
authorized here.

Use the existing original random CPU FP32 Qwen fixture: seed 1818, vocabulary 16,
width 16, one block, two query heads/one KV head, context 32; six updates with
accumulation two, beta 0.2, AdamW learning rate 0.008, weight decay 0.01, clipping
one, actual activation checkpointing and cache disabled. Keep the original
chosen/rejected masks, fixed detached reference, sampler, optimizer and objective.
Do not retune the fixture to make a safety control pass.

For the new snapshot path declare a 1 MiB payload envelope, an 8 MiB aggregate
pathname envelope, 64 entries and a 1 MiB journal envelope. Reserve payload,
commit marker and publication staging together before serialization/file creation.
The journal's full capacity is charged up front. This is cooperative coverage of
these direct snapshot files only, not all metrics, identity/status files, model
exports, scratch, subprocess writes or a physical filesystem quota.

Bind immutable capacities to the recovery contract. Each state carries a trusted
artifact-journal prefix captured before its own save. Validate that prefix on the
same open physical ledger before applying any model/optimizer/RNG state, retaining
later successful snapshots and failed partial files. A new invocation/output must
not reset capacity. Use new snapshot names on resumed invocations in the same root;
never overwrite or delete the retained parent. Independently pinned snapshot
SHA256, exact bytes and contract remain required; a ledger receipt is not their
replacement. Resume requires a separately retained artifact receipt and same root.

Acceptance controls: unchanged actual DPO numerical state against the existing
unbudgeted helper; exact same-process and fresh-process update-three continuation;
later 256-byte reserved file with eight actual partial bytes remains charged;
whole-bundle byte/entry refusal before serialization and before the first update;
serialization/publication/observer failures retain prior commits and charges;
changed receipt/capacity/root rejected before state application; active-ledger old
prefix validation does not rewind later work. Physical-backend-required mode must
continue refusing. Existing shared snapshot, DPO recovery and DPO work tests must
remain passing. Collect a new exclusive raw directory with actual exit/output,
source hashes before/after and fresh-child result; preserve all diagnostic failures.

Known limit: cloning the data-only state occurs before the serializer reservation,
so these controls bound artifact creation, not snapshot cloning memory or duration.
The production CLI will require explicit snapshot artifact capacities. No claim of
complete production containment or pretrained/GPU success follows from this proof.
