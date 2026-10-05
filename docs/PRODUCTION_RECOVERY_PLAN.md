# Runner-owned recovery before a Spark pilot

The2026-10-05 source review found safe work still available inside DXI-01/03.
The shared trusted-local format and actual SFT/DPO completed-update paths now
pass [independent CPU acceptance](../experiments/reports/2026-10-05-runner-recovery-root.md).
RLVR completed/pending paths also pass [root acceptance](../experiments/reports/2026-10-05-rlvr-runner-recovery-root.md).
Complete the remaining source integrations and independently exercise their
bounded CPU paths before requesting pretrained smoke/pilot approval. This plan does not authorize
downloads, model-scale inference/training, services, shared-environment changes
or Git operations. The learner remains Day9.

## Current evidence and remaining gaps

| Runner | Actual source interface | Unfinished source requirement |
|---|---|---|
| Story training |Completed-update recovery; independent same-journal receipt; strict21-dimensional persistent model work; cap refusal/poisoning and exact four-arm CPU replay|Story byte-I/O/validation/output admission, artifact routing, supervisor integration and approved current-source Spark replay/profile|
| SFT |Restricted snapshots; full/LoRA CPU replay; cumulative16-dimensional model/semantic ledger plus separate I/O9 admission, bounded external receipt and measured pre-model inspection|Snapshot/log/export/scratch reservations, stage-cap mapping and approved pretrained/CUDA/BF16/containment evidence|
| DPO |Original policy/reference/Adam replay; cumulative19-dimensional model/semantic ledger plus separate I/O9 admission; actual snapshot/artifact/work receipts and fixture encoding/layout schedule|Live production encoding/receipt gate, broader log/export/scratch reservations and approved pretrained/CUDA/BF16/containment evidence|
| RLVR |Separate collection/application; completed/pending no-resampling replay; cumulative23-dimensional model/semantic work plus separate I/O9 admission/independent dual-prefix receipt before inspection/model allocation|Snapshot/log/export/scratch reservations, stage-cap mapping, invocation-level enforcement and approved pretrained/CUDA/BF16/containment evidence|

### Findings at the start of this source pass

The following deficiencies describe the reviewed pre-repair sources, not the
current accepted SFT/DPO paths. Initially SFT loaded unrestricted pickle before checking configuration equality.
Its fixed seeded order is reconstructed, so it is not correct to claim that an
unrecorded Python RNG necessarily changes that order. The unchecked cursor must
nevertheless agree with the completed update/draw count. Configuration equality
mixes scientific identity with paths/runtime/endpoints; a move or extension
needs a separately declared contract, not silent bypass.

DPO initially wrote optimizer state only after final evaluation/generation/export. A
crash earlier has no periodic recovery point. A tokenizer interface and an
identity journal do not bind that optimizer file to complete policy/reference
weights. Current DPO rejects actual encoding collisions, verifies supplied source
groups, retains raw IDs/stops/cost boundaries, and labels missing legacy source
groups unverified; unique IDs do not manufacture that provenance.

RLVR cannot create its original KL reference by copying the restored policy:
that would silently change the objective. Its pending pool must retain actual
prompt/response IDs, EOS-valid masks, detached old selected likelihoods, raw
texts, stops, rewards, advantages, policy version/digest and collection RNG.
The original reference, optimizer and all applicable counters belong to the
same snapshot. A terminal export lacks this replay protocol.

## Implementation sequence and acceptance

1. Build a shared versioned trusted-local data-only snapshot contract. Validate
   an independently expected file byte digest/size and stable contract before
   deserialization; use `weights_only=True`, explicit size/resource bounds and
   exclusive immutable snapshots with a fsynced atomic commit header. Bind actual
   policy/reference/Adam states, RNG, cursor, counters and parent invocation.
   The [dense byte fix](../experiments/reports/2026-10-05-recovery-tensor-bytes.md)
   supports BF16 identity without changing old hashes; its tiny2MB loader limit
   is not an appropriate production default or proof that a large save fits.
2. Extract device-neutral completed-update functions from the existing SFT/DPO
   loops. Preserve numerical objectives/masks and original references. Resume
   into a new evidence directory; include the full committed numerical history so a crash
   between update/checkpoint/journal cannot silently duplicate or lose a step.
   Export/evaluation failure must not erase the last durable training state.
3. Split RLVR collection from application while preserving `update_model`'s
   existing behavior. Commit completed and post-full-collection phases; consume
   retained pending actions before collecting again. Restore a durable pre-update
   state after an interrupted backward/partial optimizer operation, never bless
   a half-mutated state. Mid-generation continuation is not this guarantee:
   retain partial attempts/errors/costs and declare replay from the last durable
   boundary separately.
4. Independently test the actual runner functions with original tiny local
   tensors/models: uninterrupted versus completed-step and applicable pending
   restarts, fresh-process reload, exact same-environment CPU weights/moments/
   reference/RNG/cursor/actions/metrics, and a no-collection spy on pending resume.
   Reject tampered bytes, schema, source/lock/data/parent/token-map/template/stops,
   stale pool versions and inconsistent counters before loading/applying. Retain
   failed save/export/metric/observer paths and reject existing evidence outputs.
   **Verified for shared/SFT/DPO:** 39 format,18 SFT and21 DPO tests in the
   final root panel; unchanged original objectives and actual fresh-process
   replay. A [separate seven-test activation-checkpointing control](../experiments/reports/2026-10-05-dpo-activation-checkpointing.md)
   also passes CPU FP32 replay/mode rejection with the released DPO source.
   Its logical forward counters do not measure backward recomputation dispatches
   or model-scale memory savings. Final RLVR acceptance passes19 actual-loop
   tests plus15 identity/merge controls, with exact completed/pending fresh-process
   replay and retained failed-stage observations. The earlier CPU panel
   passes [669 tests and five fresh references](../experiments/reports/2026-10-05-actual-runner-readiness.md).
   The historical [production-control panel](../experiments/reports/2026-10-05-production-control-readiness.md)
   passes766 tests with152 unchanged executable hashes, zero math/routes issues
   and five fresh references; neither panel reruns all76 notebooks.
5. Integrate a constrained approved-stage supervisor with tested platform
   containment, host reserve, actual GPU-idle checks, external deadline and declared
   total disk/token/work ceilings. The [worker controls](../experiments/reports/2026-10-05-owned-workers-and-disk-guards.md)
   prove only cooperative sampled cleanup and hard per-file limits; they are not
   a cgroup, total quota or arbitrary-command launcher.
   **Verified source subset:** all45 nonexecuting stage recipes, independently
   pinned authored fixture receipts, actual SFT/DPO/RLVR cumulative work,
   actual DPO snapshot reservations and strict mocked preflight/drain interfaces
   now have bounded CPU controls. A DPO fixture mapper derives caps from actual
   encoded examples, accumulation and full evaluation/generation panels; it
   refuses production scope and does not authenticate a supplied supplier.
   SFT/DPO/RLVR now charge their declared runner-semantic checks. The
   [semantic-validation report](../experiments/reports/2026-10-05-semantic-validation-readiness.md)
   records historical semantic evidence. The separate shared I/O9 source now
   reserves before generic reader hashing/tree checks and save cloning/serialization;
   SFT/DPO/RLVR bind both retained journals before actual CLI inspection/model loading.
   [Current I/O evidence](../experiments/reports/2026-10-05-rlvr-io-readiness.md)
   keeps caller capture, metadata/journal processing, application, inventory hashes,
   deserializer internals and physical resources outside those units. Native
   RLVR preserves its historical saved23 prefix and retains later validation
   charges; four original completed/pending fresh processes replay exactly. The
   [supervisor plan](PRODUCTION_SUPERVISOR_PLAN.md#verified-source-progress-2026-10-05)
   identifies remaining live/other-stage mapping, validation, story and output integrations.
   Mocked readiness does not authorize or establish an actual platform backend.
6. Only then request a bounded actual Spark profile/smoke/recovery stage. Name
   exact local bytes, branch, budget, stop conditions, save overhead, numerical
   tolerance and evaluation panel. Acquisition and subsequent pilots each need
   their own authority. CPU bitwise equality does not establish CUDA/BF16 parity.

The accepted RLVR integration now uses the captured actual invocation for its
redundant command field (preserving `-m`) and actually rehashes source/input/lock/
parent bytes before closure. Retain these verified safeguards; never infer proof
from a descriptive journal label. Freeze current sources and evidence separately;
historical measurements retain their own identities.

## Completion boundary

The [story-cap report](../experiments/reports/2026-10-05-story-valid-target-budget.md),
worker/disk controls and dense tensor-byte hardening are verified source progress.
They do not complete DXI-01/03 or manufacture any of the45 external campaign rows.
Shared/SFT/DPO steps1–2, RLVR step3 and their applicable step4 CPU checks now pass.
The [shared inspection-boundary source](SNAPSHOT_INSPECTION_BUDGET_PLAN.md) now
has explicit v1/v2 compatibility, independent same-journal pre-save receipts,
whole-operation envelopes and actual SFT/DPO/RLVR CPU controls. Runner-semantic
panels still do not charge this separate I/O work. The
[story-work acceptance](../experiments/reports/2026-10-05-story-work-readiness.md)
verifies persistent model work in the original separate story payload layout;
it does not migrate that layout to shared I/O9. Next safe source work is
story byte-I/O/validation/output admission; extend fixture cap
adapters and require live DPO re-encoding, reserve actual SFT/RLVR snapshots
and all remaining logs/exports/scratch. Preserve existing failed-attempt spending
and original equations. The three runner work ledgers, DPO snapshot hook,
fixture mapper and mocked preflight are source-verified, not a complete production job.
Existing sampled guards do not satisfy a hard
aggregate quota. Actual private service/cgroup/quota exercises require separately
scoped authority; technical access alone is not permission. Continue the detailed
[supervisor integration plan](PRODUCTION_SUPERVISOR_PLAN.md), not an automatic heavy launch.
Independently reviewed actual runner-owned CPU
replay is a source gate; approved pretrained replay and capability/genealogy
comparisons remain separate empirical gates.
