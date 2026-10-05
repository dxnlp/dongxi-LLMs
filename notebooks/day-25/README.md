# Day 25 — Policy identity and the system budget

Material status: ready for study; source notebooks are CPU companions, with adjacent
reference solutions and explanatory plots. Notebook execution is recorded separately
from learner mastery. No GPU/model download/server is required.

Chapter: [14](../../book/chapters/14-when-optimization-goes-wrong.md).
Lab route: [guide](../../book/labs/14-when-optimization-goes-wrong.md).
Solutions: [worked answers](../../book/solutions/14-when-optimization-goes-wrong.md).

1. [rollout versions and budget](01_rollout_versions_and_budget.ipynb)
2. [ragged KV cache and exact recovery](02_ragged_kv_cache_and_exact_recovery.ipynb)
   extends [Chapter 5](../../book/chapters/05-building-a-modern-decoder.md)
   with real compact caches, row-relative positions, EOS/PAD/cap controls and
   actual tiny DPO/RLVR completed or pending-rollout recovery. Its conditional
   integer-token policy is not pretrained language or an optimized server.
   Exercise6 also compares actual private sampler replay with a deliberately
   broken live-generator check; its adjacent reference/plot explains why a
   checkpoint check can consume work without changing the next training batch.
   Exercise7 uses actual separate shared-reader I/O accounting: two loads return
   identical tensors, a rejected third load stays spent, and reopening the older
   receipt cannot make the fourth load free. Its plot reports actual operations.
   Exercise8 adds an actual native local-random Qwen comparison: both snapshots
   hold two applied updates, but only the pending one already owns its next
   sampled group. Read actual retained IDs, first-step collection/RNG changes,
   replay equality and rejected-read spending. Its one-boundary microscope is
   separate from the full lifecycle's alternating pending/completed saves.

Learning outcome: Reject unintended stale rollouts and changed token/verifier identities; identify the dominant projected update cost.

Acceptance: Write a concrete engine-equivalence acceptance contract and a monitoring incident with observation, hypotheses and discriminating intervention.

The second session supplies a bounded correctness experiment and adjacent
worked answers. Inspect numerical batch/single agreement separately from
bitwise saved-cache continuation. The [specification](../../experiments/specs/2026-10-04-batched-cache-recovery.md)
freezes seeds and tolerances; the [report](../../experiments/reports/2026-10-04-batched-cache-recovery.md)
preserves natural cap failures and deliberately broken recovery branches.
Its exact tiny-session resume does not establish pretrained/CUDA/BF16 Spark recovery.

The [private-RNG lesson report](../../experiments/reports/2026-10-05-private-rng-verification-lesson.md)
records a separate fresh execution of the earlier expanded second notebook.
The [I/O source record](../../experiments/reports/2026-10-05-snapshot-io-readiness.md)
tracks the later shared-reader admission lesson, separately from runner semantics
and still-unfinished whole-job/physical resource gates.
The [native RLVR reader lesson](../../experiments/reports/2026-10-05-rlvr-reader-lesson.md)
and [current full source panel](../../experiments/reports/2026-10-05-rlvr-io-readiness.md)
record the next expansion:10source/reference cells8images, originalseven previews
unchanged, exact native phase recovery and permanent rejected-read work. Recovery
does not improve the tiny recipe's retained0/2 held-out accuracy.
Its [accounting follow-up](../../experiments/reports/2026-10-05-private-rng-verification-lesson-accounting.md)
includes the independent comparison generator:36 actual scalar draws in the
whole microscope, not merely30 in the five initially returned buckets.

Machine: Mac or Spark CPU. Model-scale execution requires its own frozen config,
resource guard and measured report. Animation production remains Mac-only and
requires its own instruction.
