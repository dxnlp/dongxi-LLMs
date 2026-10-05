# A native pool survives recovery without becoming a free read

The expanded Day25 notebook executes the original local-random RLVR recipe,
not a pretrained model. Both completed and pending boundaries at two applied
updates recover the identical original four-update policy, frozen reference,
Adam state, numerical history and global/rollout RNG for seeds2323/2324. The
first resumed record matches the uninterrupted native record byte-exactly. A
completed boundary collects once before that application; a pending boundary
uses its exact retained IDs with zero new collections and no training sampling
generator advance.

The [protocol](../specs/2026-10-05-rlvr-reader-lesson.md) was declared before
implementation/measurement. This controlled microscope manually publishes one
boundary per arm. It actually performs one save, one inspect and two loads,
of which one deliberately rejects its semantic callback. Reopening both same
physical journals from the saved prefixes keeps that rejected read and later
native save-validation work. These measured counts are not the full lifecycle's
$1+2U$ fresh publication schedule or its phase-specific resumed schedules.

## Actual retained evidence

The [accepted exclusive candidate](2026-10-05-rlvr-reader-lesson/run-02/verification.json)
records the actual command and successful unittest footer:6 tests in4.455s,
exit0. Two seeds' raw returned histories and work counts remain in
[phase-examples.json](2026-10-05-rlvr-reader-lesson/run-02/phase-examples.json).
Six controls cover exact trajectory, retained actions/first collection,
independent later spending, actual rejected-load work, caller RNG preservation
and undeclared seed refusal. They overlap the full acceptance suite and must
not be added to that suite's total as independent new coverage.

The same invocation freshly executes the existing Day25 notebook through the
already prepared isolated CPU kernel:10 source/reference code cells,8 actual
PNG outputs and one additional execution-only identity preamble. All earlier
seven saved previews remain unchanged. Only the new eighth figure is generated;
its SHA256 is4191be9c206790d29ca2df2591a27a8443182713df4966b3747bc28125c4cfe2.
Sixteen before/after source/input identities remain equal. The sampled available
memory minimum before children is117.55965423583984GiB with a25GiB reserve;
that observation is not a physical memory quota or full continuous profiling.

The [first failed candidate](2026-10-05-rlvr-reader-lesson/run-01/verification.json)
remains intact. Its presentation code compared nested tensor-containing records
with Python equality and encountered an ambiguous boolean. The correction uses
the existing exact structured digest and native JSON presentation helper,
without changing the recipe, compared fields, tolerance or learning question.
Subsequent errors retain tracebacks. No failed candidate was renamed a pass.

## Book placement and limits

[Chapter13§13.7.1](../../book/chapters/13-group-relative-policy-optimization.md)
and [Chapter14/answer33](../../book/solutions/14-when-optimization-goes-wrong.md)
connect the retained observation with reader admission and publication algebra.
The [notebook](../../notebooks/day-25/02_ragged_kv_cache_and_exact_recovery.ipynb)
pairs a prediction, actual arrays/plot and adjacent explained answer. Existing
CAND-ANIM-019 captures the later Mac storyboard only; no media is commissioned.

The scope is bounded offline Linux ARM64 CPU source/material verification in
the existing temporary environment, not Mac, hosted CI, pretrained/CUDA/BF16,
language capability, cross-machine replay or actual Spark resource enforcement.
Caller capture/history copies, metadata/journal processing, other identity
reads, state application, deserializer internals and output/storage containment
remain separate boundaries. Restricted trusted-local loading is not a hostile
checkpoint sandbox. No package/day/chapter/learner status, external outcome,
genealogy, installation, service, Git commit/push or publication is advanced.
