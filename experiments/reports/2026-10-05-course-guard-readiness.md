# Course source-readiness follow-up

The next source gates are clearer and three bounded improvements are implemented:
whole-update story target caps, owned-worker/disk guard controls and BF16-safe
dense snapshot bytes. They are integrated into Chapters6/14, worked solutions,
the standard-library evidence lab, the Day9 artifact and portable animation
proposals. The writing workflow placed these mechanisms inside the existing book
argument instead of creating a parallel chapter series. The course remains
15chapters/28days; the learner remainsDay9 and the goal remains active,13of18
packages complete with five partial packages.

## Actual verification

The [integrated CPU panel](2026-10-05-guard-readiness-cpu/cpu-verification.json)
records four actual commands with exit0:564tests, source math, navigation and
five fresh notebook references. Tests take29.287seconds inside unittest and
30.990968seconds at the subprocess boundary. All131 executable source hashes in
that panel remain unchanged. The source check covers54book Markdown files and
1251expressions with no issues; navigation retains15chapters/15solutions,
four appendices and76registered notebooks with no issues.

A separate fresh Day9/Day25 regression checks four disjoint notebooks,
17referencecells/eightfigures. Combined with the five-notebook panel, nine
references execute52of56code cells and produce21images. Four tagged unfinished
Day3 learner cells are preserved/skipped. This is not an all76-notebook rerun;
the [earlier all76 check](2026-10-04-course-upgrade-checkpoint.md) remains historical.
All current selected notebook hashes and the isolated CPU kernel prefix match.
Four counter/cache figures were independently inspected. The initial wrong
selector failed before any kernel; its diagnostic is retained in the final
closure record rather than counted as a notebook failure.

The [closure JSON](2026-10-05-course-guard-readiness.json) embeds both notebook
manifests, their identities, current evidence/hash checks and actual execution
of all four Python blocks in the updated Chapter6 evidence lab. Its source-only
campaign check retains45null/pending external outcomes and no actual genealogy.
No model-scale or Mac/hosted/learner result follows from these checks.

## Source improvements and limits

- [Story cap](2026-10-05-story-valid-target-budget.md):19new/12existing tests,
  four cap boundaries, zero refused forward calls, full stream rollback,
  matching-cap cumulative replay and unchanged default CPU equations.
- [Worker/disk controls](2026-10-05-owned-workers-and-disk-guards.md):eight
  real/injected paths and25focused regressions; successful leader exit can fail
  shutdown, independent-session workers can require retained handles, and a hard
  per-file cap differs from a sampled aggregate/free-space threshold. First
  diagnostic, failed collection and intermediate source snapshots remain intact.
- [Dense tensor bytes](2026-10-05-recovery-tensor-bytes.md):30tests,77old-layout
  parity cases,13archived identities and eight exact CPU continuations. BF16 bit
  serialization is verified, not BF16 neural training or larger checkpoint fit.

Source review confirmed actual SFT snapshot invariants/restore parity, periodic
DPO resume and RLVR pending-pool replay still require implementation. The
[ordered recovery plan](../../docs/PRODUCTION_RECOVERY_PLAN.md) is the next safe
source action. Actual production containment/quota, pretrained smoke/recovery,
controlled story ratings/genealogy, supported Mac and hosted CI retain their
own authority/evidence gates. No acquisition, new GPU job, shared-environment
change, service, publication, animation rendering or Git write occurred.
