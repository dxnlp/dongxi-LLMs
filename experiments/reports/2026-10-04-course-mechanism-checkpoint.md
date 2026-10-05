# Mechanisms connected to measured failures

Four further audit packages now meet their declared bounded CPU acceptance:
text reward/process supervision, learned critics and a frozen text-reward policy
loop, DPO retention controls, and matched-rollout objective gradients. The goal
remains active:7of18 packages are complete,6 in progress and5 planned at this
checkpoint. Source production does not advance the learner from Day9.

The [machine record](2026-10-04-course-mechanism-checkpoint.json) binds current
package evidence, chapter integration and route/plan identities. The fifteen
chapters and twenty-eight days remain intact. Three new visual notebooks and
an expanded text-reward session bring the verified inventory to70; no fresh
all70-session execution is claimed.

## What the new evidence teaches

| Package | Actual bounded experiment | Important remaining failure |
|---|---|---|
|[DXI-07](2026-10-04-text-reward-character.md)|Three declared character-reward seeds; independent encoding gates, calibration-only scaling and complete frozen export|Removing indistinguishable held-out inputs leaves ordinary ranking at0.5 for every seed|
|[DXI-10](2026-10-04-critic-policy.md)|Twelve actor/reward/critic arms;15,360 training paths,480 actor/240 critic steps and5,400 evaluation records|Constant answers and nonterminal-cap style loops; higher displayed prefix score is not delivered terminal reward or task quality|
|[DXI-11](2026-10-04-dpo-retention.md)|Forty-eight fixed DPO/chosen-NLL/rehearsal fits under clean/noisy/length controls|Improving margins need not improve absolute chosen likelihood; weak held-outs and imperfect warm parity remain|
|[DXI-12](2026-10-04-grpo-objective-controls.md)|Nine objectives on unchanged three-seed pools; exact analytical score gradients and complete retry costs|Gradient scale/direction is not policy quality;200 attempted responses yield only120 selected responses|

Chapters10–13 and their adjacent worked solutions/labs contain the executable
mechanisms. Chapter14 connects them into a diagnosis: representation can make
a measurement impossible, learned reward can generalize poorly, a critic can
misestimate continuation, and an objective or filter can alter the update.
These explanations do not infer a unique cause from one declining metric.

## Actual verification

All computation uses the isolated Linux ARM64 CPU environment:
Python3.12.14, Torch2.14.1+cpu and CUDA unavailable. Shared Spark GPU environment
and kernel are unchanged. OMP/OpenBLAS/MKL are explicitly limited to one thread.

- Integrated unit suite:366 pass,9.002 seconds at the recorded source checkpoint.
- Source math:54 book Markdown files,1169 expressions, zero issues; not live GitHub rendering.
- Navigation:15 chapters,15 solutions,4 appendices,70 registered notebook routes, zero issues.
- Targeted fresh Day16/18/20/22 kernels:33 executed reference cells,23 image outputs, no skipped exercises.
- Current package verification maps:65 recorded paths checked, zero hash mismatches.
- Ledger evidence paths exist; scoped changes pass `git diff --check`.

Independent replay reproduces every non-timing result of the48 DPO fits and
their three warm-ups. Root's first direct-object assertion included generation
wall times and failed; the successful comparison excludes only `seconds` and
`wall_seconds`, not scores, paths, masks, weights or costs. The critic review
recomputes closed-sum GAE on all15,360 training paths with maximum error
8.88e-16. The matched-objective review repeats eighteen tests, all three pools
and a new fresh visual notebook. The character review reproduces all three
fit/calibration/test/trace campaigns and the encoding audit exactly.

## Preserve rather than polish away

Original word-reward ledgers/weights/plots remain unchanged. The initial
character notebook's cross-version equality failure and the initial DPO
collection's shared-test failure remain beside their separately verified
corrections. Every scheduled seed, negative held-out result, truncated path and
rejected group stays in its denominator. Display-layout fixes and pre-fit
interface hardening do not silently replace earlier measured source identities.

## Next work and authority

Answer selection, audited teacher attempts, batched cache/recovery and response
distillation form the next CPU/source phase. Refinement, on-policy distillation
and integrated live visuals follow their prerequisites. Optional pretrained
training/evaluation, an actual controlled Spark campaign and Mac/hosted checks
remain open; no small fixture substitutes for them.

No model download, external API, GPU campaign, service, shared-environment
change, publication, commit/push or animation production occurred in this
mechanism phase. Mathematical animation proposals are recorded for separately
approved Mac production. The pages writing workflow shaped the chapter
integration into a continuous explanatory argument in the existing book;
no external Page was created.
