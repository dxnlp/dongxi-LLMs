# Student prefixes and private teacher targets

Status: executed original bounded CPU extension. All three seeds and eight
factorial arms remain recorded, including poor held-out reversal and cap stops.
The [specification](../specs/2026-10-04-student-prefix-distillation.md) was saved
before fitting or collection. The teacher is an authored finite policy; the student
is an actual tiny neural autoregressive decoder. This is not a pretrained teacher,
self-teacher, full state-distribution gradient or full SDPO reproduction.

## Measured command and identity

Executed on 2026-10-04, exit 0:

~~~bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  -m dongxi_llms.student_prefix_lab \
  --fixture fixtures/student-prefix/items.json \
  --spec experiments/specs/2026-10-04-student-prefix-distillation.md \
  --output experiments/reports/2026-10-04-student-prefix-distillation
~~~

The immutable [raw results](2026-10-04-student-prefix-distillation/results.json)
have SHA256 `56ee9b50a319028716b395cf56d4130ced4377b3068c5418aa1124b414258f39`.
The [contract](2026-10-04-student-prefix-distillation/contract.json) freezes the
actual fixture/spec, token mapping, seeds, budgets, eight arms, teacher rule,
reduction and gradient boundary. The [input identity](2026-10-04-student-prefix-distillation/input-identity.json)
records actual interpreter argv, separate rewritten Python argv, source/test/input
hashes, Git/dirty state and packages. All source/input bytes remained unchanged
during the command.

Execution used the existing isolated Linux ARM64 CPU environment, Python 3.12.14,
Torch 2.14.1+cpu, NumPy 2.5.3 and Matplotlib 3.10.8. No environment lock was
supplied to this invocation; the actual package snapshot is retained. No model
download, external API, installation, GPU, Mac, notebook server or service ran.

The [raw response journal](2026-10-04-student-prefix-distillation/responses.jsonl)
contains 10,656 attempts: 72 authored finite categorical teacher draws and 10,584
actual neural student responses. Its SHA256 is
`153065954a2efceb4caabea4a4cdc838e9cf056c650f4e8e7a83a2dd001b38a6`.
The [update journal](2026-10-04-student-prefix-distillation/events.jsonl) contains
721 events: 720 actual conditional-KL updates and completion. Its SHA256 is
`0a9375b8a5b380f4e7a191e156767f8604da762ada2e8868d09153061afabc9b`.
Existing outputs are refused; ordinary failure preserves partial attempts/events.

## The teacher is a declared finite control

The original task reverses two letters. Nine source-disjoint pairs supply six
training groups (AA, AB, BA, BB, CA, AC) and three test groups (BC, CB, CC).
The student reads `BOS REVERSE x y` and should generate `y x EOS`. English text
is documentation, not parsed input. The eight-token vocabulary also includes PAD
and HINT; generation has full support, temperature 1, natural EOS and cap 3.
Invalid outputs and early endings remain in task denominators.

The authored teacher assigns 0.8 to one target token and 0.2/7 to every other
token. On a clean prefix, its ordinary target is the correct next reverse token
or EOS. On a wrong prefix, the no-hint teacher repeats the last letter, or the
question's first letter if none was generated. This recovery failure is a designed
rule, not an observed property of a real teacher. With an authored correct-answer
hint, the teacher instead targets the correct token for the current action position.
Thus privilege changes teacher targets on wrong states without changing student
inputs. No natural-language rationale or causal-faithfulness claim is made.

Hints are restricted to training teacher queries. Student inputs remain the question
plus their own history, including their mistakes. Evaluation invokes only the
student on the question; it makes zero teacher queries and never appends the hint.
Independent grading may read gold afterward. A sampled HINT token is an ordinary
invalid action, not access to a private answer.

## Independent prefix, divergence and context factors

Seeds 16021, 16022 and 16023 each initialize a 2,864-parameter TinyDecoder: one
modern block, width 16, four Q/KV heads, hidden width 32, context 8, untied head,
float64 CPU, no dropout. All eight arms start from identical numerical weights
within a seed and train for 30 AdamW updates, learning rate 0.015, no weight decay,
gradient clip 1. There is no held-out checkpoint, seed or recipe selection.

The factors are fixed offline versus fresh student-generated prefixes, forward
versus reverse KL, and no hint versus privileged teacher hint. The fixed offline
pool consists of four finite-teacher sampled trajectories per training source,
collected once per seed using the privileged teacher. All four offline arms reuse
those exact states, even when later teacher scoring omits the hint. Student-prefix
arms sample four current-student trajectories per source before every update.
Coordinates pair random streams across arms, but different weights can produce
different sampled outputs. Actual model states and arms remain in attempt IDs.

The loss is a global valid-state mean. The state that sampled EOS contributes;
post-EOS and padding do not. Caps supply only states actually visited, without an
invented ending. Short trajectories receive fewer total state observations. The
same 30 updates therefore do not match state exposure, target queries or collection
work. The source stores first/last state witnesses, all update denominators,
state keys, gradients and raw action identities.

Teacher probabilities are detached and the exact vocabulary mapping is enforced.
At each observed state, the update differentiates conditional KL with respect to
the student's probabilities. It does not differentiate the sampling process or
state-occupancy distribution and has no REINFORCE term. Student-collected states
and KL direction are separate interventions; neither implies the other.

## Actual observations and retained negatives

Across all fitted offline arms, 25,200 states receive targets, with 3,840 wrong-
prefix state uses (15.24%). Student-prefix arms visit 23,331 states, with 8,212
wrong-prefix uses (35.20%). The different state distribution is measured, not
assumed. It does not establish better recovery or stronger held-out task behavior.

All original students have zero whole-response success on the held-out attempts.
The following table retains every final arm. Training columns are correct
responses out of 48 attempts; held-out columns are out of 24. Columns list the
three declared seeds in order 16021, 16022, 16023, not confidence intervals.

| Prefix source / divergence / teacher context | Train correct by seed | Held-out correct by seed |
|---|---|---|
| Offline / forward / no hint | 11, 8, 20 | 0, 1, 0 |
| Offline / forward / hint | 14, 28, 20 | 5, 6, 7 |
| Offline / reverse / no hint | 16, 11, 15 | 0, 2, 2 |
| Offline / reverse / hint | 22, 26, 10 | 2, 1, 1 |
| Student prefix / forward / no hint | 0, 4, 2 | 0, 0, 0 |
| Student prefix / forward / hint | 15, 10, 21 | 1, 0, 3 |
| Student prefix / reverse / no hint | 0, 0, 0 | 0, 0, 0 |
| Student prefix / reverse / hint | 6, 6, 21 | 0, 0, 2 |

There is no general on-policy superiority result. The finite teacher was explicitly
constructed to have a wrong-prefix recovery failure without hints; that design
limits interpretation of the context comparison. Hint-conditioned targets and
falling KL do not guarantee unseen-pair reversal. Full response, final symbol,
format, natural EOS and cap outcomes are separately retained, rather than counting
a correct last symbol as a correct sequence.

The 10,656 collected attempts have 4,214 EOS stops, 6,442 cap stops and zero ordinary
generation exceptions. Zero actual error-group skips occurred. A separate sibling
availability diagnostic counts 1,775 student-prefix group uses without a correct
complete sibling. Offline arms count 120 such uses, repeatedly reusing the same
saved groups across updates. Those are diagnostic group uses, not independent
polls, actual skipped training or a self-teacher demonstration gate. Our authored
hints do not require a sampled correct sibling. Scripted tests separately verify
real error skips with all group costs preserved and no resampling.

## Exact tail mass is not exact token detail

The separate finite control has identical top-two probabilities 0.4 and 0.2
and identical tail totals 0.4. Tail allocation differs. Full forward KL is
0.211166908232 nats; full reverse KL is 0.339322921201 nats. At k=2, both bucketed
losses are exactly zero. Tail mass is conserved, but within-tail disagreement
is invisible to that loss. The notebook additionally checks logit gradients:
the bucketed forward gradient is zero while the full gradient is not.

For k=1,2,4,7,8, student-selected token probabilities are retained individually
and every other token is summed into one bucket. There is no dropped-tail
renormalization or arbitrary floor. Both full and bucketed distributions conserve
mass, and full KL equals bucket KL plus the appropriate tail-weighted conditional
KL. Forward detail is weighted by teacher tail mass; reverse detail by student
tail mass. Coarsening has no greater KL than the full distribution. k=8 recovers
the full loss and gradient with a zero tail. k=7 also preserves full detail for
this eight-token vector because the remaining bucket contains only one token.

The raw control's gradient difference is explicitly with respect to unconstrained
probability coordinates, not neural logit gradients. The notebook's separate
logit-gradient control respects the probability simplex. These are exact authored
finite checks, not empirical approximation error for a pretrained vocabulary or
measured large-model memory/compute savings.

The independent zero-support control uses student probabilities (0.5,0.5,0)
and teacher probabilities (1,0,0). Forward KL is log 2, or 0.693147180560 nats;
reverse KL is positive infinity. The JSON uses a descriptive infinity status,
not a nonstandard numeric value. This is not used for fitting, where the finite
teacher deliberately retains positive support for every token.

## Work and resource boundaries

| Work, all seeds/arms | Attempts / updates | Generated actions | Neural attempted prefix positions | Retained KL states / teacher target queries | Padded scoring positions |
|---|---|---|---|---|---|
| Fixed finite offline pool | 72 attempts | 210 | 0 | — | — |
| Four offline fitted arms | 360 updates | 0 newly collected | 0 | 25,200 | 151,200 |
| Four student-prefix fitted arms | 8,640 attempts, 360 updates | 23,331 | 114,850 | 23,331 | 139,986 |
| All original/final evaluations | 1,944 attempts | 5,420 | 26,801 | 0 | — |

The finite offline teacher incurs 210 categorical queries, not neural forward
tokens. Conditional scoring adds 48,531 finite teacher target queries, distinct
from collection. Neural generation totals 28,751 actions and 141,651 full-prefix
positions. Finite and neural actions together total 28,961. All wrong, invalid,
early-ended and capped attempts are charged; attempted/completed counts agree
only because no ordinary generation exception occurred in the main run.

Campaign-body wall time was 63.054718 s, including checkpoint hashing, source
checks within the body and fsynced journals. Offline arm bodies totaled 3.805720 s;
student-prefix arm bodies 48.732547 s. Inner finite collection timers totaled
0.002374 s, neural training collection 3.157871 s and evaluation 0.826444 s. Inner
generation timers exclude checkpoint hashing, payload serialization and journal
writes, so they must not be compared directly with overall body time. Overall
time excludes process startup/imports and final result serialization. This is not
optimized serving latency or a profiled Spark training estimate.

Linux lifetime peak RSS was 754,740 KiB. MemAvailable observations were
123,701,740 KiB before and 123,588,740 KiB after. Two observations do not establish
a monitored minimum, allocator peak or device memory. No speed advantage or
resource forecast is inferred from the smaller symbolic vocabulary.

## Verification and interpretation

Twenty-three focused CPU tests pass, covering finite mass/support, both exact KL
gradient formulas, teacher detachment, mapping mismatch rejection, top-k tail
decomposition/lower bound, full-k zero-tail gradients, independently authored
zero-support infinities, actual neural sampling, paired coordinates, hint-free
evaluation, prompt/source leakage, EOS/cap/padding, first-backward backbone/head
reach, no post-EOS loss, retained error-group work and old-output preservation.
An initial premeasurement test exposed a duplicate witness dictionary key; it
was fixed before the passing test panel and before this campaign. No failed
measurement was repaired or replaced.

The [Chapter 15 extension notebook](../../notebooks/day-26/05_student_prefix_and_teacher_context.ipynb)
contains adjacent predictions/reference solutions, two fresh full-budget seed
16021 arm replays, fixed-state context/direction gradients, seven scientific
figures and mass/detail perturbations. Its fresh kernel and inspected previews
are bound in the [verification record](2026-10-04-student-prefix-distillation-verification.json).
The earlier complete-response source/results and exact forward-KL/T² lesson
are preserved unchanged.

Conceptual references were inspected at RLHF Book revision
`eecc49e1e1daf1be670d7242eb090240b5bb0f04`: [offline/on-policy discussion](https://github.com/natolambert/rlhf-book/blob/eecc49e1e1daf1be670d7242eb090240b5bb0f04/book/chapters/12-synthetic-data.md),
[distillation pathway](https://github.com/natolambert/rlhf-book/blob/eecc49e1e1daf1be670d7242eb090240b5bb0f04/code/distillation/README.md)
and [loss implementation](https://github.com/natolambert/rlhf-book/blob/eecc49e1e1daf1be670d7242eb090240b5bb0f04/code/distillation/loss.py).
Their code, prose, diagrams and reported results were not imported. The present
finite teacher, authored hint and direct conditional update are an original
microscope, not an implementation of that full self-distillation training stack.
No external acquisition, API, GPU, pretrained result, causal rationale faithfulness,
model-scale transfer, optimized approximation savings or learner mastery is claimed.
