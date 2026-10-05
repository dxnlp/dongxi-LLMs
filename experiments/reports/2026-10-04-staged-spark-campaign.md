# Spark campaign preparation and bounded CPU supervision

The staged campaign is prepared; no new model-scale stage has executed. The
default command emitted 45 pending rows and started zero children. The explicit
supervisor verification retained eight CPU controls, including failed exits and
refused launches. This establishes the bounded preparation instrument, not the
complete Spark evidence campaign or authorization to start it.

The [specification](../specs/2026-10-04-staged-spark-campaign.md) was saved before
tests and fixture collection. The [default preparation](2026-10-04-staged-campaign-preparation.json)
and [CPU verification](2026-10-04-campaign-supervisor-verification.json) record
actual invocations, environment, all 23 source/input hashes and unchanged-source
checks. The verification took 0.722 seconds. No model, tokenizer, corpus or API
was loaded; no GPU profile, inference, training, installation, service or Git
operation was performed.

## What the campaign now declares

The matrix keeps eight story stages, twenty-two assistant stages, fourteen
reasoning stages and one branch-specific capstone row. All 45 retain null actual
checkpoint identities, interfaces, approvals, resource measurements, exits,
evaluations and failures. Actual genealogy is empty. A prerequisite stage is not
a policy-weight parent: profiles and smokes never silently warm-start pilots.
Derived adapter merges and completed/pending replay have their own explicit gates.

The story comparison is fresh matched initialization and training order at
peak learning rate 3e-4 versus 1.5e-4, with proportionally scaled floor rates.
The fixed 14,000-update candidate has proposed ceilings of 50M valid training
targets and 229,376,000 physical padded positions, plus four hours per arm.
These are unapproved limits, not observed exposure, a memory-fit guarantee or
a runtime forecast. The current runner does not enforce the proposed explicit
valid-target cap; implementing and verifying that guard is a pre-pilot blocker.

The [existing Day 9 result](2026-09-14-tinystories-learning-result.md) remains an
actual historical anchor: 14,000 updates, 48,839,975 valid target presentations
and actual child exit 0. It does not become one arm of a newly controlled
comparison merely because its settings resemble the proposal. New paired arms
must share actual data/source/interface bytes, initialization and masks; any
departure must be a separately documented intervention.

The assistant branch begins with genuine Qwen3-0.6B-Base, compares full SFT with
rank-8 Q/V LoRA, then declares the full-SFT final checkpoint as the downstream
parent before evaluation. A failed full branch blocks DPO; it does not trigger
automatic favorable LoRA substitution. An explicit adapter merge is required
for full-policy downstream tools. The unchanged SFT parent, DPO and chosen-only
SFT controls are separate rows. Optional chosen-NLL/rehearsal extensions retain
their unresolved coefficients, additional supervision and new approval needs.

The reasoning branch separately identifies genuine base and post-trained
weights. Raw base, explicitly templated base and instruct thinking-off/on
are separate interface rows. The thinking toggle uses the same instruct weights;
it neither establishes a new model nor proves faithful reasoning. G4/G8 RLVR
pilots share sixteen updates but have different response-token ceilings,
4,096 versus 8,192. Those maxima are not actual valid tokens or equal-compute
comparisons. Later 1.7B and multi-seed confirmations remain optional and gated.

## Evaluation is specified before the new training

The original twelve-opening story panel freezes four attempts per opening:
greedy plus temperature-0.8/full-support samples with seeds 909/1909/2909.
It freezes a 256-token cap, context 1024 and checkpoint coordinates
0/400/4000/8000/14000. Missing checkpoints remain missing cells. Five independent
0/1/2 dimensions cover grammar, entity/object consistency, causal continuity,
repetition and ending. Two blinded raters, disagreement and source-opening
paired uncertainty are required; no ratings have been collected or invented.
Corpus contamination review and observed tokenizer binding are still pending.
Under the declared complete panel, two arms would request 480 completions and
at most 122,880 output tokens; this is a proposal, not consumed compute.

Assistant evaluation freezes disjoint value/source groups, the audited template,
one shift, answer/termination masks, task-specific metrics and initial/final
generation. Reasoning evaluation keeps all twenty original arithmetic/algebra/
multi-step items, while marking problem overlap with the proposed four RLVR
training pairs. Seen or development examples do not enter the held-out headline.
Task-family slices, invalid/unsupported/ambiguous answers, truncation, raw IDs,
likelihoods and all attempted costs stay visible. A correct answer alone does
not establish rationale faithfulness or broad reasoning transfer.

The four logical design hashes are recorded in JSON. They bind the proposed
panels, rubrics and settings; they are not hashes of an observed loaded model's
tokenizer or template interface. Publication items do not choose a favorable
learning rate, parent, checkpoint, coefficient or decoding setting. Preference
margins, story coherence, reasoning correctness and unrelated retention remain
separate measurements, not a universal checkpoint-ranking percentage.

## Actual supervisor fixture evidence

Only six fixed self-spawned standard-library CPU leaf children ran; two controls
were refused before spawn. All eight predefined outcomes passed their checks.
A passing control does not mean a failed child succeeded: its actual exit and
failure status remain in the raw records.

| Control | Actual child exit | Recorded outcome |
|---|---:|---|
| Normal completion | 0 | completed |
| Deliberate failure | 7 | failed |
| Self-signal SIGUSR1 | -10 | failed, actual signal retained |
| Ignore TERM past deadline | -9 | TERM then KILL, reaped |
| Injected preflight 24 GiB | no child | gate rejected |
| Injected running 24 GiB | -15 | owned child stopped |
| Injected conflicting service | no child | gate rejected |
| Injected observer failure | -15 | failure journal and owned-child cleanup |

The blocking child had a 0.2-second external cap and 0.1-second TERM grace.
It ignored TERM, received KILL and was reaped after 0.3251 seconds, including
grace/reaping overhead. This is actual outside-the-child supervision, not a
deadline checked only between that child's operations. It is not a realtime
scheduling guarantee. The observer-failure row's elapsed work is retained in
its journal rather than a separate per-child seconds field.

Seventeen actual Linux MemAvailable samples had minimum 118.0453 GiB. The
injected 24-GiB values are controls, not observed host shortages. Sampling was
configured at 0.02 seconds; inspection, scheduling and fsync can widen actual
intervals. Memory is sampled, not continuously bounded. The reserve cannot be
configured below 25 GiB. The read-only signature/large-RSS scan returns sanitized
PID/RSS/reason fields, never process arguments or environments. It is not proof
that every possible GPU process is idle, and no discovered process is signalled.

Fsynced preflight/sample/start/signal/failure/actual-exit journals, fixture stdout,
stderr and per-row results are preserved under
`experiments/reports/2026-10-04-campaign-supervisor-fixtures/`. The parent report
records each journal hash. Child environments contain only the fixed CPU module
path, locale and CPU settings; credentials are not forwarded by this instrument.

## Independent checks and current identities

Fourteen focused tests passed in 0.951 seconds. They independently check
non-executing preparation, distinct parent branches, acyclic prerequisites,
valid versus padded budgets, frozen rubric identity, seen-problem exclusion,
historical-evidence boundaries, actual exit/signal handling and deadline cleanup.
They also reject weaker reserves, unknown children and existing output paths;
confirm preflight refusals never signal discovered PIDs; exercise observer
interrupt/error cleanup; and confirm a failed signal-journal write cannot strand
the owned child. Synthetic `/proc` cases establish sanitized signature/large-RSS
behavior without inspecting or signalling a real conflicting process.

Default preparation SHA-256:
`34db95a7dbdcbfbfb6b108943d5cb172b2026d5db8f8528942032be5963816c4`.
CPU verification SHA-256:
`9a40d5e3e6edfdfddbe767f0a39671589f71b9a98475ec43c394951c595f9ac1`.
The JSON records all current source identities. These package hashes are not
a substitute for the root's separately collected stable course-wide test panel.

Reproduce preparation without spawning any child:

```bash
PYTHONPATH=src /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python -m dongxi_llms.staged_campaign \
  --report /tmp/NEW-campaign-preparation.json
```

Explicitly verify only fixed CPU children into new paths:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=src /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  -m dongxi_llms.staged_campaign --report /tmp/NEW-campaign-supervisor.json \
  --verify-cpu-fixtures --fixture-workspace /tmp/NEW-campaign-supervisor-fixtures

CUDA_VISIBLE_DEVICES= PYTHONPATH=src /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  -m unittest discover -s tests -p test_staged_campaign.py -v
```

## Pending external evidence

The production campaign still needs separate exact acquisition and stage
approval, local checkpoint/data/interface validation, actual Spark profiles,
finite smoke and pretrained completed/pending recovery, enforceable token/disk/
deadline budgets, fixed raw evaluation and actual branch genealogy. Existing
pretrained DPO/RLVR runners lack the required verified replay interface. The
CPU recovery lab does not satisfy that production gate.

This supervisor is restricted to a directly owned leaf child. It neither
contains grandchildren nor supervises a production launch tree; it does not
authenticate approval, prevent every memory spike or replace a platform cgroup.
Production integration, model quality, cross-device replay, Mac execution and
hosted supervision remain unverified. Every external stage remains null/pending,
the improvement campaign is not complete and the learner stays Day 9.
