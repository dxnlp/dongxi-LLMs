# Chapter 13 lab route — grouped rewards to a decoder update

Prerequisites: Chapter 12 policy gradients and Chapter 5 decoder tensor shapes.
Machine: CPU on Mac or Spark; no server/model download required. Use the course
Torch/Matplotlib environment. Each notebook includes prediction prompts,
adjacent executable answers, measured plots and a controlled modification.

| Day | Session | Main question | Intervention |
|---|---|---|---|
|22|[Group advantages](../../notebooks/day-22/01_group_advantages.ipynb)|What signal survives reward normalization?|Population versus sample std; constant group|
|22|[Ratios and KL](../../notebooks/day-22/02_token_ratios_kl.ipynb)|Which probability movements receive pressure?|Advantage sign; clipping; exact KL derivative|
|22|[Matched-rollout objectives](../../notebooks/day-22/03_objective_weighting_and_filtering.ipynb)|What changes without new samples?|Reductions/scalings, analytic gradients, clipping and rejected-group costs|
|23|[Autoregressive RLVR](../../notebooks/day-23/01_decoder_rlvr.ipynb)|How do sampled answers reach decoder parameters?|Group 4 versus8 with cost recorded|
|23|[Verifier contract](../../notebooks/day-23/02_verifier_contract.ipynb)|What exactly is being rewarded?|Substring exploit and strict parsing|
|23|[Constructive reasoning controls](../../notebooks/day-23/03_reasoning_tasks_and_positive_controls.ipynb)|Can a learnable task improve while minority examples still fail?|Three sampled-decoder seeds; frozen/oracle/lookup controls; EOS/cap and held-out slices|

Start with a prediction about all-failure groups. Inspect reward/advantage arrays,
then ratios and gradients. Run the real tiny decoder update only after explaining
the detach boundary. Read every held-out row, including failures. The
[bounded report](../../experiments/reports/2026-10-04-grpo-diagnostics-distillation.md)
records actual CPU execution; its result is not a Qwen score.

Study the additive constructive control after that negative result, not instead
of it. The [new specification](../../experiments/specs/2026-10-04-reasoning-controls.md)
freezes source/template/task-family splits and a sampled shared-decoder budget;
the [report](../../experiments/reports/2026-10-04-reasoning-controls.md) retains
every seed, minority failure, raw evaluation response and cost. Exact expectation
is diagnostic. The separate lookup arm is an intentional memorization control.
The English math panel validates oracle/grader coverage only; the symbolic
decoder is not an English reasoning model. Five data-backed/schematic figures
and adjacent answers distinguish those levels of evidence. Explain the always-one
baseline before interpreting a perfect positive-only slice.

The matched-rollout extension applies no optimizer updates. First explain the
exact reduction coefficients; then compare analytical selected-score derivatives
and measured shared-parameter gradients on unchanged samples. Keep the clipping
fixture labeled as constructed, and the retry-cost experiment separate from the
first-attempt objective pool. [Its report](../../experiments/reports/2026-10-04-grpo-objective-controls.md)
retains all three seeds and rejected groups. Regenerate five figures; do not
interpret cosine or gradient size as a ranking of reasoning capability.

The [Spark model-scale contract](../../experiments/specs/day-23-qwen-rlvr.md)
specifies the optional locally loaded Qwen pathway, smoke budget and acceptance
gates. Model files and platform resources are required for that separate run.

Actual model execution additionally requires an existing `--environment-lock`.
Shared identity fingerprints supported saved token semantics, encoding, template
and stops and journals source/environment/input evidence before loading. Legacy
course checkpoints fail closed without explicit audited adoption. The
[identity lesson](../../notebooks/day-01/04_checkpoint_interface.ipynb) and
[CPU report](../../experiments/reports/2026-10-04-run-identity.md) explain the
boundary; tiny random HF execution is not a pretrained Spark result.

### Recovery is a separate source exercise

Read §13.7.1 with the [runner-owned source plan](../../docs/PRODUCTION_RECOVERY_PLAN.md).
Explain the completed and post-full-collection boundaries before inspecting the
implementation. A pending pool must be applied once without collection; original
reference, old likelihoods, source/RNG, complete numerical history and collected
versus applied work all belong to that comparison. Mid-generation continuation
and partly applied optimizer operations are explicitly outside that guarantee.

The optional actual-model CLI requires an explicit `--snapshot-max-bytes` even
for a new run. Choose it only within an approved save/load memory and disk envelope;
the tiny CPU test cap is not a production-size recommendation. Resume jointly
supplies `--resume`, independently retained `--resume-contract`, expected
`--resume-sha256` and `--resume-bytes`, with a new exclusive `--output` directory.
An unchecked adjacent marker cannot serve as its own expected receipt. Source,
parent, lock, tokenizer and numerical recipe are rechecked rather than bypassed
because only the output path changed. These are source contracts, not permission
to load a pretrained model or launch a Spark run.

New runs also require `--work-limits` with all23 RLVR logical dimensions and
`--work-journal-max-bytes`; resume supplies the same physical `--work-journal`.
Whole collection and application, current/old/reference scoring, full evaluation
and recovery-validation work are reserved before their operations. Pool checking
can itself run a likelihood forward or temporary sampling replay; count that
separately from new training. A pending pool remains fixed and applies without
resampling. Budget refusal never filters an inconvenient response or shortens
the original temperature-one/full-support objective. Logical positions, actual
valid actions, rectangular slots and unknown failed-call work remain distinct;
none supplies a physical compute/storage quota or pretrained result.

Actual CLI paths also require complete explicit `--snapshot-io-limits` and a
separate `--snapshot-io-ledger`; resume supplies `--resume-io-receipt`. Its
payload bound must match the I/O envelope. Both independent journal prefixes
are bound before shared inspection/model allocation. Identity hashing must not
read the resume payload first, and a terminal pending pool refuses before any
output/journal/model creation. The bounded metadata path and other source/data
identity reads remain outside the I/O9 claim.

Continue to Day25 Exercise8 and Chapter14 worked answer33. Predict the first
application's collection/RNG changes from each phase; then inspect actual retained
IDs, native trajectory equality and permanent rejected-load work. Distinguish
that manual one-boundary experiment from the full source lifecycle's
$1+2U$ fresh save schedule. These newly explicit source requirements are not
authorization for a pretrained run or an upgrade of old work capacities.

### Read actual evidence without repairing the result

Read Chapter13§13.8.3 and the
[partial native report](../../experiments/reports/2026-10-05-native-reasoning-rlvr-evidence.md).
This is a receipt-reading exercise; do not load a model, launch a queued stage,
edit a response or replace a grade. Instruct/thinking-off/cap32/cap128 and
Base/raw/cap32/cap128 and both thinking-on budgets are incorporated as actual
evidence. Base/custom-chat at caps32 and128 is separately failed/partial, not
accepted. All eight conditions were attempted, with680 retained records of800
planned, including two errors and120 missing. G4 recovery run01 fails its
cleanup gate despite three native exits0; fresh02 now has accepted exact recovery.
G8/01 also has accepted recovery and now accepted pilot16. Common evaluation
outcomes and both per-cap comparisons are now measured.
G4 pilot01 has now completed native16/export but failed terminal logging
acknowledgment; it is not accepted. Its export is now separately consistency-closed
without rewriting that failed supervision.

Open its linked preparation, acceptance and returned-supervision JSONs. Trace
sample1009/math10 from the retained raw record to extraction/canonicalization,
format, stopping and grade. Before reading [worked answer19](../solutions/13-group-relative-policy-optimization.md#19-follow-one-response-through-independent-predicates),
write a short defense of these questions:

1. Which fields prove a whole child actually completed, and which merely establish
   response coverage? Why can those checks pass with zero declared correct rows?
2. Why can `5 + 6 = 11` naturally stop and pass `any` format while its declared
   whole-output grade is `UNSUPPORTED`? What is different from a supported wrong
   numeral? Explain both why “no reasoning” overstates this record and why silently
   rescuing the last numeral would change the experiment.
3. Find math9's overlap annotation. Which problem roles belong in a controlled
   held-out headline? Why are44 sampled attempts and11 greedy attempts not55
   independent problems, and why would any-candidate availability not measure a
   delivered best-of-four system? Distinguish the eight original fixture-train
   labels from the two true RLVR-overlap flags on math2/math9. Why does neither
   annotation establish that actual full400/chosen/DPO policies trained on the
   math rows or that upstream checkpoint contamination occurred?
4. Propose one separately declared interface experiment, naming its unchanged
   controls and new evidence obligation. Keep the current raw outputs and grades
   intact; do not claim the proposal has run or would improve the result.
5. Read the Base/raw/cap32 receipt next. Why do its more frequent caps not isolate
   instruction training's causal effect when compared with Instruct/native chat?
   Identify which weight, serialization and logical-panel bindings agree or
   differ, and what the separately failed same-weights custom-chat row does and
   does not establish without complete coverage.
6. Read the Base/raw/cap128 correct rows and their overlap annotations. Why do
   five correct coverage rows become only two sampled and one greedy controlled
   held-out responses? Explain supported boxed extraction, natural stopping and
   any-candidate availability without inventing a best-of-four delivered system
   or concluding that128 tokens always outperform32.
7. Inspect both failed Base/custom-chat verdicts, supervisors and sample1019/math6
   IDs. Locate unmapped ID151768 in the retained24-action trajectory. Explain
   why a decode error differs from mathematical `UNSUPPORTED`, why40 retained
   records cannot replace100 planned responses, and why missing60 is not60 wrong
   answers. Defend preservation of failed work without a vocabulary filter,
   replacement draw, shortened denominator or implied repair. How would you
   distinguish continuing the other fixed conditions from repairing these failed
   rows? Explain why680 retained slots of800 invoked plans, including two errors,
   neither establish800-slot coverage nor add a new mandatory successful-Base/chat
   gate. Keep each child/supervisor time and failed cost under its own boundary.
8. Join the Instruct/thinking-off32 and128 preparations and original response IDs.
   What establishes a cap-only intervention rather than new weights, a thinking
   toggle or a new parser? Explain why13/44 sampled and7/11 greedy are separate
   headline systems, not20/55 deployment accuracy. Inspect correct-but-capped
   math19/seed1009 and math18/seed1019, then distinguish a supported boxed answer,
   natural termination, visible rationale and strict EOS-complete reward. Why is
   any-candidate availability8/11 not an implemented best-of-four delivered score?
9. Read the thinking-on32/128 receipts, then compare their native rendered prompts
   with thinking-off at the same caps. Identify the unchanged checkpoint/template
   and the actual flag-induced prompt difference. Trace sample1009/math10 without
   stripping its thinking text or rescuing an internal numeral. Why do all-capped,
   unsupported results coexist with complete accepted execution and permissive
   format-validity? Defend the100 same-seed/prompt/action-prefix joins across the
   two thinking-on caps without claiming cross-device determinism. Explain the
   extra uncached cost without predicting an unrun longer horizon or ranking
   reasoning ability. See [worked answer21](../solutions/13-group-relative-policy-optimization.md#21-a-supported-thinking-flag-is-not-a-completed-answer).
10. Open the report's failed G4 recovery receipts. Why can three native exits0,
    completed reports, closed work/I/O tickets and terminated helper processes
    coexist with `Observed descendant cleanup timed out`? Locate the missing
    CPU comparison/acceptance instead of calling later process absence a repair.
    Join the shared journal's four applications and three collections to the
    three invocation IDs: why do three numerical cursors of2 not mean six fresh
    updates, and what proves the pending group was not collected again? Trace
    zero task advantages alongside nonzero measured gradients and exact KL,
    without labeling movement reward learning or inferring a skipped update.
    Keep native/supervisor/outer time boundaries and the unchanged600-second
    child envelopes separate. See [worked answer22](../solutions/13-group-relative-policy-optimization.md#22-native-exit0-does-not-waive-the-cleanup-and-replay-gates).
    Then distinguish the separately reviewed controller startup correction from
    a scientific change or a waived gate. Why does fresh G4 run02 require its
    own actual acceptance, and why must pilot01 explicitly select accepted02 rather
    than silently fall back from failed01? Keep prelaunch declarations, CPU
    controls and actual native receipts under separate time/evidence boundaries.
11. Read fresh G4 run02's twelve actual checks and retained closing bindings.
    Which exact final components and update2 metrics establish a real replay,
    rather than merely successful saves or scalar agreement? Explain why its
    shared four applications/three collections are not refunded failed01 costs,
    six fresh updates or four positive-reward groups. Why does pending02's empty
    descendant observation not prove the historical timeout cause? Name the
    separate pilot/export/common20 receipts needed before claiming quality.
    See [worked answer23](../solutions/13-group-relative-policy-optimization.md#23-an-accepted-replay-still-needs-a-separate-quality-experiment).
12. Read accepted G8/01 next. Which twelve actual checks and final components
    establish its own exact replay rather than equality with G4's final policy?
    Trace384 sampled batch slots,314 fresh valid actions and434 applied targets
    through the masks and three invocation histories. What does the pending
    120-target application avoid doing? Why do seven EOS/one cap then two EOS/six
    caps coexist with all-zero strict rewards, and what can the recorded
    gradient/KL values establish without a quality ranking? Keep dense stopped-row
    sampling costs, valid output tokens and paired publication scores distinct.
    See [worked answer24](../solutions/13-group-relative-policy-optimization.md#24-a-batch-slot-valid-action-and-replayed-target-are-different-costs).
13. Read the failed G4 pilot01 report, returned supervisor and adapter failure
    together. How can completed native16/export and actual exit0 coexist with
    `Final logging timed out` and `final_record_retained=false`? Distinguish this
    from the earlier descendant timeout and from an external deadline. What do
    sixteen journaled applications,1508 slots/1004 valid targets,33 saves and64
    natural zero-reward responses establish—and what do they not? Why must the
    default evaluator refuse an unaccepted surviving export? What does the later
    explicitly declared export-consistency receipt validate without rewriting
    that verdict? Compare accepted G8 pilot16's4888 slots/2335 valid targets and
    sixteen zero-advantage groups. Then read the actual common20 comparisons:
    why do all100 correct counts34→35→36 not contradict held-out sampled
    counts13→11→13/44 and unchanged greedy7/11? Explain what correct-but-capped
    counts2/2/1 measure, and why neither fewer caps nor this small fixed-slice
    difference establishes a general method ranking. See
    [worked answer25](../solutions/13-group-relative-policy-optimization.md#25-trained-but-unaccepted-is-neither-no-training-nor-a-quality-pass).

Then read [worked answer20](../solutions/13-group-relative-policy-optimization.md#20-an-iteration-is-not-a-quality-claim)
with the frozen native RLVR application path. Predict what must be inspected to
distinguish constant group rewards, optimizer applications, committed iterations
and held-out improvement. Explain why a pending rollout must be applied without
recollection and why the runner's four diagnostic prompts cannot replace the
original common twenty-item panel. The failed G4 attempt supplies retained
operation/history measurements, not accepted replay or a pilot-quality outcome.
Fresh G4/02 and G8/01 separately establish exact two-update recovery, not pilot quality.
Do not infer the remaining outcomes from a ready adapter or baseline rows.
Generation's BF16/eager label also does
not specify the training kernel or create an inference-throughput comparison.

Completion evidence: explain the objective's reduction/std conventions; prove
old/reference parameters receive no gradient; reproduce the aligned first-token
log probability; identify at least one verifier false positive; interpret
zero-variance groups and one held-out failure. Material execution alone does not
establish that the learner has demonstrated those explanations.
