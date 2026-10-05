# Reasoning is measured at a checkpoint, interface and output budget

This educational Spark study compares pinned Qwen3-0.6B-Base and Qwen3-0.6B
Instruct under the original raw/chat/thinking32/128-token conditions. It does
not establish general mathematical competence, faithful reasoning traces,
deployment safety or a public model release. Both subsequent G4/G8 training
children completed16 updates; their four common post-training evaluation
conditions now retain400 actual new responses.

## Actual ancestry and interfaces

Base uses revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; Instruct uses
`c1899de289a04d12100db370d81485cdf75e47ca`. The declarations and actual local
file identities are retained in each preparation/acceptance receipt. A pinned
cache name is not remotely authenticated proof of correspondence to upstream.
Thinking-off and thinking-on share the Instruct weights but render different
input token sequences. Neither toggle makes Instruct into Base.

RLVR starts separately from the pinned Instruct parent. It does not inherit
the course's custom-template full400 assistant, chosen100/DPO100 descendants,
disposable recovery weights or randomly initialized story model. Recovery
exports are gates, not pilot parents.

## Data, evaluation and missing responses

The original panel has20 authored items, four predeclared sampled seeds and
a separate greedy diagnostic. Nine fixture-development/known-overlap
diagnostics stay separate from eleven controlled held-out items. Only two
items have the actual `rlvr_train_problem_overlap` flag. This annotation is
not evidence of upstream contamination or mathematics training in the separate
assistant/preference branch.

Six complete initial conditions retain600 responses. Base/chat32 and128
each retain40 responses, including one decode error, and leave60 missing.
Thus680/800 planned responses exist;120 are missing. The two errors share the
same problem, attempt seed and24-action prefix, so they are not independent
rare-event samples. ID151768 is inside the model's configured output range
but absent from the decoder mapping. No filtering, resampling or retroactive
acceptance changes the original experiment.

| Complete initial conditions, held-out only | Sampled correct | Greedy correct |
| --- | ---: | ---: |
| Base/raw32 |0/44|0/11|
| Base/raw128 |2/44|1/11|
| Instruct/thinking-off32 |0/44|0/11|
| Instruct/thinking-off128 |13/44|7/11|
| Instruct/thinking-on32 |0/44|0/11|
| Instruct/thinking-on128 |0/44|0/11|

The frozen bounded whole-output parser distinguishes supported correctness,
unsupported grammar, invalid answers and errors. `UNSUPPORTED` does not mean
mathematically wrong, and permissive format validity does not mean accuracy.
Instruct/off128 includes two correct-but-capped held-out samples; all eleven
greedy answers stop naturally. Both thinking-on caps exhaust every response
budget. Correctness, natural stopping, truncation and visible reasoning are
separate outcomes. Any-correct sampled availability is not a tested selector
or the answer actually delivered by greedy decoding.

## Execution and recovery boundaries

Initial generation loads BF16 weights and uses eager CUDA/full-prefix uncached
decoding. This is a recorded code path, not an instrumented kernel trace.
Original whole-child deadlines and25-GiB reserve remain enforced by the sampled
supervisor; point samples do not prove a continuous memory floor. Emitted
tokens, forwarded positions and child/supervisor/outer time have different or
overlapping scopes and are not interchangeable FLOP or billing measures.

G4 recoveryrun01 has three actual native exits0, but its pending-resume
supervisor fails the observed-descendant cleanup acknowledgment. Its final
exact state comparison is not reached and acceptance is absent. A separately
declared, CPU-reviewed controller-only import correction precedes fresh
recovery02, which now passes all twelve recovery checks with three actual
supervised exits0 and equal final numerical state/update2 metrics. Its pending
resume reuses the saved rollout and retains the same physical journal. The
completed-resume role passes the cleanup acknowledgment gate with an observed
owned descendant;
the pending role has no observed descendant. This is not a controlled proof of
the historical timeout's import-latency cause. The original failure/spending
remain retained, and exact replay does not establish a task-quality gain.
G8 recoveryrun01 also passes all twelve recovery checks with three actual
supervised exits0 and exact final state/update2-metric comparison. Its native
child times are182.051605/137.331498/101.909522 seconds; the enclosing
504.835865-second adapter interval overlaps them.

Both fresh fixed16 pilots now have completed native training/export evidence.
G8 passes its three pilot checks and supervision acknowledgment. G4's actual
native exit is0 but its returned supervisor fails only final logging; its original
failed status and missing pilot acceptance remain unchanged. Separate
export-consistency observations verify both final16 artifacts; the new explicit
run02 consumer does not refitG4 or reclassify its failed supervision.

| Actual fixed16 training | G4 | G8 |
| --- | ---: | ---: |
| Training responses, all naturally stopped |64|128|
| Valid response targets |1,004|2,335|
| Dense slots/sampling draws, including post-stop masks |1,508|4,888|
| Optimizer applications |16|16|
| Positive task rewards or nonzero task advantages |0|0|
| Own four-prompt strict greedy accuracy, before/after |0/4 →0/4|0/4 →0/4|
| Whole native child seconds |1,275.736629|1,277.196072|

Zero task advantages remove the task-reward term's gradient, not the reference-
KL term or AdamW application. Equal updates therefore do not imply equal useful
tokens or physical rollout work; neither completed training nor a nonzero
gradient establishes new arithmetic ability. G8's own diagnostic strings change
without becoming strict correct answers. These four-prompt diagnostics are not
the common20 held-out evaluation.

## Common post-training outcomes

The explicit run02 [cap32](../../experiments/reports/native-rlvr-common20-cap32-20261005-run-02.json)
and [cap128](../../experiments/reports/native-rlvr-common20-cap128-20261005-run-02.json)
comparisons join400 new RLVR responses with200 previously accepted unchanged-
Instruct responses; they are not600 new generations. All four new evaluation
children pass their four checks and exit0 with completed supervision. The
original failed G4 training supervisor remains failed.

| Controlled held-out slice | Unchanged Instruct | G4 final16 | G8 final16 |
| --- | ---: | ---: | ---: |
| Cap32, four sampled attempts per item |0/44|0/44|0/44|
| Cap32, greedy |0/11|0/11|0/11|
| Cap128, four sampled attempts per item |13/44|11/44|13/44|
| Cap128, greedy |7/11|7/11|7/11|

All-original cap128 correctness, pooling four samples and one greedy diagnostic
over20items, rises34/100→35/100→36/100. That diagnostic-containing aggregate
does not establish held-out learning or a universal model rank. The two-sample
G4 regression is also not population-level evidence of degradation. Correct-
but-capped held-out sampled answers are2/2/1 respectively; correct answers and
natural stopping remain different outcomes. Nine development diagnostics are
not nine proved training overlaps: only two items carry the actual RLVR overlap
flag. The predeclared final16 checkpoint was not selected by these results.

Current evaluation durations include their own load/generation/terminal work.
CPU notebook verification overlaps parts of the G8 evaluation, so timings are
not isolated throughput measurements. No positive reward-driven learning,
new-task transfer, training-seed generalization or broad reasoning gain is
established by this fixed small experiment.

Read the [source-bound reasoning report](../../experiments/reports/2026-10-05-native-reasoning-rlvr-evidence.md),
[independent initial review](../../experiments/reports/2026-10-05-native-reasoning-baselines-independent-review.md),
[cleanup diagnosis](../../experiments/reports/2026-10-05-native-rlvr-recovery-cleanup.md)
and [controller review](../../experiments/reports/2026-10-05-native-rlvr-controller-runtime-independent.md)
for exact raw records, invocation identities, costs and acceptance boundaries.

## Intended and excluded use

Use the local artifacts to study evaluation contracts, output budgets,
sampling, saved-rollout recovery and reward gradients. They are not verified
factual assistants or safety-reviewed services. Broad multilingual behavior,
human preference, robustness, training-seed transfer, Mac execution and
cross-machine model reload remain unassessed. Any release needs separately
authorized terms/redistribution review; this card does not authorize it or
advance the learner's Day9 position.
