# Chapter 15 — Worked solutions

Companion to the [chapter](../chapters/15-distill-evaluate-and-defend.md) and
[Day 26–28 notebook route](../labs/15-distill-evaluate-and-defend.md).

## 1. Availability and selection differ

More candidates make a correct answer available more often under the independent
model, but an imperfect ranker may systematically favor wrong answers. In our
simulator, the frequent wrong class receives the highest score. A larger pool
therefore gives it more opportunities to displace correct candidates. Plot oracle
availability and actual selected accuracy separately, and include generation cost.

## 2. Independence is a substantive assumption

The formula $1-(1-p)^N$ assumes independent candidate correctness indicators with
the same $p$. Shared prompts and a fixed policy do not guarantee independence of
errors in a deployed system, especially with duplicated outputs or reused search
paths. The finite simulator deliberately samples independently. Its formula is
not an empirical law for a language model's correlated answers.

## 3. Winner's curse

Selecting by $Q+e$ conditions on a high score. Among candidates with similar
quality, those with positive scorer error are more likely to win. Unbiased error
over the whole candidate pool can therefore become positive among winners.
Use an evaluator independent of the selecting score and retain candidates so
selection failures can be inspected. The scorer's own winner scores cannot
validate its correctness.

## 4. Rejection changes the prompt mixture

Prompts with low acceptance probability yield few accepted examples or exhaust
their retry budgets. Easy prompts dominate unless collection deliberately
controls per-prompt counts. Track attempted, accepted, rejected and timed-out
counts by source and difficulty. Training on accepted answers may be useful,
but it should not be described as learning the original prompt distribution
without measuring the induced shift.

## 5. Soft targets express alternatives

A generated token gives one observed outcome. A soft teacher distribution can
place probability on several alternatives and communicate relative plausibility.
The student gradient then compares its probabilities with the teacher's full
vector rather than a one-hot label. This preserves uncertainty and also transfers
teacher errors. Correctness requires a separate evaluation contract.

## 6. Temperature derivation

For $L=T^2[-\sum_i q_i^{(T)}\log p_i^{(T)}]$ plus a teacher-only constant,
differentiating log-softmax contributes $1/T$ and the soft-target residual:

$$
\frac{\partial L}{\partial z_i}=T(p_i^{(T)}-q_i^{(T)}).
$$

Without scaling the derivative is $(p_i^{(T)}-q_i^{(T)})/T$. At high temperature,
the probability difference itself tends to shrink as $1/T$, so unscaled
gradients shrink roughly as $1/T^2$ for fixed centered logits. The factor is a
gradient-scale convention, not a guarantee of identical training across
temperatures. The notebook checks analytical/autograd agreement at 1, 2, 4.

## 7. Vocabulary identities must agree

ID 42 can mean different token strings in two tokenizers. Even vocabulary sizes
that match do not establish a coordinate correspondence. A direct categorical
KL requires aligned outcomes. Different tokenizers can instead support teacher
text/sequence supervision or an explicit alignment method. Record tokenizer
revision and special-token/chat-template behavior for both systems.

## 8. Prefix distribution affects the lesson

Offline teacher trajectories train on teacher-selected histories. A student
generating freely visits its own histories, including its mistakes. On-policy
distillation queries teacher targets at those student states and can address
that coverage difference, at additional teacher cost. It does not automatically
remove model-capacity limits or guarantee stable optimization. Compare prefix
coverage and held-out free generation rather than only teacher-forced loss.

## 9. Structural validity is a limited claim

The genealogy checker confirms unique IDs, known parents, no cycles and one
evaluation identity. It does not inspect weight file bytes, verify dataset
contents or judge panel quality. Synthetic fixture hashes are labeled as such.
Real cards must use computed artifact hashes and measured reports, and scores
must be evaluated under the declared decoding and candidate budgets.

## 10. An honest defense includes missing evidence

Say which implementation and CPU mechanisms were verified, which model-scale
runs were actually executed, and which remain prepared. An absent Qwen run
cannot be replaced by a tiny arithmetic gain or a paper's score. Defend the
ready method and its acceptance gates, then name the next bounded experiment.
Release preparation can be complete while a public capability claim remains
unsupported; the model card must preserve that distinction.

In the [historical short-run native case](../../experiments/reports/2026-10-05-current-model-defense.md),
identify the exact Base,20-update disposable profile, separate fresh full/LoRA
references and precision-qualified FP32 validation export. Do not manufacture
a selected400-update SFT parent or DPO/RLVR descendants from those short-run
receipts. Both native replay
paths can pass while generated answering remains0/8. Preserve the failed BF16
handoff beside the distinct FP32 pass and the proxy phrase-score gain beside
all capped responses. A replay pair's455 numerical training targets do not
erase684 physical training plus1,440 development presentations or the failed
attempt's elapsed time. That is a defensible partial account, not a completed
campaign or a claim that the learner has mastered its mechanisms.

The later [actual full and LoRA400 comparison](../../experiments/reports/2026-10-05-native-sft400-comparison.md)
does establish the predeclared full400 parent. Its120/120 whole answers and
natural endings, versus0/120 for Base and merged LoRA, concern40 held-out lexical
groups within three shared templates. Attach later preference descendants only
from their own accepted export receipts. Do not use the degenerate grouped
intervals to claim certainty about new tasks, another training seed or a retuned
adapter recipe.

For the separate fresh story lineage, distinguish 400 completed updates from
the unfinished 14k schedule. The actual rubric joins 192 continuations and both
AI reviewers, with 56 candidates showing at least one disagreement across 64
dimension-level disagreements. Control400's 45 natural endings do not
contradict its 0.0208 mean ending-quality rating: stopping and narrative resolution
measure different things. Its lower fixed dev NLL than half-LR400 is not broad
proof of better storytelling. Keep the 288 absent later cells null, and do not
describe separate AI instances as independent human or model populations. The
[worked evidence report](../../experiments/reports/2026-10-05-native-story-first400-comparison.md)
and portable model-free archive make this conclusion auditable without loading
the trained model.

## 11. A correct candidate does not choose itself

Majority counts eligible answers, not truths. In the actual eight-candidate
parity pool, correct0 receives three votes and incorrect1 receives five.
Gold-blind majority must return1 under its declared rule, while the evaluator's
any-correct diagnostic is true. The latter cannot be used as a deployable
selector without a checker that legitimately distinguishes answers. A model's
likelihood can rank alternatives, but high likelihood is not that checker.
Retain the entire pool, tie rule and invalid candidates; do not select a new
favorable pool after seeing the failure.

## 12. Count work before calling a comparison fair

Equal attempts can generate different valid lengths and stopping patterns.
Equal generation-token caps can also admit different numbers of complete
attempts; our retrospective rule charges a whole overshooting attempt even
though it is not selectable. Neither control includes all prompt-forward,
rescoring, selection and memory work by itself. The actual experiment records
these boundaries separately. All arms include measurement rescoring, so their
times do not establish optimized serving latency. Fix the claim first, then
choose a budget whose measurements address it; disclose the other differences.

## 13. A format gate is not a hidden correctness oracle

The gate checks whether a response satisfies the declared interface: supported
tokens, bounded length, real END and source/deduplication rules. Correctness is
a separate score. Keeping well-formed wrong responses lets top and random
selection compete on the same pool and exposes the effect of the selection rule.
Removing every wrong answer before that comparison would predetermine its
available supervision. In the recorded teacher-data pool,24 of36 accepted
candidates are wrong. Those are executed programmatic responses, not claims
about a pretrained teacher's behavior.

## 14. Match the budget that addresses the claim

Record body-plus-END targets, update count, repeated exposure, padded positions,
teacher attempts/retries and generation work. The recorded top and random arms
both cover12 prompts/examples but expose42 versus46 valid targets per update,
or3360 versus3680 over80 updates. Length-random matches42 without filtering
correctness. Even matching these counts does not equalize target information,
gradients or wall time. Keep both the matched quantities and remaining
differences explicit; held-out free generation, not selection score, measures
the trained student's transfer.

## 15. One realized response is not a distribution

Response SFT provides one-hot targets along selected teacher trajectories.
Even perfect reproduction of those examples does not identify probabilities
for unobserved alternatives or student-generated prefixes. Soft KL instead
provides a full aligned conditional target at a chosen state. The actual tiny
students learn their six selected parent traces but retain different held-out
failures. Keep this empirical sequence objective beside the exact three-logit
KL lesson rather than treating them as the same experiment.

## 16. Printed validity does not prove causal use

An arithmetic checker can establish that a printed total agrees with the
operands. It cannot establish that the model internally reasoned through or
causally relied on that text. The recorded teacher produces80 wrong-step/
right-answer and24 valid-step/wrong-answer responses among432 evaluation
attempts. A final-answer scorer and a printed-step scorer measure distinct
properties, and neither alone proves internal faithfulness. Missing steps in
answer-only outputs are explicitly null, not a negative rationale label.

## 17. Follow the delivered state, not only the endpoint

A valid-format tie may accept a wrong replacement for a correct draft. A later
round can undo it, leaving final accuracy unchanged while hiding a harmful
intermediate decision. The contrarian control accepts54 harmful and54 helpful
changes and returns to its original answer. Preserve proposal grade, acceptance,
delivered grade and stop reason per round. A format score is not a truth score,
and a programmatic flip is not neural reasoning.

## 18. State the unit that is matched

Counting every emitted draft/critique/revision symbol can match the outputs of
independent programmatic binary draws. It does not match prompt forwards,
attention work, rescoring, memory, latency or real model-token billing. Historical
model attempts in the replay also underfill or overshoot whole-attempt ceilings.
Their incurred costs remain recorded even when they cannot vote. Do not call a
shared serialized ceiling a compute-matched inference benchmark.

## 19. An EOS-looking suffix is not a stop event

A capped result has not satisfied the declared natural-stop contract. Keep its
IDs, serialized count and failure reason, but skip the revision callback. This
both accounts for incurred work and avoids treating incomplete advice as usable.
The original normal-path acceptance missed this case; explicit capped-critique
regression tests and a new preserved-source campaign establish the hardened
boundary without rewriting the earlier evidence.

## 20. Separate coverage from target geometry

Prefix source selects the histories at which teacher advice is requested.
KL direction determines the conditional probability mismatch being optimized
there. Change one while holding the other fixed before attributing a result.
Privileged teacher context is a third independent intervention. The actual
factorial lab uses detached aligned targets and excludes hints from student
inputs/evaluation; its poor held-out results do not invalidate the exact
conditional-gradient checks or establish a universal ranking.

## 21. Collection does not automatically create an occupancy gradient

For $J(\theta)=\sum_s d_\theta(s)L_\theta(s)$, the formal derivative contains

$$
\nabla_\theta J=\sum_s d_\theta(s)\nabla_\theta L_\theta(s)
+\sum_s L_\theta(s)\nabla_\theta d_\theta(s).
$$

The lab samples histories and treats them as fixed observations during backward.
It estimates the first conditional term at collected states, not the second
occupancy term. Detached teacher targets are necessary but do not add that
missing derivative. State how an objective is implemented; do not silently
equate current-policy data collection with a full policy-gradient calculation.

## 22. Aggregation preserves mass, not all information

The exact tail decomposition leaves a nonnegative conditional KL inside the
unselected outcomes, weighted by teacher tail mass for forward KL or student
tail mass for reverse KL. Equal retained entries and equal total tail mass make
the bucket loss zero without making those conditional distributions equal.
The full loss and its logit gradient may remain nonzero. The notebook checks
mass/decomposition and this counterexample separately; at $k=V$ it recovers the
full objective. Discarding or arbitrarily renormalizing the tail is a different
operation and does not inherit the same identity.

## 23. The population is part of the result

The original panel varies40 lexical values inside three shared copy/reverse/
extract templates. The full400 checkpoint succeeds on every source group,
making the observed paired difference and its fixed source-group bootstrap
interval equal to one. Resampling those groups cannot create new tasks, natural
conversation or a new application distribution. Report the narrow transfer,
natural endings, costs and failed Base/LoRA controls; do not generalize beyond
what was sampled. A permissive format score is a separate quantity and gives
all three models full marks despite their different answer quality.
