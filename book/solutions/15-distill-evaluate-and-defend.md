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
