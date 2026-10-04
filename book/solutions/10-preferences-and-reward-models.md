# Chapter 10 — Worked Solutions

Companion to [Preferences and Reward Models](../chapters/10-preferences-and-reward-models.md).
Runnable references immediately follow predictions in the [notebooks](../labs/10-preferences-and-reward-models.md).

## 1. Soft-target gradients

Writing $L=\mathrm{softplus}(r_a-r_b)-q(r_a-r_b)$ gives
$\partial L/\partial r_a=\sigma(r_a-r_b)-q$ and the opposite derivative
for $r_b$. When the predicted probability is below the observed preference
frequency, gradient descent raises the first score and lowers the second.
These are score derivatives. A shared network's parameter update combines both
score Jacobians and may affect other answers too.

## 2. Offset versus scale

Adding $c(x)$ cancels in $r_a-r_b$, leaving loss, ranking, and probability
unchanged. Multiplying scores by two doubles the margin, preserving ranking
but changing probability toward more extreme confidence. An additive offset is
unidentified by within-prompt pair data. Scale is identified only relative to
the assumed preference-noise model; changing it requires calibration evidence.

## 3. Nonzero converged loss

For repeated identical comparisons with fraction $q$ favoring the first answer,
the optimum predicts $p=q$. Its expected loss is
$-q\log q-(1-q)\log(1-q)$, positive whenever $0<q<1$. Zero expected gradient
at that point indicates a fitted distribution. Individual votes still produce
opposing gradients. Do not interpret irreducible label disagreement as a failed
optimizer or manufacture unanimous labels to force the loss down.

## 4. Cyclic judgments

Take strong preferences A over B, B over C, and C over A. A single scalar model
would require $r_A>r_B>r_C>r_A$, an impossibility. A population mixing different
criteria can produce such judgments without any one annotator being careless.
Inspect subgroup and rubric differences; fit a richer contextual judgment model
if the intended objective needs them. A scalar ranking is a modeling assumption.

## 5. Group-aware splitting

Assign prompt/source groups to train, calibration, and final test before making
pairs. Keep all variants of one prompt in its assigned split. Normalize and
hash exact texts, and audit near duplicates under a declared rule. Swapped
orders of one pair stay in the same split. A random row split would overcount
similar comparisons as independent evidence and can leak answer content.

## 6. Accuracy does not measure confidence

On ten unanimous positive labels, probabilities .51 and .99 both produce ten
correct rankings. Their mean binary Brier scores are $.49^2$ and $.01^2$.
If the labels actually favor the first answer only six times, .99 becomes
overconfident. Ranking, NLL, and calibration must be interpreted against the
same declared label population; a model can win one metric and lose another.

## 7. Temperature fitting is training

Choosing $\tau$ to minimize calibration-set NLL estimates a parameter from
those labels. Its measured performance on the same set is optimistic. Freeze
the temperature before final evaluation and report both the original and
calibrated mapping. Temperature changes certainty but cannot repair wrong
rankings or an unobserved population shift.

## 8. Diagnose length preference

First ask whether length belongs in the rubric. Then construct correctness-
matched length variants and length-matched correctness contrasts. Preserve
format and content provenance, and assess more than one prompt family. If
longer wrong answers win while short correct answers lose under a correctness
rubric, the reward has learned a shortcut. A length penalty alone might conceal
the symptom while introducing a new preference for unhelpful brevity.

## 9. Padding and endpoint selection

For right padding, the final nonpadding index is `attention_mask.sum(-1)-1`.
For arbitrary padding, find the maximum valid position explicitly. Decide
whether the selected representation includes EOS or a turn-ending token, and
hold that choice fixed. Using the last array index scores padding for short
answers; subtracting one too many can score the previous word and drop answer
information. Test two different lengths with identical content boundaries.

## 10. Optimization shifts response selection

A fixed pair set samples ordinary answers. A reward-maximizing policy can seek
unusual answers that exploit the predictor's error. Accurate ordinary held-out
ranking therefore does not guarantee accuracy on optimized responses. Audit
policy-generated samples over time, retain independent outcome measurements,
and look for nuisance features improving while the task stays unchanged. This
is a reason to preserve reward failures, not only good examples.
