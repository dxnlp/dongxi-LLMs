# Chapter 10 — Worked Solutions

Companion to [Preferences and Reward Models](../chapters/10-preferences-and-reward-models.md).
Runnable references immediately follow predictions in the [notebooks](../labs/10-preferences-and-reward-models.md).

## 1. Soft-target gradients

For the fixed prompt, $r_a=r_\phi(x,a)$ and $r_b=r_\phi(x,b)$.
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

## 11. Display order is not answer identity

If the original left answer moves from A to B, a judge choosing that answer must
emit B instead of A. The collector maps the slot back to the original candidate
ID; the canonical left/right outcome stays left. If the canonical pair itself
is swapped, the same candidate is now right and the declared label reverses.
The [collection notebook](../../notebooks/day-15/03_preference_collection_and_judges.ipynb)
tests both operations and retains the displayed text and raw verdict. A mapping
bug can create incorrect labels even with a correct judge.

## 12. Distinct audits expose distinct vulnerabilities

Order consistency measures canonical selection under an A/B display swap.
It says nothing about whether that selection follows correctness or length.
The simulated longer-answer judge has order consistency 1 but changes on 37.5%
of matched verbosity observations. The injection-sensitive judge is also
order-consistent and changes on 37.5% of matched injection observations.
The repeat-unstable rule instead disagrees on 75% of paired repeats. These are
original deterministic controls, not empirical statements about AI judges.
Preserve all category counts and compare against separately reviewed labels;
passing one check does not certify an evaluator.

## 13. Declare the population that receives weight

Equal-pair weighting gives the nine-row source three times the influence of
the three-row source. To weight sources equally, assign per-pair weights 1/9
and 1/3, then normalize across sources. All siblings, swaps and nuisance variants
remain in their source's split. Repeated judgments may estimate a within-pair
preference frequency, but they are not new independent source questions.
For the notebook's authored demonstration, row agreement is 0.5 while
source-balanced agreement is 1/3. The difference is a change of population
weighting, not a changed prediction or contradiction in the raw evidence.

## 14. A tie is not a failed measurement

Equal observed ordinal ratings become a tie under a declared conversion;
missing ratings become an abstention. A duplicate/conflicting JSON field,
unsupported verdict or missing reason makes the observation invalid. A timeout
is an invalid transport observation, retaining any partial raw response.
Do not assign any of these an automatic half-win label. A tie model, binary
soft-target approximation or exclusion rule must be chosen explicitly, with
coverage reported. An ordinal score difference is not a calibrated reward
margin, so keep the raw ratings and conversion version rather than inventing
preference confidence.

## 15. Padding changes array positions not text positions

For a left-padded mask `[0,0,1,1]`, the final valid array index is 3; length minus
one is 1 and selects padding. Use the maximum valid position and assign token
position IDs from the cumulative valid-token count. Reject all-padding rows
and suppress padded query outputs without generating an all-masked softmax
NaN. The [text reward notebook](../../notebooks/day-16/03_text_reward_and_process_labels.ipynb)
measures left/right and extra-padding parity. EOS-inclusive and EOS-excluded
endpoints are separate saved contracts, not interchangeable offsets.

## 16. Terminal correctness does not label the steps

For 2+3, a false “2+3=6” followed by final 5 has terminal target 1 and local
target 0. A valid “2+3=5” followed by final 6 has terminal target 0 and local
target 1. Put terminal supervision at final EOS and local supervision only at
the explicit step marker. Padding and unrelated tokens are not process-label
positions. Use a source-aware weighting rule when sources supply unequal
numbers of labeled boundaries; do not let long traces define the population
accidentally.

## 17. A critic and a process judge have different targets

The process label asks whether a displayed step is valid under the annotation
rubric. A critic estimates expected remaining reward under a policy and its
continuation distribution. A locally invalid step might precede an eventual
correct answer, and a locally valid step might precede a wrong one. Both
quantities can be implemented as scalar heads, but shared tensor shape does
not make their supervision or interpretation equivalent.

## 18. Unknown IDs can destroy the distinction being learned

The held-out arithmetic records contain 7 and 8, both mapped to `<unk>` because
neither appeared in the fitted training vocabulary. Their opposite validity
labels can then have identical token sequences. The numerical model cannot
recover a distinction removed by encoding. All three measured seeds have 0.5
held-out terminal and process accuracy, with poor probability losses retained.
The unseen nouns “box” and “book” also collapse to one ID, making all four
baseline test pairs duplicate calibration encodings despite disjoint source
IDs. This is not independent calibration evidence, general arithmetic learning
or a unique diagnosis of optimizer failure. Changing tokenization is a new experiment, not a
quiet replacement of the failed test cases.

## 19. Freeze the interface and the selection rule

Save complete numeric parameters, architecture, ordered token meanings,
normalization/separators, unknown policy, EOS and endpoint semantics, source
identity and input hashes. Require nonempty lists of unique nonblank source IDs,
then validate the payload and actual tensor shapes on
reload, compare an independently expected interface, then disable reward
gradients. The course's seed 1601 checkpoint was selected before testing and
reloads to exactly equal CPU scores. A self-hash alone is not authentication
against an actor who rewrites all metadata. A frozen reward can still be
exploited, so policy optimization must retain independent factual metrics.

## 20. Audit what the model actually receives

The word vocabulary maps unseen “box” and “book” to one unknown ID. Different
raw source names then produce identical calibration/test inputs. Check complete
prompt/completion sequences, unordered candidate pairs including swaps and
complete causal prefixes at labeled STEP boundaries. Different source IDs
cannot make those inputs independent. Repeated equal-label prefixes within
one source are related observations; arbitrary shared short prefixes are not
the supervised unit. The character arm passes these exact checks, but does
not prove semantic independence, independent human labels or a large sample.

## 21. Information preservation is not an abstract rule

Distinct character IDs ensure that numbers 7 and 8 are no longer indistinguishable
at encoding. The learner could now represent their difference, but training
need not identify arithmetic or a general color-matching rule. All three
character seeds fit training labels and still reach 0.5 ordinary heldout
ranking; terminal accuracy is 0/0.5/0 and local validity 0/0.5/0.5. That rejects
a claim of successful transfer in this fixture, not every possible character
model or a unique optimizer diagnosis. Do not tune more updates or choose a
new seed after observing those test failures under the frozen protocol.

## 22. Calibration and serialization answer narrower questions

Positive temperature scaling preserves the order of reward margins. Temperature
4 chosen on calibration labels can reduce overconfidence and improve heldout
NLL/Brier without fixing the wrong ranking. Exact serialization equality
compares the same saved state and interface within one execution environment.
Fresh training under another Torch version may produce different weights;
record that difference rather than replacing an old checkpoint. The notebook
retains its initial historical-retraining equality failure and verifies exact
same-state reload, while the new character fit/export/reload remains exact in
its locked CPU environment.
