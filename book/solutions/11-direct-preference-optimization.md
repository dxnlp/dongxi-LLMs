# Chapter 11 — Worked Solutions

Companion to [Direct Preference Optimization](../chapters/11-direct-preference-optimization.md).
The [lab route](../labs/11-direct-preference-optimization.md) links the runnable references.

## 1. Exponential tilt and support

Differentiate the Lagrangian with respect to $\pi_y$:
$r_y-\beta(\log(\pi_y/\pi_{\mathrm{ref},y})+1)+\lambda=0$.
Exponentiation gives a reference weight times $\exp(r_y/\beta)$ times a
shared constant; normalization determines $Z$. Positive reference support,
finite reward, positive beta, and an unrestricted probability simplex establish
the finite optimum. A neural policy's capacity and shared prompts impose
additional constraints. Zero reference mass forbids positive current mass under
finite forward KL.

## 2. Partition cancellation

The reconstructed reward is $\beta\log(\pi/\pi_{\mathrm{ref}})+\beta\log Z(x)$.
Both answers share $x$, so subtracting rewards removes $\log Z(x)$. Responses
to different prompts generally have different partition terms, so the same
pairwise cancellation does not justify arbitrary cross-prompt reward comparisons.

## 3. Log-probability derivatives

Let $m=\beta(c-l-c_{\mathrm{ref}}+l_{\mathrm{ref}})$ and
$L=\mathrm{softplus}(m)-qm$. The derivatives are
$\beta(\sigma(m)-q)$ for $c$ and its negative for $l$. At policy/reference
equality with a hard winning pair, they are $-\beta/2$ and $\beta/2$.
Reference derivatives are absent because it is fixed; code must enforce that
boundary, not assume it from a variable name.

## 4. Two beta comparisons

For fixed reward, increasing beta reduces the exponential tilt $r/\beta$.
For fixed current/reference equality and one preference example, the initial
gradient magnitude is beta over two. For fixed soft preference target, the
required pair log-odds movement is $\log(q/(1-q))/\beta$. These compare
different held-fixed quantities. Learning rate and training duration mediate
actual neural movement, so beta is not a one-dimensional “aggressiveness” knob.

## 5. Ratios do not control absolute likelihood

Suppose chosen probability changes from .2 to .1 and rejected probability from
.1 to .01. The ratio increases from two to ten while chosen probability halves.
The remaining mass goes elsewhere. DPO's pairwise likelihood can improve under
that change. Shared transformer updates make such movements possible; independent
generation evaluation is therefore needed. The tiny decoder experiment preserves
an instance of this diagnostic tension rather than hiding it.

With the first distribution as reference and $\beta=0.5$, the margin changes
from zero to $0.5\log(10/2)=0.804719$ and hard-pair loss from 0.693147 to
0.369640. The chapter's three-answer table makes the missing mass explicit.
This is a constructed counterexample to what pair loss guarantees, rather
than a measured training path. Keeping the third probability fixed would
remove this freedom and force higher chosen probability for higher pair odds.

## 6. One causal shift

For token positions `[prompt_0, prompt_1, A, EOS, PAD]`, model inputs are the
first four positions and labels the last four. The aligned scored labels are
`[unscored_prompt_1, A, EOS, unscored_PAD]`. Their completion mask is
`[False, True, True, False]`. A is predicted from the state at prompt_1. Use
that target mask for likelihood; use a separate attention mask to prevent pad
positions supplying context.

## 7. Sum versus average

A product of conditional probabilities becomes a sum of log-probabilities.
Dividing by token count gives the logarithm of a geometric mean per token,
not the probability of the entire sequence. It may be useful in a separately
specified length-adjusted objective, but the original derivation no longer
applies unchanged. Inspect preference-length bias and EOS errors before changing
the mathematical contract.

## 8. Reference-cache identity

Checkpoint weights, tokenizer revision, chat template, encoded prompt and answer,
EOS and mask boundaries, dtype, stochastic evaluation settings, and sequence
normalization all affect cached scores. Cache each encoded branch with an
identity covering these inputs. After changing a template, the same raw answer
string is not evidence that the cached log-probability remains valid.

## 9. A controlled comparison

Keep one SFT checkpoint untouched. Train DPO on frozen pair data; optionally
train chosen-SFT and shuffled-label arms from copies of the same checkpoint.
Freeze independent task prompts and decoding. Report examples, valid completion
tokens, updates, gradient passes, wall time, and optimizer settings. Chosen-SFT
does not see the rejected answer; shuffled labels test dependence on judgment
direction. These controls answer different questions and should not be called
equally informative simply because the update count matches.

## 10. Preference gain, correctness regression

The measured result supports improved ranking on that held-out preference
population and worsened correctness on the independent task under the frozen
decoding contract. It does not establish a universal improvement or a single
cause. Inspect absolute chosen/rejected likelihoods, EOS, response length,
format/rubric correlation, reference drift, and sample-level failures. Check
shift and masks before expanding the training budget. Preserve the negative
comparison and its confidence intervals.

## 11. Explicit auxiliary gradients

Global chosen-token NLL is $-\sum_i c_i/\sum_i n_{w,i}$, so its derivative
with respect to any $c_i$ is $-1/\sum_i n_{w,i}$. Multiply by $\lambda_{\mathrm{SFT}}$ and add
the mean-pair derivative $\beta(\sigma(m_i)-1)/B$. The counts are fixed data,
not differentiable policy outputs. Rehearsal adds a separate parameter
gradient and remains present when gamma is nonzero, even if $\lambda_{\mathrm{SFT}}=0$.
Only $\lambda_{\mathrm{SFT}}=\gamma=0$ recovers ordinary DPO; the independent parameter-gradient
tests verify exact recovery and the explicit combined expression.

## 12. More supervision can teach the wrong thing

Chosen NLL points toward the recorded winner, not an independent oracle. A
swapped label can therefore supply a wrong supervised target. It also cannot
guarantee another task's retention, because shared parameters couple the tasks.
Rehearsal supplies that task's demonstrations, but its finite weight and
examples do not protect every behavior. Compare an untouched warm model,
all fixed arms/seeds and per-task generation. The new symbolic comparison
cannot establish a universal remedy or pretrained language capability.

## 13. Length and information controls

Longer-chosen pairs introduce both length asymmetry and a different recorded
style. Matched-long pairs remove the branch length asymmetry while retaining
the style change. Neither changes the independent canonical answer+EOS
evaluation. DPO uses sequence sums; chosen NLL uses its declared token mean.
Margins can rise while both absolute likelihoods fall, or while free generation
breaks stopping/style. Rehearsal adds targets and forwards; chosen NLL reuses
scores but adds objective supervision. Publish those budgets instead of calling
equal updates equal compute. The [new notebook](../../notebooks/day-18/02_dpo_retention_and_preference_controls.ipynb)
places runnable references beside each prediction.

## 14. The original reference must survive recovery

DPO compares current chosen/rejected log-likelihoods with the original frozen
reference's corresponding scores. At update 3, the current policy generally
differs from that reference. Replacing the reference by the resumed policy
resets its relative margins at the recovery boundary and supplies different
gradients for subsequent updates. Equal tensor shapes and token meanings do
not imply equal distributions or equal objectives.

Retain the actual original reference state and its independently bound identity,
along with policy, Adam, sampler/RNG and cumulative work. Validate all of them
before applying a completed snapshot. A new output directory changes invocation
provenance, not the reference. An intentional reference reset needs a separately
named intervention; it is not same-recipe recovery.

The [later native DPO replay](../../experiments/reports/native-dpo-replay-20261005-run-03/acceptance.json)
checks this on the actual full400 parent: the original reference remains
unchanged and the fresh-process result matches the clean numerical state and
metric tail. All native children and the separate CPU comparison exit 0.
That is stronger evidence than an export existing, but still not evidence
that preference training improves independent answers. Earlier timed-out
invocations and their already-spent work remain in the experiment history.

## 15. A seed is not a sampling protocol

Shuffled cycling visits each pair once per traversal; replacement sampling can
repeat one pair and omit another. Identical seed values do not make these two
algorithms consume the same examples or random draws. Compare actual index
traces, encoded chosen IDs/masks, source/data identity, valid chosen targets
and private sampler endpoints from the same parent and fixed recipe.

The [matched control](../../src/dongxi_llms/chosen_sft_control.py) uses the native
DPO replacement sampler for the chosen-SFT accumulation window. Its chosen loss
is a global valid-target mean, whereas DPO is a mean of preference-pair losses
using response sums. Matched chosen exposure does not remove this objective
difference or DPO's rejected/reference work. Freeze independent evaluation and
retain all outcomes; do not infer preference quality, equal compute or production
recovery from a matching index trace.

## 16. Recover the chosen objective, not its unspent allowance

The numerical trajectory contains six completed updates, but the failed earlier
attempt also reserved a full window. Restoring update 2 is not refunding that
attempt: the same physical journal retains seven reserved updates,56 chosen
targets and 336 input positions versus six completed updates,48 targets and 288
positions. The [two fresh-process tests](../../experiments/reports/2026-10-05-chosen-sft-accounted-recovery.md)
recover exact weights, Adam, RNG, history and next replacement draws.

Chosen-SFT's training objective has no rejected supervision or frozen-reference
forward. A separately counted preference-validation panel can still evaluate
chosen and rejected scores against a fixed reference. A payload validator
requiring DPO margins/reference state would check a different algorithm. Bind
its actual global chosen-token denominator, full forwards, masks, sampler and
optimizer layout with its own contract. Then independently bind payload bytes
and both journal prefixes before application. The counted opt-in API does not
make the legacy CLI accounted or supply pretrained/GPU quality evidence.

The later [actual native replay](../../experiments/reports/2026-10-05-native-chosen-replay.md)
supplies pretrained recovery evidence at two updates, not a quality pass. Its
source trajectory contains 40 chosen labels/266 policy positions, while source
plus fresh resumed tail physically presents 60/400. Both independent generation
panels remain 0/4 correct with 4/4 natural stops. Do not add validation reservations
to those training counts or substitute this recovery state for a fresh pilot.

## 17. A pairwise winner is not necessarily the generated answer

The actual 100-update pilots match parent, replacement draws, chosen IDs/masks
and 2,047 chosen targets. They do not match rejected supervision, objective
reduction, full forward geometry or reference work. DPO additionally scores
1,600 rejected targets and makes 800 reference training forwards. Chosen-only's
diagnostic reference forwards belong to validation, not its training objective.

A favorable reference-relative margin means that the chosen/rejected odds
improved against the fixed reference. It does not require the chosen answer
to beat every unrecorded continuation, or each chosen token to win the greedy
decision at its prefix. Free generation can leave the recorded answer path.
Both sequence probabilities can also fall while their relative margin rises.
Natural EOS is not exact answering; an omitted article is not the same error
as an omitted object name, even when both fail the unchanged strict contract.

The [native report](../../experiments/reports/2026-10-05-native-preference-comparison.md)
retains the own-run FP32/BF16-autocast 0/4→4/4 chosen-only and 0/4→1/4 DPO
panels. The separately accepted common comparison observes the same location
counts with BF16-loaded greedy 64 generation. Both retain 120/120 original
instruction answers; both keep the same five correct reasoning IDs and add
only `math-10`, changing the annotated diagnostic from 5/20 to 6/20. All responses
stop naturally, with no caps or item-level correctness loss in these panels.

The four validation pairs and four held-out location prompts also differ.
An aggregate margin cannot isolate the cause of a specific held-out omission.
The nine diagnostic math rows reflect the original fixture/development and
RLVR-overlap policy, not training examples seen by these assistant descendants.
This annotation does not reconstruct upstream pretraining contamination.

Mean chosen logp improves in both arms: −0.003524 for chosen-only and −6.642914
for DPO versus the parent's −11.486888. DPO's stronger rejected suppression
produces the larger unscaled relative margin, not more exact answers. The
eleven controlled held-out reasoning items change 0/11→1/11; the remaining
nine seen/development diagnostics stay separately labeled. Known overlap,
bounded parsing, shared instruction templates, one recipe and one seed limit
the claim. No universal ranking, broad reasoning gain or human preference
assessment follows. Do not substitute own-run precision settings for the
common results, pool the different tasks, or add overlapping cost lanes.
