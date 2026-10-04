# Chapter 11 — Worked Solutions

Companion to [Direct Preference Optimization](../chapters/11-direct-preference-optimization.md).
The [lab route](../labs/11-direct-preference-optimization.md) links the runnable references.

## 1. Exponential tilt and support

Differentiate the Lagrangian with respect to $\pi_y$:
$r_y-\beta(\log(\pi_y/\pi_{\text{ref},y})+1)+\lambda=0$.
Exponentiation gives a reference weight times $\exp(r_y/\beta)$ times a
shared constant; normalization determines $Z$. Positive reference support,
finite reward, positive beta, and an unrestricted probability simplex establish
the finite optimum. A neural policy's capacity and shared prompts impose
additional constraints. Zero reference mass forbids positive current mass under
finite forward KL.

## 2. Partition cancellation

The reconstructed reward is $\beta\log(\pi/\pi_{\text{ref}})+\beta\log Z(x)$.
Both answers share $x$, so subtracting rewards removes $\log Z(x)$. Responses
to different prompts generally have different partition terms, so the same
pairwise cancellation does not justify arbitrary cross-prompt reward comparisons.

## 3. Log-probability derivatives

Let $m=\beta(c-l-c_{\text{ref}}+l_{\text{ref}})$ and
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
