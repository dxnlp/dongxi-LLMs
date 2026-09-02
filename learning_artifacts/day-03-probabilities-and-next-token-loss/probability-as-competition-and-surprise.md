# Probability as Competition and Surprise

- Day: 03
- Date opened: 2026-09-01
- Status: introduced
- Book destination: Chapter 3 sections on logits, softmax, and negative
  log-likelihood
- Related evidence: planned manual derivation and PyTorch verification
- Related production tasks: `ANIM-CE-001`, `ANIM-NTP-001`,
  `ANIM-LOGLOSS-001`; candidate
  `CAND-ANIM-001` and user-approved candidates `CAND-ANIM-002` and
  `CAND-ANIM-003`, and `CAND-ANIM-004`; temperature candidate
  `CAND-ANIM-007`

## Questions that drive the discussion

- Is a next-token probability an absolute belief or a relative allocation among
  the tokenizer's candidate IDs?
- Why do relative logit differences matter while a shared additive offset does
  not?
- Why should increasing one candidate's score reduce other candidates'
  probabilities even when their logits stay fixed?
- Why does the training objective use the negative logarithm of the observed
  target probability?
- How is predictive confidence different from factual truth or calibration?

## Learner's starting model

From Day 2, the learner correctly explained:

```text
equal logits → equal exponential weights → equal normalized probabilities
```

The learner also understood that softmax creates competition: adding or
strengthening another plausible token can reduce the probability of an unchanged
candidate. Token probabilities are therefore conditional on the context,
tokenizer, vocabulary, model parameters, and decoding temperature rather than a
standalone percentage of truth.

## Mechanism to develop

For one contextual state $h$, the model produces one logit $z_i$ per candidate
ID. Softmax converts relative evidence into a distribution:

\[
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}.
\]

The next conceptual step is to interpret the observed target probability as
predictive surprise:

\[
L = -\log p_y.
\]

The negative logarithm is not merely a convenient curve. It turns products of
conditional sequence probabilities into sums of token-level surprise, severely
penalizes confident errors, and connects language modeling to information and
compression. These claims still require a careful derivation and executable
verification.

Three independent reasons are now introduced and approved for animation:

1. the chain-rule product of conditional token probabilities becomes an additive
   sequence NLL under `-log`;
2. through softmax, log loss retains a strong target-logit correction when the
   model is confidently wrong, unlike the intuitive `1-p_target` alternative;
3. expected log loss is a proper scoring rule, so its optimum matches the target
   conditional distribution rather than collapsing onto only the modal outcome.

The learner judged all three mechanisms animation-worthy. They remain
`introduced`, not `verified`, until the Day 3 analytical and executable evidence
is present.

## Decoding ties and temperature

After softmax, vocabulary coordinate $i$ is the probability of token ID $i$.
Greedy decoding selects an `argmax`. If several coordinates have exactly the
same maximum, the mathematical argmax is a set rather than one unique ID. A
concrete implementation must use a tie policy: common tensor `argmax`
implementations return the first maximal index, while a decoder can deliberately
sample uniformly among the tied maxima. Exact ties are uncommon with trained
floating-point logits, but they can arise through symmetry, initialization,
quantization, or constructed examples. Near-ties are not exact ties; numerical
precision and kernel details may then determine the ordering.

Temperature rescales the logit gaps before softmax:

\[
p_i(\tau)=
\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)},
\qquad \tau>0.
\]

Its clearest interpretation comes from a probability ratio:

\[
\frac{p_i(\tau)}{p_j(\tau)}
=
\exp\left(\frac{z_i-z_j}{\tau}\right).
\]

- $\tau=1$ leaves the ordinary softmax unchanged.
- $0<\tau<1$ magnifies logit gaps and sharpens the distribution.
- $\tau>1$ shrinks gaps and flattens the distribution.
- As $\tau\to0^+$, mass concentrates on the maximal logits; exact tied maxima
  remain equally weighted under pure softmax.
- As $\tau\to\infty$, a finite logit vector approaches the uniform distribution.

Dividing by a positive temperature preserves logit ordering, so it cannot change
the greedy argmax. A literal $\tau=0$ is undefined in the formula; interfaces
that offer "temperature zero" normally implement greedy decoding as a special
case or reject the value.

Temperature and top filtering have different roles. Temperature continuously
changes relative probability ratios among candidates. Top-$k$ removes all but
the $k$ highest-ranked candidates; top-$p$ retains a smallest high-probability
prefix whose cumulative mass reaches a threshold. The survivors are
renormalized and sampled. Temperature can change which candidates qualify under
top-$p$ because it changes cumulative probabilities, while positive temperature
does not change the ranking used by top-$k$.

## Evidence state

- `demonstrated`: equal logits produce equal probabilities.
- `introduced`: probabilities are normalized competition over the current model
  vocabulary.
- `introduced`: exact greedy ties require an implementation policy, and
  temperature rescales logit gaps without changing their ordering.
- `not yet demonstrated`: executable unequal-logit ratios and temperature
  limits, negative-log surprise, sequence likelihood, cross-entropy, and
  perplexity.

## Learner prediction and refinement

When asked what happens to valid but unobserved continuations, the learner
predicted that training assigns probabilities to those alternatives. The forward
distribution does assign probability to every candidate, so alternatives can
retain nonzero mass.

The important refinement is that one ordinary one-hot cross-entropy example does
not label the other candidates as valid. Its local gradient rewards the observed
target and generally pushes every non-target logit down relative to it, including
semantically valid alternatives. Such alternatives gain support only through
other observations, shared contextual generalization, or a different target
construction such as soft or multi-target supervision. In expectation over a
representative data distribution, maximum likelihood can recover probability
mass across alternative continuations even though each individual sample names
only one next token.

## Open edges

- Explain why adding one constant to every logit leaves softmax unchanged.
- Connect one-logit differences to probability ratios without making arithmetic
  the main lesson.
- Derive negative log-likelihood from conditional sequence probability.
- Distinguish low loss, calibration, factual accuracy, and capability.
- Verify the manual example against PyTorch.
