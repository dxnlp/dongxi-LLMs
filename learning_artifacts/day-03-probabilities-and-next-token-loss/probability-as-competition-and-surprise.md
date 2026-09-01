# Probability as Competition and Surprise

- Day: 03
- Date opened: 2026-09-01
- Status: introduced
- Book destination: Chapter 3 sections on logits, softmax, and negative
  log-likelihood
- Related evidence: planned manual derivation and PyTorch verification
- Related production tasks: `ANIM-CE-001`; candidates `CAND-ANIM-001` and
  user-approved `CAND-ANIM-002`

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

## Evidence state

- `demonstrated`: equal logits produce equal probabilities.
- `introduced`: probabilities are normalized competition over the current model
  vocabulary.
- `not yet demonstrated`: unequal logit ratios, temperature, negative-log
  surprise, sequence likelihood, cross-entropy, and perplexity.

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
