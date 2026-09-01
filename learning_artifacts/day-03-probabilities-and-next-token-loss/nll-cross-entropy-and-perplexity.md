# NLL, Cross-Entropy, and Perplexity

- Day: 03
- Date opened: 2026-09-01
- Status: introduced
- Book destination: Chapter 3 sections on token loss, sequence likelihood,
  cross-entropy, and perplexity
- Related evidence: planned manual derivation and PyTorch verification
- Related production tasks: `ANIM-CE-001`, `ANIM-LOGLOSS-001`

## Questions that drive the discussion

- In what sense are negative log-likelihood and cross-entropy the same token
  loss, and in what sense are they different concepts?
- What roles do the target distribution $q$ and model distribution $p$ play?
- Why does one-hot cross-entropy reduce to `-log p_target`?
- How do valid token losses become a sequence or batch training loss?
- What does perplexity mean, and when is comparing perplexity misleading?

## Mechanism introduced

Cross-entropy measures expected surprise when outcomes follow $q$ but code
lengths or probabilities come from $p$:

\[
H(q,p)=-\sum_i q_i\log p_i.
\]

For an observed next-token target $y$, ordinary language-model supervision uses
a one-hot $q$. All terms except the target vanish:

\[
H(q,p)=-\log p_y.
\]

Thus token-level one-hot cross-entropy and the NLL of the observed token are
numerically identical. NLL emphasizes the likelihood of the observed sample;
cross-entropy emphasizes the relationship between target and model
distributions. A training framework commonly uses `cross_entropy` to compute a
sample estimate of expected NLL.

Across valid target positions, a common reduced loss is:

\[
L_{mean}=\frac{\sum_t m_t\left(-\log p_t(y_t)\right)}{\sum_t m_t},
\]

where $m_t$ is 1 for an included target and 0 for padding or another ignored
position. The summed sequence NLL and mean token cross-entropy answer different
reporting questions and should not be conflated.

The deeper proper-scoring identity is:

\[
H(q,p)=H(q)+D_{KL}(q\|p).
\]

Because $H(q)$ is fixed and KL divergence is nonnegative, expected
cross-entropy is minimized at $p=q$. This result is introduced but still needs
derivation and a controlled verification.

Perplexity will later be defined as the exponential of mean natural-log loss:

\[
\operatorname{PPL}=e^{L_{mean}}.
\]

Its interpretation as an effective branching factor is useful only under a
fixed tokenizer, target policy, and evaluation distribution.

## Evidence state

- `introduced`: one-hot cross-entropy equals observed-token NLL.
- `introduced`: valid token NLLs can be summed or averaged with an explicit mask
  denominator.
- `demonstrated`: the learner correctly reasoned that when human-language
  continuations are genuinely uncertain, a perfect model with $p=q$ still has
  cross-entropy $H(q)>0$; matching removes model mismatch, not intrinsic data
  entropy.
- `not yet demonstrated`: the proper-scoring decomposition, perplexity
  interpretation, causal target alignment, and manual/PyTorch agreement.

## Learner explanation and boundary refinement

The learner connected token-level one-hot cross-entropy to observed-token NLL and
concluded that a perfect model need not have zero loss because human language is
uncertain. The precise population statement is:

\[
p=q \quad\Longrightarrow\quad H(q,p)=H(q),
\]

which is positive whenever the true conditional distribution has multiple
outcomes with nonzero probability.

Zero cross-entropy remains mathematically possible when the true conditional
distribution is deterministic and the model assigns probability one to its sole
outcome. A sufficiently flexible model can also drive empirical loss on a finite
memorized training sample near zero without eliminating population uncertainty.
This separates population cross-entropy from NLL measured on one finite dataset.

## Animation opportunity check

The explicit mathematics automatically triggers animation review. No new
candidate is needed: user-approved `CAND-ANIM-002` and task `ANIM-CE-001` already
cover target probability → NLL → one-hot cross-entropy identity → masked mean.
`ANIM-LOGLOSS-001` covers the proper-scoring decomposition conceptually. All
production remains on the Mac Studio after verification.

## Open edges

- Derive the one-hot reduction slowly from the full cross-entropy sum.
- Distinguish empirical one-hot samples from a population target distribution.
- Verify `H(q,p)=H(q)+KL(q||p)` numerically.
- Explain why perplexities from different tokenizers are not directly comparable.
- Connect token targets to causal shifting and ignored positions.
