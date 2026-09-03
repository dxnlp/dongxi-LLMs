# NLL, Cross-Entropy, and Perplexity

- Day: 03
- Date opened: 2026-09-01
- Status: introduced
- Book destination: Chapter 3 sections on token loss, sequence likelihood,
  cross-entropy, and perplexity
- Related evidence: planned manual derivation and PyTorch verification
- Related production tasks: `ANIM-CE-001`, `ANIM-NTP-001`,
  `ANIM-LOGLOSS-001`; automatic candidate `CAND-ANIM-005`

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

Equal sequence NLL also does not imply equal token-level behavior. For four
targets, model A can assign probability $0.5$ at every step, giving product
$0.5^4=0.0625$. Model B can assign probabilities approximately
$[0.99,0.99,0.99,0.0644]$, whose product is also $0.0625$. Their total NLLs are
therefore equal, but A spreads moderate uncertainty across the sequence while B
concentrates nearly all surprise in one catastrophic, overconfident miss.

The scalar sequence score deliberately forgets this distribution over
positions. Generation trajectories, calibration, tail risk, and optimization
can differ: B's gradient is concentrated at the catastrophic position, and one
bad choice there can redirect every later generated context. Token-level loss
traces, quantiles, maximum loss, calibration checks, and free-running evaluation
answer questions that the mean alone cannot. Under the narrow claim "likelihood
of this exact sequence," however, equal products really are equal; the stronger
behavioral distinction requires these additional criteria.

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

## Exact logit-gradient derivation

For one position, let $z\in\mathbb{R}^V$ be the logits, let

\[
S=\sum_j e^{z_j},
\qquad
p_k=\frac{e^{z_k}}{S},
\]

and let the target distribution $q$ satisfy $\sum_k q_k=1$. Cross-entropy is

\[
L=-\sum_k q_k\log p_k.
\]

Since

\[
\log p_k=z_k-\log S,
\]

the loss can be rewritten as

\[
\begin{aligned}
L
&=-\sum_k q_k(z_k-\log S)\\
&=-\sum_k q_kz_k+\left(\sum_kq_k\right)\log S\\
&=-\sum_k q_kz_k+\log S.
\end{aligned}
\]

Differentiate with respect to one logit $z_i$:

\[
\begin{aligned}
\frac{\partial L}{\partial z_i}
&=-q_i+\frac{1}{S}\frac{\partial S}{\partial z_i}\\
&=-q_i+\frac{e^{z_i}}{S}\\
&=p_i-q_i.
\end{aligned}
\]

The same result follows through the full softmax Jacobian,

\[
\frac{\partial p_k}{\partial z_i}
=p_k(\delta_{ki}-p_i),
\]

and the chain rule:

\[
\begin{aligned}
\frac{\partial L}{\partial z_i}
&=\sum_k
\frac{\partial L}{\partial p_k}
\frac{\partial p_k}{\partial z_i}\\
&=\sum_k\left(-\frac{q_k}{p_k}\right)
p_k(\delta_{ki}-p_i)\\
&=-q_i+p_i\sum_kq_k\\
&=p_i-q_i.
\end{aligned}
\]

For a one-hot target $y$:

\[
\frac{\partial L}{\partial z_y}=p_y-1<0
\quad\text{when }p_y<1,
\]

while every non-target $i\ne y$ has

\[
\frac{\partial L}{\partial z_i}=p_i>0.
\]

A direct illustrative gradient-descent step on logits would be

\[
z_y\leftarrow z_y+\eta(1-p_y),
\qquad
z_i\leftarrow z_i-\eta p_i\quad(i\ne y).
\]

Thus the target score moves up and every non-target score moves down; a
high-probability wrong candidate receives the largest downward correction. The
gradient components sum to zero,

\[
\sum_i(p_i-q_i)=1-1=0,
\]

which is the differential expression of softmax's shared-logit-shift invariance.

In a real model, logits are intermediate activations rather than independent
optimizer parameters. Backpropagation routes this logit gradient through the
output head, transformer, and embeddings; an optimizer updates those parameters.
Therefore the direct logit update above explains local direction, not the exact
post-update logits of the complete network.

## From logit correction to model learning

Let the final hidden state be $h\in\mathbb{R}^D$ and the output head be
$W\in\mathbb{R}^{V\times D}$ with bias $b\in\mathbb{R}^V$:

\[
z=Wh+b.
\]

Writing $g=p-q=\partial L/\partial z$, the chain rule gives

\[
\frac{\partial L}{\partial b}=g,
\qquad
\frac{\partial L}{\partial W}=g h^\top,
\qquad
\frac{\partial L}{\partial h}=W^\top g.
\]

Each vocabulary row $W_i$ acts as a learned detector for candidate token $i$.
The outer product $g h^\top$ strengthens the target row in the direction of the
current contextual state and weakens competing rows in proportion to their
predicted probabilities. Meanwhile, $W^\top g$ sends a single blended error
signal back into the transformer, asking it to produce a more discriminating
context representation next time. Backpropagation then distributes that signal
through all operations that created $h$.

This is credit assignment, not direct storage of a correction for one logit:
the training example changes reusable parameters, so it can alter predictions
in many other contexts as well.

## How one-hot examples learn a distribution

Suppose indistinguishable instances of the same context have an empirical
next-token distribution $r$: for example, `dog` occurs with frequency $0.7$ and
`cat` with frequency $0.3$. Every individual training row still supplies a
one-hot target $q$, but its expected value is the frequency distribution:

\[
\mathbb{E}[q]=r.
\]

Because the per-example logit gradient is $p-q$, the expected gradient for that
context is

\[
\mathbb{E}[p-q]=p-\mathbb{E}[q]=p-r.
\]

The competing one-hot updates therefore balance when $p=r$. A `dog` example
temporarily pushes probability toward `dog`, and a `cat` example pushes it toward
`cat`; across representative repetitions, their frequencies determine the
equilibrium. This is how one-hot supervision can learn a non-one-hot conditional
distribution without any single row explicitly listing all valid alternatives.

## Evidence state

- `introduced`: one-hot cross-entropy equals observed-token NLL.
- `introduced`: valid token NLLs can be summed or averaged with an explicit mask
  denominator.
- `analytically demonstrated`: substituting softmax into cross-entropy and
  differentiating yields `dL/dz_i = p_i-q_i`; its signs and zero-sum property
  explain target/non-target correction and shared-shift invariance.
- `introduced`: the chain-rule path from the logit gradient through the output
  head into its parameters and the contextual hidden state.
- `introduced`: repeated one-hot outcomes have expected gradient $p-r$, so their
  equilibrium prediction matches the empirical conditional frequencies $r$.
- `introduced`: equal sequence NLL can hide radically different distributions
  of token-level surprise, confidence, gradient concentration, and generation
  risk; the scalar is equal under its narrow contract but not a full behavioral
  equivalence test.
- `demonstrated`: the learner correctly reasoned that when human-language
  continuations are genuinely uncertain, a perfect model with $p=q$ still has
  cross-entropy $H(q)>0$; matching removes model mismatch, not intrinsic data
  entropy.
- `demonstrated`: the learner correctly bounded a lower-perplexity result to
  similarity with or prediction of the tested corpus and rejected the stronger
  conclusion that the lower-perplexity model is generally better.
- `not yet demonstrated`: the proper-scoring decomposition, executable gradient
  agreement, empirical-frequency convergence, perplexity interpretation, causal
  target alignment, and full manual/PyTorch agreement.

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

When comparing two models, the direct supported statement is narrower than
“shares similar patterns”: under the same tokenizer, preprocessing, causal target
alignment, loss mask, reduction, and test corpus, the lower-perplexity model
assigned higher geometric-mean probability to the observed next tokens. Pattern
similarity, training-data overlap, better generalization, or contamination are
possible explanations that require additional evidence. The result alone does
not establish broader capability, reasoning, factuality, calibration, preferred
generation behavior, safety, or performance on another distribution.

## Animation opportunity check

The explicit mathematics automatically triggers animation review. No new
candidate is needed: user-approved `CAND-ANIM-002` and task `ANIM-CE-001` already
cover target probability → NLL → one-hot cross-entropy identity → masked mean.
`ANIM-NTP-001` already covers the now-complete analytical `p-q` derivation and
its gradient-descent directions. `ANIM-LOGLOSS-001` covers the proper-scoring
decomposition conceptually. All production remains on the Mac Studio after
verification.

The introduction of `PPL=exp(mean NLL)` triggered `CAND-ANIM-005`. It remains in
`discuss`: after the derivation, the learner can decide whether it should extend
`ANIM-CE-001` or become a separate Mac Studio short.

## Open edges

- Derive the one-hot reduction slowly from the full cross-entropy sum.
- Distinguish empirical one-hot samples from a population target distribution.
- Verify `H(q,p)=H(q)+KL(q||p)` numerically.
- Explain why perplexities from different tokenizers are not directly comparable.
- Connect token targets to causal shifting and ignored positions.
