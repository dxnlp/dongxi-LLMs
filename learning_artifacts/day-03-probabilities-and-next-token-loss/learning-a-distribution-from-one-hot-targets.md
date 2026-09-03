# Learning a Distribution from One-Hot Targets

- Day: 03
- Date opened: 2026-09-03
- Status: demonstrated analytically and in code
- Book destination: Chapter 3 section on empirical next-token distributions
- Experiment specification:
  `experiments/specs/2026-09-03-next-token-distribution.yaml`
- Evidence report:
  `experiments/reports/2026-09-03-next-token-distribution.md`

## The apparent paradox

Every ordinary next-token example names one observed target. Nevertheless, a
model can learn that several continuations are valid with different
probabilities. The distribution is not contained in one label; it emerges from
the population of examples and the parameters they share.

For a fixed context with empirical target distribution $r$, one example has
one-hot target $q$ and logit gradient

\[
\frac{\partial L}{\partial z}=p-q.
\]

Taking the expectation over sampled targets gives

\[
\mathbb{E}_q[p-q]=p-r.
\]

The expected update is stationary when $p=r$. Individual examples pull toward
their observed targets; their long-run frequencies determine where those pulls
balance.

## What the controlled experiment showed

The precommitted experiment used one abstract context, two logits initialized to
zero, and a full batch containing 70 class-0 and 30 class-1 targets. SGD changed
the prediction from `[0.5,0.5]` to approximately `[0.7,0.3]`. Mean
cross-entropy fell from `0.69314730` to `0.61086428`, matching the empirical
entropy `0.61086434`. PyTorch autograd agreed with `p-r` to maximum recorded
absolute error `5.96e-08`.

At equilibrium, the probability ratio determines the logit gap:

\[
\frac{p_0}{p_1}=e^{z_0-z_1}
\quad\Longrightarrow\quad
z_0-z_1=\log\frac{0.7}{0.3}\approx0.8472979.
\]

The observed final gap was `0.84729773`. The absolute logit heights remain
arbitrary because a shared shift changes neither softmax probabilities nor loss.

## Interpretation boundary

This verifies distribution learning in a transparent empirical system. It does
not show that a transformer recovers the true human-language distribution. Real
contexts are rarely repeated exactly; generalization requires shared learned
representations, representative data, sufficient capacity, and held-out
evidence. The experiment demonstrates the optimization mechanism, not semantic
understanding or deployment calibration.

## Animation connection

The verified checkpoints and fixed 70/30 target stream supply evidence for
`ANIM-NTP-001`: one-hot targets alternate, `p-q` gradients pull in competing
directions, and the probability bars settle where expected gradient vanishes.
Production remains assigned to the Mac Studio.
