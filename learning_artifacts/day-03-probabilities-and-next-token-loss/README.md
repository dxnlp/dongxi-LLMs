# Day 3 — Probabilities and Next-Token Loss

- Date opened: 2026-09-01
- Day status: in progress
- Book destination: planned Chapter 3, `Learning the Next Token`
- Required build: manual numerical derivation, PyTorch verification, and a tiny
  next-token model

## Questions and expected outcome

Day 3 asks how a contextual vector becomes a trainable prediction:

- What does a next-token probability mean—and what does it not mean?
- Why does softmax treat logits as relative evidence among competing tokens?
- Why is negative log-probability a natural measure of predictive surprise?
- How do token losses combine into cross-entropy and perplexity?
- How are input positions aligned with their next-token targets under causality?
- How do padding and other ignored positions affect the reduced loss?
- Can a tiny model visibly learn a next-token distribution from text?

The day is complete only when the derivation, transparent implementation,
precommitted experiment, evidence report, and coherent Chapter 3 contribution are
present. Conversational understanding alone is not sufficient.

## Topic artifacts

1. [`probability-as-competition-and-surprise.md`](probability-as-competition-and-surprise.md)

Create additional focused topics as the lesson reaches cross-entropy,
perplexity, causal shifting, and the tiny-model experiment. Do not turn this
index into a transcript.

## Starting knowledge from Day 2

The learner already distinguishes logits from probabilities, explained why equal
logits produce equal softmax probabilities, and understands that one token's
probability depends on its competitors. The unequal-logit mechanism, sequence
likelihood, negative log, perplexity, and causal target alignment remain Day 3
work.

The preferred teaching mode is profound mechanism-level discussion with concrete
examples. Arithmetic supports derivation and executable verification; it should
not become a stream of calculation quizzes.

## Animation opportunity

`visuals/animations/PROPOSALS.md` already contains `CAND-ANIM-001`: input tokens
to causal logits and probabilities while next-token targets align with preceding
positions. Its state remains `discuss`.

The learner approved `CAND-ANIM-002`, an LLM-context animation showing target
probability → per-token NLL → one-hot cross-entropy identity → masked aggregation
across valid positions. It has been promoted into the expanded `ANIM-CE-001`
packet. Production and rendering belong exclusively on the Mac Studio after the
Day 3 derivation and PyTorch verification are committed.

The learner also approved `CAND-ANIM-003`, showing standard next-token training
from `p` and one-hot `q` through `p-q`, the corresponding gradient-descent logit
directions, and the emergence of a distribution across repeated diverse
examples. It is promoted into `ANIM-NTP-001`; Mac Studio production waits for the
verified Day 3 gradient and target-frequency experiment.

The learner approved all three conceptual reasons for negative log loss as an
animation package: sequence products become additive token surprise; confident
softmax errors retain a strong correction; and expected log loss rewards matching
the full data distribution. `CAND-ANIM-004` is promoted into
`ANIM-LOGLOSS-001`; all production remains on the Mac Studio after verification.
