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
positions. Its state is `discuss`; production requires explicit learner approval.

