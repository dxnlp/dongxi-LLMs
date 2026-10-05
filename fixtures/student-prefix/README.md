# Original student prefixes and teacher context

Nine original two-symbol reversal items have six training source groups and
three held-out source groups. The model reads a four-token symbolic question,
not the English documentation. Actual TinyDecoder student sampling uses all
eight output IDs, with natural EOS and a three-action cap.

The teacher is an authored finite policy, not a pretrained or neural teacher.
Its separately recorded answer hint is supplied only to training teacher queries.
Students never receive it, and evaluation generation never invokes that teacher.
The independent scorer may read references to grade generated outputs.

The tail and zero-support objects are separately labeled exact finite distribution
controls, not model generations. All losses, prefix weighting and sample coordinates
are frozen in the companion specification before collection or fitting.

The historical campaign/source snapshots are retained separately from the
[predeclared API-hardening follow-up](../../experiments/specs/2026-10-04-student-prefix-hardening.md).
Conditional tail-detail decomposition accepts strictly positive full distributions;
the separate exact full-KL function retains zero-support infinity semantics.
Training pools must contain one frozen campaign/phase/update/arm/checkpoint
cohort, never a mixture of individually valid sampling records.
