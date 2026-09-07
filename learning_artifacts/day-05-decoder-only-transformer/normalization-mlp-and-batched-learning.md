# Normalized branch inputs, local feature processing, and batched learning

- Date: 2026-09-07.
- Source: live Day 5 discussions through notebooks 04, 05, and 07.
- Canonical book placement: Chapter 5 foundation, sections 5.5–5.8.
- Evidence level: explanations offered and acknowledged; no independent
  explanation-back or completion of all learner notebook exercises established.

## LayerNorm: make the numerical operation tangible

The learner requested a more understandable explanation after the first formula.
Use one token's [10,20,30,40] features: subtract mean 25, then divide by feature
standard deviation to obtain approximately [-1.34,-.45,.45,1.34]. Compare with
[100,200,300,400]. Explain relative feature differences before introducing gamma
and beta. Statistics are per-token over D; learned affine parameters are shared
across positions. Epsilon makes scale invariance approximate and normalized
variance slightly below one; constant vectors become beta after the affine step.

Do not say normalization preserves all information, that the learned affine
parameters recover each token's original magnitude, or that post-affine output
must have zero mean/unit variance. In pre-norm, LN prepares the branch's reading
while the residual bypass keeps the incoming X. In post-norm, LN acts on the sum.
The full-time normalization counterexample shows that attention masking alone
does not guarantee an entirely causal model. Notebook 04 supplies the control.

## Positionwise MLP: local does not mean context-free

Connect attention's gathering of information between positions to an MLP's
transformation of available features within each position. The same parameters
act at every position, with different contextual input vectors. Expand 16 to 32
features, apply GELU, and project back to 16; no token positions are added.
Expansion is learned recombination, not literal duplication. Without activation,
two affine projections collapse into one; show the row-vector/PyTorch transpose
conventions explicitly. Notebook 05 separates an MLP-only position intervention
from the whole decoder's cross-position dependencies.

## Token-wise losses versus optimizer-step granularity

The learner asked whether learning is token by token. Clarified that the model
predicts at all legal input positions with a single parameter snapshot, computes
per-position cross-entropy, averages over supervised positions, then backpropagates
and updates shared parameters once. Causal masking permits parallel training
without future leakage. Teacher-forced inputs are supplied observations; free
generation instead appends selected predictions, usually without weight updates.

Notebook 07 has two six-position input sequences, labels shifted once, logits
[2,6,16], and 12 supervised targets. Use the existing seed-505, 160-step report
for numerical claims rather than inferring the learner's results. The batch's
unique token/successor associations can be memorized without rich contextual
reasoning; perfect training accuracy does not establish useful specialization
of every head or generalization. Reversed contexts have no declared gold labels.

## Synthesis and portable reuse

The learner requested creating the Day 5 chapter content now, while retaining
Days 6–7 as later sections of the same Chapter 5. Produced the foundation with
seven notebook links, existing static figures, implementation connections,
bounded evidence, twelve conceptual exercises, and worked answers. Do not
describe this as the full modern architecture chapter or a completed learning day.

Animation opportunity check reuses CAND-ANIM-014 (normalization), 015 (MLP),
018 (learning-rate scale), ANIM-CE-001 and ANIM-NTP-001 (position losses and
shared updates), and ANIM-EMB-001 (end-to-end parameter paths). Add the batch
aggregation sequence to their existing canonical material rather than launching
a new production task. All rendering remains Mac Studio, approval-gated.

## Synthesis verification

Executed the chapter's exact Python example in the existing Torch environment
with `PYTHONPATH=src`: final loss 0.0007856183219701052, accuracy 1.0, logits
[2,6,16], matching the existing fixed experiment. Separately checked the rounded
LayerNorm example and the scalar 1-versus-2 head-output gradients. Checked 92
local Markdown/figure targets before the final index update. All 47 repository
tests passed in 3.572 seconds; `git diff --check` passed. This is a repeat of
the teaching contract, not new empirical evidence about model capability.
Notebook files and the active server were not altered by this writing task.
