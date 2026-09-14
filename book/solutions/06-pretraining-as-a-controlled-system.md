# Chapter 6 — Worked Solutions

Read the [chapter](../chapters/06-pretraining-as-a-controlled-system.md) first.
The [three Day 8 notebooks](../../notebooks/day-08/README.md) contain adjacent,
runnable reference solutions; this guide explains the conceptual exercises.
Executing a reference is not evidence of independent mastery.

## 1. Document splits and contamination

Two different document IDs can contain identical text, overlapping passages,
translations or paraphrases. A document-level split prevents one document's
windows from being randomly scattered across splits, but does not establish
independence of the content. The fixture detects normalized exact overlap only.
A real corpus needs a declared deduplication and contamination policy, with its
limitations recorded. A low held-out loss is less convincing if the held-out
content has effectively been seen during training.

## 2. Shorter windows with the same targets

Our policy predicts each byte and EOS once regardless of window length. But a
target just after a new window boundary cannot use the previous window's hidden
states or tokens. It is a prediction from a shorter prefix. Padding overhead
can also change. Therefore equal supervised target counts do not mean equal
context, equal compute or equal difficulty. Notebook 1 changes lengths 8, 16 and
32 while checking that the target count stays fixed.

## 3. EOS, loss masking and attention masking

EOS is an ordinary learned vocabulary entry with a boundary convention. The
network can attend across it unless the attention mechanism forbids that access.
A loss mask says “do not score this prediction,” not “hide this source.” An
attention boundary removes information-flow edges. For packed independent
documents, a block-diagonal causal attention mask can enforce isolation; an EOS
alone cannot. In the fixture, documents occupy separate windows and right-padding
sources are in the future of every valid prefix position.

## 4. What the budget formula counts

$SbTAR$ counts processed tensor positions under fixed full microbatches and
accumulation windows. It equals valid target presentations only if every
position contributes to loss. With ignored labels, it is an upper bound on those
targets. It never measures unique information: rereading the same corpus adds
presentations. Keep counters for updates, processed positions, valid targets and
unique source content. The teaching recipe's 768-position ceiling must be
reconciled with its actually consumed labels.

## 5. The short-microbatch contribution

Let $\overline\ell_j$ be microbatch $j$'s mean over $n_j$ valid targets. The
correct global mean is

$$
L=\sum_j\frac{n_j}{\sum_r n_r}\overline\ell_j.
$$

The gradient has the same weights. For one target and sixteen targets, the
weights are $1/17$ and $16/17$. Taking half of each mean overweights the single
target sixteenfold relative to each target in the larger microbatch. Backward
each summed microbatch loss divided by the total valid count, with parameters
held fixed until all contributions are accumulated. Notebook 1 verifies the
whole parameter gradient, not merely the reported scalar loss.

## 6. AdamW and the newest gradient

The first moment is a weighted history of signed gradients. A small new negative
gradient need not outweigh several earlier positive ones. AdamW can therefore
continue moving in the negative parameter direction even though plain SGD on
the newest gradient would move positively. Adaptive scaling and decoupled decay
also distinguish its update from raw SGD. This is a mechanism, not a guarantee
that momentum makes a better decision on each step. Notebook 2 exposes the
recurrence and a positive, positive, slightly negative gradient sequence.

## 7. Doubling accumulation

If microbatch size, length and valid fraction stay fixed, doubling accumulation
doubles targets per optimizer update. A schedule that peaks after three updates
now peaks after twice as many targets. It also changes the gradient estimate and
the frequency of weight updates per token. It does not automatically double
activation memory because microbatches can be processed sequentially. State
whether an experiment fixes updates, token exposure or another resource; do not
claim that only one harmless implementation detail changed.

## 8. Clipping placement and nonfinite values

Clipping is nonlinear. For scalar contributions 10 and -9 with limit 1, summing
then clipping gives 1; clipping separately then summing gives 0. Accumulate
first, unscale if using loss scaling, check finiteness, then clip the global
gradient once. NaN or infinite gradients are not trustworthy directions that
can be repaired by choosing a smaller norm. Reject or explicitly handle the
failed update. Finally, a bound on the incoming gradient is not a direct bound
on AdamW's transformed update plus weight decay.

## 9. BF16 range versus detail

Exponent bits determine a large part of representable range; fraction bits
determine relative resolution within that range. BF16 has a wider exponent
range than FP16 but fewer fraction bits. That is why 65536 remains finite in
BF16 while overflowing FP16, yet a small increment near 1 may survive FP16 and
round away in BF16. Neither property alone establishes the stability or accuracy
of a training computation. The notebook's cast demonstration does not execute
mixed-precision CUDA training.

## 10. Memory is more than persistent parameter state

The $16P$ estimate includes FP32 weights, gradients and two Adam moments. It
omits activation storage, attention intermediates, optimizer temporaries,
allocator overhead and framework/process memory. Actual implementation choices
can also change the ledger. On Spark, CPU and GPU activity share a constrained
memory environment; a tensor-byte estimate is not a measured host reserve.
Profile the exact candidate, record peak allocations and minimum host available
memory, and honor the reserve before authorizing a longer run.

## 11. Validation grouping invariance

Evaluate the same frozen model and fixed target set with several batch sizes,
including unequal tails. Accumulate summed NLL and valid-label counts, then
divide once. Compare the final means within a declared floating-point tolerance.
Also check that parameters and gradients were not updated and the prior model
mode is restored. Notebook 3 uses float64 to make a strict small-fixture
comparison easy; production tolerances should match the tested dtype and backend.

## 12. Separate recovery controls

First save all state at an update boundary. Continue the original run and keep
its subsequent batch IDs, losses and final parameters. Load the checkpoint in
fresh sessions:

- Complete restoration should match the reference under the local replay contract.
- Omit optimizer state but restore the cursor: batch IDs should still match,
  while moment-dependent updates and eventually parameters differ.
- Restore optimizer state but omit the cursor: subsequent batch IDs differ,
  changing gradients even though the saved model weights were identical.

Finite loss and successful process exit can occur in all three branches. The
discriminating evidence is the continuation trajectory and data identity, not
whether the file loaded. Exact CPU replay in this fixture does not establish
cross-device or distributed reproducibility.

## 13. Correct masking and inefficient computation

The loss mask removes ignored positions from the objective, not necessarily
from embedding, attention, MLP and output-projection computations. This
implementation processes fixed1024-position windows even when stories are
short. Its48.84M valid targets occupy about21.29% of229.38M processed positions.
That is a measured utilization fraction, not a guaranteed speedup factor.

A comparison must preserve intended supervision, per-document causal access,
tokenization and reduction, and disclose changes to ordering or available
context. An EOS separator alone does not forbid cross-document attention.
Fixing updates while changing targets per update also changes exposure.

## 14. Better likelihood, inconsistent characters

Teacher-forced evaluation conditions on recorded prefixes. Free generation
conditions on the model's own preceding choices. A mistaken speaker attribution
can become part of the subsequent history. Separately, average loss improvement
can be driven by frequent syntax while character tracking remains unreliable.
Neither explanation requires a faulty causal mask.

The objective can reward long-range consistency when it helps predict observed
tokens; it does not prohibit learning it. Our evidence establishes improved
fixed-development likelihood and remaining behavioral failures, not the unique
mechanism responsible for each failure.

## 15. Repetition and the overfitting hypothesis

Retain the exact prompt, checkpoint, decoding mode, temperature, seed and length
limit. Reproduce the symptom, then distinguish decoding from training effects.
Evaluate fixed training and held-out samples under matched frozen-checkpoint
settings, preferably across several checkpoints. Audit relevant content when
making contamination or memorization claims.

Our development loss continued decreasing. That does not support a claim of
development-loss deterioration here, or prove absence of memorization. The
last500 online losses came from changing weights and are averaged by batch;
final development NLL uses fixed weights and valid-target weighting. Their
difference is not a clean generalization gap. The defensible current label is
repetitive/inconsistent generation with unresolved cause.

## 16. Seeds, greedy decoding and length limits

Greedy decoding selects a maximum-scoring token; there is no random draw for
the seed to change. With positive-temperature sampling and identical inputs
and runtime, a fixed seed repeats the tested trajectory. Raising the length
limit usually permits more steps without changing its initial segment. These
are expected behaviors, not evidence that controls are necessarily broken.

Change one relevant variable at a time and verify submitted settings. Backend
probes alone do not validate browser event handling. Temperature modifies
selection probabilities, not learned parameters.

## 17. EOS and narrative closure

EOS establishes that the model selected termination before the cap in that
generation. It does not establish consistent characters, resolved causality,
factuality or natural plot closure. Our archived final greedy first-prompt
sample emits EOS but contains unexplained speaker-role changes. Report
termination and narrative quality separately.

## 18. An honest next-experiment proposal

A systems proposal is length-aware batching: hypothesize higher useful-target
throughput while preserving the intended prediction problem. Declare baseline,
exposure, model, dtype, safety cap and ordering policy. Measure wall time,
throughput, memory and evaluation invariants. Failed equivalence checks block
interpreting a speed change as a pure implementation benefit.

A behavior proposal is more training under a frozen evaluation contract.
Specify whether this means restarting with a longer learning-rate schedule
or extending the existing checkpoint; those are different recipes. Keep prompts
and decoding fixed, assess entity/event continuity, and allow little or no gain.
Do not pick the nicest sampling seed.

Neither trained comparison has been performed. Short batch-size profiles and
sampling probes answer narrower questions. A proposal is not outcome evidence
or approval to launch it.

## Integration: a convincing Days 8–9 defense

Explain the path from a source document to a recoverable update using the
[bounded specification](../../experiments/specs/2026-09-09-day8-bounded-pretraining.md).
Identify which claims are measured by the CPU fixtures, which are analytical
estimates, and which are now supported by the
[completed Spark run](../../experiments/reports/2026-09-14-tinystories-learning-result.md).
Use the [evidence-reading lab](../labs/06-reading-a-pretraining-run.md) to
reproduce the accounting and inspect complete archived samples.
If a run has decreasing loss but
leaked validation, an incorrect loss denominator or violated memory reserve,
do not call the whole experiment successful. Completion is a conjunction of
its stated criteria, not one attractive curve.
