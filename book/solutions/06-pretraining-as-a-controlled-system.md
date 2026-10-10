# Chapter 6 — Worked Solutions

Read the [chapter](../chapters/06-pretraining-as-a-controlled-system.md) first.
The [three Day 8 notebooks](../../notebooks/day-08/README.md) contain adjacent,
runnable reference solutions; this guide explains the conceptual exercises.
Executing a reference does not establish independent reasoning about the mechanism.

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

$SbnAR$ counts processed tensor positions under fixed full microbatches and
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
memory, and honor the reserve before selecting a longer run.

## 11. Development grouping invariance

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
implementation processes fixed 1024-position windows even when stories are
short. Its 48.84M valid targets occupy about 21.29% of 229.38M processed positions.
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
last 500 online losses came from changing weights and are averaged by batch;
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

The length-aware systems intervention remains unrun. The separately declared
fresh control/half-learning-rate comparison has reached its first 400 updates,
not its 14,000-update horizon. It preserves the original schedule, seed and
target exposure; [the measured report](../../experiments/reports/2026-10-05-native-story-first400-comparison.md)
does not establish final storytelling competence. Short batch-size profiles
and sampling probes still answer different questions.

## 19. Do not change the objective to spend the remaining budget

The cap is a ceiling, not a promise to spend every target. After 9 targets under
cap 16, the next complete 8-target update cannot fit into the remaining 7. Removing
one label or a microbatch changes the training objective. Refuse the whole update
before forward/backward, LR mutation or optimizer mutation, and restore collection's
shuffle permutation, cursor, epoch and RNG. Keep completed weights, moments,
gradients and counters unchanged. Resume must restore the cumulative allowance,
not start it at zero. The [CPU report](../../experiments/reports/2026-10-05-story-valid-target-budget.md)
tests this boundary, including an epoch crossing and exact same-cap continuation.
It does not test recovery from a partly executed optimizer update.

## Evidence-reading extensions

Exercises 19–21 retain their original numbers because they test conceptual
budget, replay and coverage boundaries. Operational receipts are documented in
[Appendix D](../appendices/d-reproduction-and-environments.md#budget-boundaries-and-spent-work).

## 20. Numerical replay does not refund failed work

Completed target exposure describes successful optimizer boundaries:9 plus 8
is 17. The persistent allowance describes all admitted attempts:9 for the first
update,8 for the failed operation and 8 for its retry is 25. Restoring an older
checkpoint restores weights, moments, stream/RNG and successful counters, not
the journal's later reservations. The independently retained receipt binds the
old prefix to the same physical journal; a new output directory cannot reset it.

Check every declared dimension before the next complete model operation. If a
cap would be crossed, make no forward/backward/optimizer call and restore the
planning step's stream state. Do not trim labels or change the loss denominator.
If computation has already started and AdamW may have changed only some weights,
poison that session: it cannot update, observe or publish. Construct a fresh
session and restore a previously durable completed boundary. Keep the failed
ticket even when the later numerical trajectory replays exactly.

Development, uncached generation and activation probes are separate work. A
successful replay is not proof that their repeated calls are free, that all
reserved work actually completed, or that a physical resource ceiling was
enforced. Source and evidence boundaries are frozen in the
[story work specification](../../experiments/specs/2026-10-05-story-persistent-work.md).
The [actual native companion](../../experiments/reports/2026-10-05-story-work-lesson.md)
also checks cap 24: only 7 places remain, so retrying the whole 8 refuses with 9
successful targets/17 reserved places, zero loss calls and unchanged state.

## 21. The requested boundary is not the schedule horizon

Success depends on the declared boundary and other acceptance conditions. A
predeclared first 400 tranche can complete successfully while its 14,000-update
schedule remains unfinished. The
[actual control receipt](../../experiments/reports/native-story-control-pilot-20261005-01/acceptance.json)
and [half-rate receipt](../../experiments/reports/native-story-half-lr-pilot-20261005-01/acceptance.json)
each support 400 updates, exit zero and their tested source/resource envelope.
Both present 1,389,548 valid training targets in 6,553,600 training positions.
Those receipts alone do not supply any story-quality score.

Shortening the schedule horizon changes decay at those same updates. The result
would no longer test only the declared proportional learning-rate intervention
under the original horizon. Label it as another recipe rather than hiding the
change behind the same update count.

The [measured publication comparison](../../experiments/reports/2026-10-05-native-story-first400-comparison.md)
now includes 192 actual continuations from four native children and fixed slice
NLL. Control development NLL is 3.388772 versus 3.652813 for half-rate on the
same 64 windows/13,132 valid labels; the separate training-source selection has
64 windows/13,285 labels. This supports lower observed-target loss on that
development slice, not a whole-corpus claim or an overfitting diagnosis.

Stopping is a separate observation: control reaches EOS in 45/48 continuations,
half-rate in 25/48. Yet their two-AI-reader ending means are 0.020833 and 0 on
the rubric's 0–2 scale. A completed token stream need not complete the plot.
The initial random checkpoints score 1.5 on freedom from repetition and zero on
the other four dimensions: lack of loops does not imply coherence either.
Keep all five axes instead of declaring a winner from one number.

Both readers scored every actual candidate, retaining 56 disagreements under
the declared mean policy. The 800-draw, seed-1010 paired intervals resample
twelve openings with their four recipes together, not 48 independent stories.
These AI judgments and small-panel intervals do not establish human agreement,
new-opening performance or between-training-seed uncertainty. Publication also
uses automatic SDPA rather than the deterministic MATH-only training/recovery
entry, so this is not a cross-backend bitwise equivalence test.

The 288 later cells remain missing, not poor-quality zeros. Neither accepted
tranche establishes 14,000-update behavior or final convergence. Keep the
September completed run separate from this fresh intervention.

## Integration: a convincing Days 8–9 defense

Explain the path from a source document to a recoverable update using the
[bounded specification](../../experiments/specs/2026-09-09-day8-bounded-pretraining.md).
Identify which claims are measured by the CPU fixtures, which are analytical
estimates, and which are now supported by the
[completed Spark run](../../experiments/reports/2026-09-14-tinystories-learning-result.md).
Use the [evidence-reading lab](../labs/06-reading-a-pretraining-run.md) to
reproduce the accounting and inspect complete archived samples.
If a run has decreasing loss but
leaked development, an incorrect loss denominator or violated memory reserve,
do not call the whole experiment successful. Completion is a conjunction of
its stated criteria, not one attractive curve.
