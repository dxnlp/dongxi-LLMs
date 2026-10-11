# Integrated course assignments

The fifteen chapters teach mechanisms; these six assignments ask you to connect
them into a working experiment and defend a decision. They share the existing
twenty-eight-day route. Complete the small transparent reference before choosing
a larger experiment. The post-training assignment receives the most time.

Your deliverable is a result someone else can inspect: inputs and identities,
working implementation, a controlled failure, measurements, and an explanation.
An unsuccessful intervention can be an excellent assignment when it establishes
why the proposed conclusion is unsupported. Reference solutions and course
verification establish material readiness; your prediction and explanation
establish your own learning.

## Route and assessment

| Assignment | Main route | Central decision |
|---|---|---|
| [A Tokens to a story model](#a-tokens-to-a-story-model) | Chapters2–6, Days2–9 | Does the text-to-loss-to-generation pipeline implement the intended prediction problem? |
| [B Corpus preparation](#b-corpus-preparation) | Chapters6–8, Days8,10–11 | Which data decisions change useful coverage, repeated exposure and leakage? |
| [C Explain runtime](#c-explain-runtime) | Chapters4–6,14, Days6–8,25 | Which operation is costly, and does the proposed change address that cost? |
| [D Predict under a budget](#d-predict-under-a-budget) | Chapters5–7, Days7–10 enrichment | What can smaller observations justify about an untouched budget? |
| [E From demonstrations to policy updates](#e-from-demonstrations-to-policy-updates) | Chapters7–14, Days10–25 | Which intervention changes delivered behavior under one compatible interface? |
| [F Transfer and defend](#f-transfer-and-defend) | Chapter15, Days26–28 | What did the student inherit, and which comparison supports the final recommendation? |

Review each deliverable on four dimensions. **Correctness:** shapes, masks,
probabilities, units and independently checked outputs agree with the declared
mechanism. **Comparison:** the control, changed factor, split and budget make the
question identifiable. **Interpretation:** concrete outputs support the stated
claim, including adverse slices and failures. **Reproduction:** source,
configuration, environment and representative records allow another reader to
reconstruct the argument. A collection of successful commands answers only part
of this rubric.

Use the existing run/evaluation identities and experiment templates. Retain
original responses and failure records. Write a prediction before running each
reference; follow it with the adjacent code and mechanism explanation. Changing
the reference's inputs creates a new observation rather than updating a saved
result automatically.

## A Tokens to a story model

Begin with one short original sentence. Follow its raw bytes, token boundaries,
IDs, embeddings, contextual states, next-token targets, summed loss and parameter
update. Then decode a continuation and explain why the training target differs
from the model's generated action. Reuse the Chapter2–6 mechanisms and the
[Day8 training sessions](../notebooks/day-08/README.md).

Predict whether splitting text into arbitrary chunks preserves its token IDs.
The [tokenizer contract companion](../notebooks/extensions/cs336/05_tokenizer_contracts.ipynb)
checks segmentation, special markers, frozen merges and serialization. Its
reference distinguishes an iterable of records from chunks belonging to one
record. A merge can cross a chunk boundary; an adapter must preserve the declared
record, and buffering has a memory cost.

Use one tiny training configuration, fixed validation, fixed generation settings
and one checkpoint recovery comparison. Explain the learning-rate clock and
the number of actual targets. Introduce one controlled defect, such as a wrong
shift or missing optimizer state, and identify the first invariant it violates.
Use the existing masking/recovery companions rather than building a second
trainer.

Submit a traced batch, tokenizer identity, checkpoint identity, before/after
outputs, recovery check and one-page explanation. Acceptance requires causal
information isolation, finite independently checked loss/gradients, matched
recovery under the declared tolerance and honest generation outcomes. Use the
retained [TinyStories evidence](../experiments/reports/2026-09-14-tinystories-learning-result.md)
to connect the small mechanism with the real model: lower prediction loss and
coherent storytelling are separate observations.

## B Corpus preparation

A data pipeline chooses what the model sees. Start from the original bounded
HTML/text fixture in the
[corpus companion](../notebooks/extensions/cs336/06_corpus_curation.ipynb).
Inspect extraction, each retention rule, matching normalization, verified
duplicate groups and source/split membership. Preserve raw, matching and
training text as distinct representations.

Predict which altered copy escapes exact hashing and which useful short example
a threshold removes. The reference first computes exact shingle similarity,
then uses seeded approximate retrieval to propose pairs. Verify candidates
before assigning duplicate components. A connected group can contain members
whose direct similarity is below the threshold; explain why grouping that
component is a policy decision.

Freeze evaluation source groups before selecting thresholds. Compare raw,
exact-deduplicated and curated variants with one tokenizer and matched
initializations. Match valid targets per update where feasible; otherwise report
exposure-matched and update-matched comparisons separately. Track unique source
coverage, omitted tails, repeated presentations, learning-rate clocks and slice
NLL. Follow the mixture lesson with a scarce source: a large objective share can
mean many repetitions of a small source.

Submit extracted documents, a reason-coded retention ledger, candidate/oracle
precision and recall, a duplicate graph, split audit, exposure table and data
card. Include useful discarded text and retained unwanted text. Counts must
reconcile across stages. For the bounded training reference, retain all declared
seeds and fixed endpoint outputs; cleaner-looking data need not improve the
measured objective or generations. The fixture teaches the mechanism and does
not validate a web-scale quality classifier.

## C Explain runtime

Start by naming a timed phase: forward, backward, optimizer step, prefill or
decoding. Derive the operation and estimated byte ledger for its exact shapes.
The [resource companion](../notebooks/extensions/cs336/01_resource_accounting.ipynb)
connects these units with arithmetic intensity and a device/precision-specific
roofline. Logical payload, retained tensors, allocated/reserved bytes and process
peak are different quantities.

Predict whether increasing batch size will increase arithmetic intensity. Change
only batch size, then report timing and spread under the declared protocol.
For CUDA, warm up and synchronize; report kernel and end-to-end timing separately.
When FLOPs come from a matmul estimator, label the throughput estimated matmul
FLOP/s. A short profiler trace helps locate operations, but collect normal timing
without profiler overhead. CPU observations remain CPU observations.

In the [attention companion](../notebooks/extensions/cs336/02_online_attention.ipynb),
predict what happens when a later tile introduces a larger score. The stable
reference rescales earlier contributions when updating the running maximum;
the broken variant does not. Compare outputs and gradients with dense attention
on masked and partial tiles. Ordinary autograd can retain intermediates even
when the forward avoids one large score matrix.

Reuse the checkpointed story decoder to compare equal losses and gradients with
different recomputation/storage behavior. Submit the operation map, correctness
checks, raw timings, backend/environment identity and an intervention decision.
A truthful finding of no speedup satisfies the scientific rubric. The bounded
Spark profile is a separate hardware receipt; a Python attention loop does not
establish a fused-kernel speedup.

## D Predict under a budget

Use the [scaling companion](../notebooks/extensions/cs336/03_scaling_prediction.ipynb)
to fit independently authored simulated observations. Hold out an entire compute
budget, freeze the fitted method and prediction, then reveal the target. Equal
compute profiles ask how model size and token presentations compete; they are
different from following one model's checkpoints over time.

Predict whether the lowest observed loss at the edge of the model-size grid
identifies an interior optimum. The reference detects insufficient coverage,
boundary minima and unstable fits. Report residuals, sensitivity and held-out
prediction error. A neat curve cannot supply uncertainty about an untested
architecture or a different data-reuse regime.

Submit the observation table, fitting code, assumptions, frozen prediction,
revealed error and a recommendation under two budgets. Label simulated data
throughout. The small empirical extension has its own fixed corpus, tokenizer,
search allowance, model sizes, token budgets and seeds. Retain failed runs and
predictions; a tiny local matrix supports a local statement, not a forecast of
large-model performance. This enrichment route does not block Assignment E.

## E From demonstrations to policy updates

Use one explicitly compatible teaching interface: vocabulary, serialization,
model configuration, target ownership, stop IDs and sampling support. Existing
standalone SFT, DPO, critic and GRPO microscope checkpoints use different
interfaces. Reuse their kernels after adapting them to the common contract.
The [integration companion](../notebooks/extensions/cs336/07_posttraining_integration.ipynb)
carries an actual CPU parent through SFT and reward-based sibling interventions.

First predict whether repeating a narrow set of demonstrations will substitute
for distinct source coverage. Match supervised target exposure, then compare
delivered answers and stopping against the frozen parent. Use underlying
source/problem groups, paired template forms on common held-out sources, and an
independent withheld-family diagnostic. Report the fixed constant/majority
baseline. A new template and a new source must not silently become one isolated
template claim.

Next freeze the SFT parent. Collect attempts, retain accepted and rejected
responses, and compare accepted-response SFT with a reward-based sibling.
Account for teacher/generated tokens separately from student supervision and
optimizer updates. Keep all seeds and predetermined endpoints. Reward learning
can fail while accepted-response fitting succeeds, or the reverse.

Before interpreting RL, use the
[reward audit](../notebooks/extensions/cs336/04_reward_signal_audit.ipynb).
Predict which responses are semantically correct but fail the strict integer or
termination predicate. Inspect complete text, parser support, formatting,
correctness, natural stopping and truncation. Replaying historical records under
a richer checker is a new measurement; historical strict rewards remain unchanged.

Measure all-wrong, all-right and mixed groups on training/development sources.
The balanced task supplies multiple distinct sources for both classes. Show the
task-reward gradient separately from KL and optimizer effects. A nonzero total
gradient does not prove reward signal. A difficulty policy changes the sampled
prompt population and can add discarded collection work: compare it with the
unfiltered distribution and preserve finite-group dependence.

Submit the actual checkpoint tree, shared evaluation contract, SFT coverage
comparison, collection ledger, sibling outputs, verifier adversaries and failure
diagnosis. Acceptance requires independently checked masks/likelihoods, actual
compatible ancestry, untouched held-out selection, all raw responses and a
bounded claim. A capability gain is an empirical outcome rather than a required
answer. A later native pilot first needs sampled successes and failures, mixed
groups and a separately measured task-gradient signal on training/development
data. An absent signal calls for diagnosis before longer RL.

Use existing preference/judge, learned-reward, DPO-retention and critic lessons
to explain alternative interventions on the same question. Keep the actual
story, Base/SFT/preference and Instruct/RLVR roots separate when comparing this
teaching chain with the [native campaign](../experiments/reports/2026-10-05-final-native-campaign-defense.md).

## F Transfer and defend

Reuse the [Day26 selection/distillation route](../notebooks/day-26/README.md)
and the actual teacher/student records. Identify what each student target says:
a final answer, a printed step, a hard token or a probability distribution.
Predict whether selecting correct final answers will remove incorrect reasoning
steps; inspect concrete counterexamples before making the transfer claim.

Choose one real comparison from the course evidence. Reconstruct ancestry,
training/selection populations, inference contract, held-out denominator,
uncertainty and physical cost. Separate oracle candidate availability from the
deployed selector's success. Preserve failed and missing observations. A result
from a synthetic genealogy or readiness fixture cannot supply weight identities
or quality measurements for that comparison.

Submit a short scientific defense: one design recommendation, its supporting
responses/measurements, an adverse slice, a plausible alternative explanation
and one next experiment that would distinguish them. Include exact source/report
links and reproduction commands. Explain a limitation that could change the
recommendation. The result is a reviewable course artifact; publishing it is a
separate action.

## Reference execution

The [companion index](../notebooks/extensions/cs336/README.md) gives the new
mechanisms and their adjacent references. Use the isolated CPU teaching
environment in [Appendix D](../book/appendices/d-reproduction-and-environments.md).
The registry keeps each companion attached to its chapter/day. The existing
notebook verifier executes references in fresh kernels and stores new executed
copies, leaving learner files intact. Specifications and reports retain actual
platform, inputs, limits and outputs; the [CS336 goal](CS336_IMPROVEMENT_PLAN.md)
records readiness and bounded native evidence separately.
