# Course improvements from Stanford CS336

Date: 2026-10-11. Status: proposed, with source and local coverage review complete.
Local baseline: `78640952e7e6ffb35c57f4ab48569e6a660e93b6`, branch `main`.

The most useful upgrade is to connect our existing mechanisms into substantial
engineering assignments, then fill four remaining gaps: corpus curation,
hardware performance, scaling estimates, and reasoning experiments with usable
reward signal. Keep the fifteen chapters and twenty-eight days. Preserve D005:
SFT and reinforcement learning receive the greatest depth; pretraining remains
concise and empirical. The learner remains on Day 9 with coherent short English
stories as the first DongxiGPT objective.

This is a new proposal alongside the completed
[eighteen-package upgrade](COURSE_IMPROVEMENT_PLAN.md),
[editorial pass](BOOK_EDITORIAL_PLAN.md), and
[foundations pass](BOOK_FOUNDATIONS_PLAN.md). Their criteria and empirical records
remain the baseline. The assignments below are implementation targets, not new
completion claims or authorization for model-scale experiments.

## What to take from CS336

CS336 makes students carry an implementation through progressively harder
problems: a working language model, systems optimization, compute allocation,
data preparation, and post-training. Its current offering is Spring 2026.
[Official course](https://cs336.stanford.edu/)

Our course already supplies most of the basic mechanisms and a richer sequence
of preference, policy-gradient, critic, verifier, and distillation lessons.
The improvement should make those components meet in an experiment where the
reader must explain a decision and its consequences.

| Stanford assignment | Useful lesson for our course | Local adaptation |
|---|---|---|
| [A1 Basics](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf) | Build components that survive an end-to-end training and generation contract. | Reuse ByteBPE, decoder, loss, optimizer and recovery; strengthen tokenizer boundaries and one integrated deliverable. |
| [A2 Systems](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf) | Measure bottlenecks and understand attention IO, checkpointing and parallel training. | Teach a CPU reference and bounded single-GB10 measurements; keep a Triton kernel as an extension. |
| [A3 Scaling](https://github.com/stanford-cs336/assignment3-scaling/blob/03e9372992e913061b9e78b5cfcb62ad8a87de35/cs336_assignment3_scaling.pdf) | Predict a useful model/data allocation under a budget, then test the prediction. | Fit independently authored simulated curves first; propose a separate small empirical study. |
| [A4 Data](https://github.com/stanford-cs336/assignment4-data/blob/0555bea66369872d912652debf10b115ca0688c8/cs336_assignment4_data.pdf) | Turn source documents into a corpus and measure filtering and deduplication decisions. | Use original local HTML/text fixtures, exact and near-duplicate audits, and distribution-aware data cards. |
| [A5 Alignment and reasoning](https://github.com/stanford-cs336/assignment5-alignment/blob/c2734a26308710949fe13226960a1e8cece94b7e/cs336_spring2026_assignment5_alignment.pdf) | Connect prompt baselines, verifiable reward, grouped sampling and controlled RL experiments. | Reuse our SFT/post-training modules and negative results; add an integrated task/verifier/difficulty diagnostic. |

The current A2 leaderboard uses two B200 GPUs; A3 offers actual queued B200
training through a Stanford-restricted API. Those are different execution
conditions from our Spark. Reproduce the scientific questions using local
instruments, with each hardware result attached to its actual device.
[A2 handout](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf),
[A3 handout](https://github.com/stanford-cs336/assignment3-scaling/blob/03e9372992e913061b9e78b5cfcb62ad8a87de35/cs336_assignment3_scaling.pdf)

## Coverage that changes the recommendation

| Area | Current evidence | Remaining gap |
|---|---|---|
| Tokenization | `course_foundations.ByteBPE`, `tiny_bpe.py`, `meteor_bpe.py`, Unicode and embedding notebooks | Educational ByteBPE explicitly omits pretokenization, normalization and special-token policies; iterable encoding and serialization contracts need a focused extension. |
| Model and training foundations | Chapters 3–6; transparent attention and decoder; AdamW, masks, accumulation, schedules and recovery | Mostly adequate. Combine them in a larger assignment with interface and causal-invariance checks. |
| Data identity | `stories_data.py`, `evaluation_lab.contamination_groups`, `instruction_data_lab.group_split` | Normalized exact matching exists; executable extraction, near-duplicate retrieval and retention audits are missing. |
| Data mixtures | `instruction_data_lab.mixture_exposure` and Day 11's mixture notebook | Add finite source capacities and repeated exposure to the existing token-share lesson. |
| Resource accounting | Decoder parameter/FLOP/KV estimators; native profile and recovery evidence | Costs are covered, but arithmetic intensity, a roofline explanation and controlled operation benchmarks need a companion. |
| Efficient attention | Causal attention, modern SDPA, GQA and exact cache equivalence | Online/blockwise softmax and an independent IO-aware attention reference are missing. |
| Scaling | Budget and model-size examples; measured TinyStories learning curves | No scaling-fit/validation lab. Checkpoints from one model size cannot establish a model-size scaling law. |
| Evaluation and post-training | Frozen grading, paired uncertainty, text reward models, DPO controls, critics/GAE, GRPO variants, sampling accounting and response distillation | Reuse these. The main gap is an integrated experiment with task-reward signal and a frozen independent evaluation. |
| Reasoning controls | Three-seed sampled shared-decoder CPU control; native RLVR results with zero task advantages | The CPU control uses conditional symbolic syntax. It does not establish learnability of the native model/task/interface. |

The local anchors are the [notebook curriculum](NOTEBOOK_CURRICULUM.md),
[CPU reasoning control](../experiments/reports/2026-10-04-reasoning-controls.md),
[native campaign defense](../experiments/reports/2026-10-05-final-native-campaign-defense.md),
and [foundations record](../experiments/reports/2026-10-10-book-foundations-pass.md).
Existing negative results are useful starting cases for the new assignments.

## Six integrated assignments

These are milestones across existing sessions, not six additional course days.
Extend current lab guides and reuse the seventy-six notebook routes. New
notebooks belong in optional extension routes when the mechanism needs its own
session. Each milestone ends with one reviewable artifact and a short argument.

| Assignment | Chapters and days | Deliverable and decisive check |
|---|---|---|
| A. Tokens to a story model | Ch2–6; Days2–9 | Tokenizer identity, one traced batch, a small checkpoint, fixed generations and a recovery comparison. Trace an original sentence from text through an update and back to decoded output. |
| B. A corpus that changes the experiment | Ch6–8; Days8,10–11 | Extracted documents, retention ledger, verified duplicate groups, frozen splits and a data card. Show both a missed duplicate and useful text removed by a filter. |
| C. Explain where runtime goes | Ch4–6,14; Days6–8,25 | Operation/byte ledger, attention equivalence checks and a CPU benchmark report, with an optional Spark profile. Predict the bottleneck, change one variable and account for the observed result. |
| D. Predict under a compute budget | Ch5–7; Days7–10, optional extension | Fit and uncertainty report, an untouched prediction target and a decision under two budgets. Separate simulated curves from any later measurements. |
| E. From demonstrations to policy updates | Ch7–14; Days10–25 | Shared evaluation contract, SFT control, preference/RL comparisons, rollout audit and failure diagnosis. Establish whether task reward supplies a usable gradient before interpreting a native RL run. |
| F. Transfer and defend | Ch15; Days26–28 | Teacher-selection audit and one defended real comparison, with ancestry, costs, uncertainty and failures. Support each central conclusion with a stored response or measurement. |

Assignment E remains the largest. Existing model, data and evaluation lessons
establish its prerequisites. The new scaling/attention extensions and optional
Spark profiling do not block it; they provide depth after the post-training
path is coherent.

## Chapter changes

Each addition starts with a concrete question, develops the mechanism, invites
a prediction, and returns to the observed consequence. Reuse existing
derivations and figures where they already answer the question.

| Chapter | Content change | Lab connection |
|---:|---|---|
| 1 | Reuse the existing correctness/performance/capability distinction to review one integrated experiment design. | Assignment briefs state a claim, budget, control and rejected inference. |
| 2 | Develop tokenizer boundary contracts: a control marker and a split text chunk can change the token stream. Keep the current ByteBPE microscope. | Extend Day2/Lab2 with special-token isolation, serialization and explicitly framed iterable encoding. |
| 3 | Extend the existing tokenizer-dependent perplexity example into corpus-level nats/bits per byte, with scored-text, mask and EOS conventions. | Add an aggregation exercise to the existing loss notebook; compare equivalent probabilities under different token segmentations. |
| 4 | Explain how blockwise normalization preserves attention while reducing materialized intermediates. | CPU online-softmax microscope with output/gradient checks and a deliberately unstable implementation. |
| 5 | Connect tensor layouts and model costs to arithmetic intensity; distinguish theoretical payload from actual execution. | Operation map plus one decoder batch in Assignment C. |
| 6 | Teach extraction, duplicate policies and retention as changes to the training distribution; add one compact scaling-estimation section. | Corpus preparation, source-exposure plots and a separate prediction-under-budget extension. |
| 7 | Expand contamination from exact hashes to verified document/source families. Explain the distinction between a calibration set and the untouched prediction/evaluation target. | Duplicate-component split audit and held-out scaling prediction. Reuse mathematical grading and paired uncertainty. |
| 8 | Extend mixture weights to finite source capacities, repeated exposure and removed-source slices. | Extend Day11's mixture/data-card notebook using Assignment B outputs. |
| 9 | Add a compatible unsaturated SFT comparison: distinct source coverage versus repeated examples at matched target exposure. | Reuse the canonical SFT kernels; compare frozen parent, SFT and accepted-response/policy controls across source/template/family slices. |
| 10 | Reuse label disagreement and reward shortcuts in the common assignment's frozen response population. | Reuse the text reward and judge-audit labs; freeze the learned scorer before the policy stage. |
| 11 | Reuse the existing chosen-only versus preference comparison, including rejected/reference work, in the common assignment. | DPO retention controls and actual chosen/DPO outcomes provide the reference; require common outputs and a stated matching rule. |
| 12 | Reuse estimator/behavior-likelihood derivations to inspect optimizer applications in the common assignment. | Sampling-support, learned-critic and GAE lessons support one real or bounded CPU update audit. |
| 13 | Quantify a difficulty policy's changed prompt population, finite-group effects and selection bias. | Compare unfiltered and selected prompt distributions with the existing objective controls and actual collection cost. |
| 14 | Reuse the existing failure taxonomy in the integrated assignment, adding the verifier transport and systems cases. | Verifier audit plus the benchmark report; retain the original strict native failures. |
| 15 | Extend the existing real-comparison defense into the assignment's assessed transfer deliverable. | Reuse response distillation, genealogy and campaign records; add a design recommendation and a limitation that could change it. |

Machine commands remain in appendices/runbooks. Each new exercise has its
adjacent reference solution, a controlled failure, and a plot with labels,
reading guidance and regeneration code. Mathematical mechanism proposals enter
the existing animation queue during implementation; rendering remains a
separately authorized Mac task.

## Implementation packages

P0 packages provide the main upgrade. P1 packages add focused depth after that
path is coherent. P2 work remains optional or a later course version.

### CS336 01 Assignment structure and assessment

Priority P0. Extend the sixteen existing `book/labs/` guides and create
`docs/COURSE_ASSIGNMENTS.md` with the six briefs above. Reuse run identities,
existing evaluation contracts and actual reports. Use a common review rubric:
mechanism correctness, controlled comparison, interpretation and reproducibility.

Acceptance: each assignment names inputs, reused modules, one deliverable, an
independent acceptance check and a failed-case interpretation. Every chapter
links to its milestone. The twenty-eight-day mapping, existing exercise IDs and
learner cells are preserved. A passed reference implementation is separate from
a learner's submitted explanation. Avoid a second experiment orchestration
framework or a duplicate status ledger.

### CS336 02 Tokenizer contracts and comparable likelihood

Priority P1. Add `src/dongxi_llms/tokenizer_contracts.py` as an adapter around
the existing `ByteBPE`; retain its default behavior. Extend
`tokenizer_mechanics_lab.py`, the Day2 tokenizer notebook, and the Day3 loss
notebook. Add scored-byte aggregation to an existing loss/evaluation module.

Specify segmentation, normalization, special-token admission, overlapping marker
selection, serialization and malformed-byte decoding. Define whether iterable
elements are independent records or chunks of one record. If supporting chunks,
buffer unfinished pretokens and demonstrate why arbitrary chunkwise encoding
does not preserve whole-text tokens. State the associated memory boundary.

Acceptance: original merges remain unchanged; Unicode and unseen UTF-8 text
round-trip; special markers cannot be merged into ordinary text; serialized
identity reproduces IDs. Whole/iterable equivalence holds under the declared
framing. Bits per byte uses summed scored NLL over corresponding raw bytes,
with explicit EOS/context/mask conventions, and refuses incompatible contracts.
Use original edge cases rather than copying Stanford tests.

### CS336 03 Corpus curation and exposure

Priority P0. Add `src/dongxi_llms/corpus_curation_lab.py` and original local
HTML/text fixtures. Extend the Day8 data notebook, Day10 contamination exercise
and Day11 mixture notebook. Reuse `stories_data.py` and
`pretraining_lab.TrainingSession` for their existing training/accounting contracts.
The latter currently fixes its corpus and seed; add minimal corpus/seed inputs
while preserving its default reference behavior and byte vocabulary.

Start with a bounded fixture containing useful short text, code, multilingual
text, navigation, altered copies and fictitious contact strings. Use an
established HTML parser, then retain raw text, matching normalization and training
text separately. Every removal receives a reason code; inspect removals and
retained examples by authored content slice.

For near duplicates, teach shingles and exact Jaccard first. Add seeded MinHash
and locality-sensitive hashing for candidate retrieval, verify candidates with
exact similarity, then group verified edges. Compare approximate retrieval with
an exact all-pairs oracle on the small fixture. Connected components are a
grouping policy: two members connected through a third need not be directly
similar. Preserve independent evaluation sources before selecting filter rules.

Extend existing mixture accounting with source token capacity. For source token
share $p_s$, total presented tokens $T$, and available source tokens $N_s$,
effective exposure is $p_sT/N_s$. Specify whether these are input or supervised
tokens. Display objective share beside repetitions and distinct retained coverage.

Acceptance: counts reconcile from extraction to split membership; failures retain
document identity. Exact duplicates, perturbed copies, false candidates, empty
texts and transitive groups have independent expected outcomes. Report candidate
recall and precision against the exact oracle. No declared source/duplicate group
crosses the fixture's splits. A rule's score is not a calibrated quality estimate.

The integration reference compares raw, exact-deduplicated and curated variants
under one tokenizer and matched initializations. Freeze context, visibility,
loss reduction, update geometry and the learning-rate schedule. Match valid
targets per update where feasible; otherwise identify exposure-matched and
update-matched controls separately and report the Adam/schedule difference.
Use three predeclared CPU seeds and a maximum of 24 updates per arm initially;
record runtime, repetitions, slice NLL and fixed generations. These are proposed
caps to validate in a smoke reference, not measured results. Preserve inconclusive
or worse curated outcomes. The plots are a retention waterfall, similarity map,
duplicate graph and exposure/learning curves. Production-scale crawl processing
is a later extension.

### CS336 04 Hardware performance and checkpointing

Priority P0 for the explanatory CPU instrument; optional Spark measurements.
Add `src/dongxi_llms/systems_performance_lab.py` and
`notebooks/extensions/cs336/01_resource_accounting.ipynb`. Link it from
Chapters5–6 and Day25. Reuse decoder cost/cache estimators and current native
profiling evidence.

Build an operation ledger for matrix multiplication, attention, prefill and
single-token decoding. Explain arithmetic intensity and the compute/bandwidth
roofline using explicit inputs. Estimated traffic is a model with stated cache
and reuse assumptions, not a measurement of memory-controller traffic.
Reuse `stories_training.StoriesDecoder(activation_checkpointing=...)` and its
existing gradient-parity tests to compare checkpointing on/off on a small MHA
configuration with Q/K normalization disabled: retained activation storage,
recomputed work, matching loss and matching gradients. No new checkpointing
implementation is needed.

For a later Spark profile, measure a fixed small shape matrix, then alter batch
size or sequence length one at a time. Separate kernel timing from end-to-end
timing; warm up, synchronize correctly, retain repetitions and report median and
spread. Record device, software, dtype, attention backend and allocator/memory
semantics. A backend name does not prove which fused kernel ran.

Acceptance: FLOP/byte units and dimensions reconcile; output/gradient tolerances
are declared before benchmarking. Checkpointing preserves the objective and
shows recomputation rather than asserting a speedup. The roofline has labeled
assumptions and estimated/measured points for the same device and precision.
When deriving throughput from the existing cost helper, call it estimated matmul
FLOP/s and identify the counted operations and timed phase. Report this quantity
without utilization when a suitable device/precision peak is unavailable; do not
substitute H100/B200 peaks for GB10 or put CPU measurements on a GPU roofline.
An asynchronous timer without synchronization is the deliberate failure case.

### CS336 05 IO-aware attention

Priority P1, after CS33604. Add `src/dongxi_llms/online_attention_lab.py` and
`notebooks/extensions/cs336/02_online_attention.ipynb`. Link from Chapter4's
transparent attention and Chapter5's cost discussion.

Implement independently authored stable online softmax and blockwise causal
attention in PyTorch. Trace running maxima, normalizers and weighted values on
small matrices. Compare materialized intermediate sizes with a dense reference.
Backward verification in this P1 reference uses ordinary PyTorch autograd.
It may retain intermediates: label forward workspace estimates separately from
measured saved storage and process/device peak. A transparent recomputed
autograd backward is a separately scoped extension; the forward calculation
does not establish backward memory savings.

Acceptance: output and gradients match the dense reference across tile sizes,
causal/padding boundaries and ragged lengths. Define the fully masked-row result
and test it. Large-magnitude logits expose the unstable variant. Show tile flow
and intermediate-size plots. A CPU reference establishes mechanism correctness,
not FlashAttention performance.

A single-device Triton forward/backward extension is P2, conditional on actual
compatibility and usefulness of the installed GB10 stack. It has independent
gradient checks and benchmarks against the available SDPA backend; unsuccessful
performance results remain results. Multi-GPU FSDP/DDP implementations and
leaderboard reproduction are deferred. A small communication/cost diagram can
explain their role without claiming distributed execution.

### CS336 06 Scaling estimates and prediction

Priority P1. Add `src/dongxi_llms/scaling_lab.py` and
`notebooks/extensions/cs336/03_scaling_prediction.ipynb`; integrate compact prose
in Chapters5–7. This replaces a broad architecture/model-size sweep.

Use original, explicitly simulated runs to learn fitting and identifiability.
The dense short-context approximation $C\approx6ND$ relates training FLOPs $C$,
parameter count $N$ and presented tokens $D$; explain exclusions before using it.
Fit optima along equal-compute curves, examine residuals and uncertainty, and
hide a larger simulated budget until the prediction is frozen. Include sparse
coverage, a changed architecture and data reuse as deliberate fit failures.

Acceptance: recover a known simulated relationship within declared tolerance;
diagnose insufficient budgets, boundary minima and invalid logarithm inputs.
Do not fit or tune on the hidden prediction target; report prediction error on
the entire held-out budget. Retain failed fits and show sensitivity to fit range
and data points. Separate training/dev/held-out loss,
unique data and repeated presentations, FLOPs and wall time. Old TinyStories
checkpoints illustrate learning over time but do not become a scaling fit.

A later empirical extension may use three tiny sizes, three target budgets and
two seeds under one tokenizer/corpus/protocol. First profile the smallest run;
freeze total runtime and token caps before starting the matrix. This is an
unexecuted protocol, and even a completed tiny matrix would support only local
prediction within its range. Stanford API access is unnecessary.

### CS336 07 Integrated post-training and reward signal

Priority P0; largest teaching investment. Reuse `sft_lab.py`, `dpo_lab.py`,
`critic_policy_lab.py`, `grpo_objective_controls.py`, `reasoning_controls.py`,
the bounded mathematical grader and current native runners. Add a focused
`src/dongxi_llms/posttraining_assignment.py` consumer and
`notebooks/extensions/cs336/04_reward_signal_audit.ipynb`. It should assemble
existing kernels and diagnostics rather than implement another trainer.

Existing microscopes use different vocabularies, encodings and model
configurations. Define one compatible integration fixture: vocabulary, prompt
serialization, model configuration, supervised/response masks, stop IDs and
sampling support. Adapt the reusable kernels to that interface. Trace an actual
CPU parent through SFT, then branch from that SFT checkpoint into accepted-response
SFT and a reward-based update, retaining the frozen parent at each comparison.
Standalone microscope checkpoints are not interchangeable. The historical
random-init stories, Base/SFT/preference and acquired Instruct/RLVR roots remain
separate in the actual campaign genealogy.

The new SFT comparison starts from measured unsaturated behavior on this
compatible fixture. Compare broader unique-source coverage with repeated examples
at matched target exposure. Group by underlying problem/source; evaluate paired
template forms on the same held-out sources and retain an independent unseen
task-family slice. Report answer and termination scores separately. This tests
coverage and delivered behavior alongside the existing Full400/LoRA evidence.
Optional native follow-through needs a broader unsaturated task than the current
three shared copy/reverse/extract templates.

The current native collector rewards terminated responses passing
`verify_integer`; the richer saved-response grader answers a different contract.
Replay the same retained text through both instruments and inspect disagreement.
Separate correct answer, supported parsing, formatting, natural termination and
truncation. Any richer training reward needs a new named protocol and independent
tests; it cannot replace the historical strict results.

Next inspect all-wrong, all-right and mixed groups by task slice. Reuse the
existing constructive CPU control and add a separately named, independently
solvable parity or signed-operand task. Both answer classes must have multiple
distinct source groups in the relevant splits; compare against explicit constant
and majority-answer baselines, including their partial accuracy. The original
nonnegative sum-positive task has only one negative pair, so simple reweighting
cannot establish this source diversity. Predeclare a
difficulty policy on training/development data and record its changed prompt
distribution, discarded groups and collection cost. Selecting mixed groups
changes the estimator's population; it is not free extra signal.

For the native extension, the first milestone is a frozen capability and reward
audit, followed by a same-parent SFT control and a bounded RL pilot only if the
training/development slice supplies usable signal. Freeze evaluation first.
Select difficulty and caps using development evidence, never held-out results.
If signal remains absent, the diagnostic is the result; do not run a longer
campaign to satisfy an expected improvement.

Compare plain GRPO with one motivated existing normalization/weighting control;
reuse the current on/off-policy likelihood and rollout-freshness lessons.
For a tiny fixed reward distribution, enumerate finite groups and compare
expected task-gradient direction and variance with the large-group approximation.
Preserve the dependence between a response and its group's normalizer; treating
those quantities as independent would change the calculation.
MaxRL/GSPO can appear as optional reading after the relevant estimator is
understood, rather than creating mandatory algorithm implementations.

Acceptance: original native negative rewards replay unchanged; verifier
disagreements retain full text and reason codes; adversarial checker probes have
independent references. The compatible CPU chain preserves actual checkpoint
ancestry and interface identity through SFT and both sibling interventions.
Its balanced fixture reports all predeclared seeds and constant/majority
baselines. The signal gate requires genuine sampled successes and failures on
training/development data, mixed groups and an independently measured task-reward
gradient component. Record mixed-group fraction, task-advantage norm,
objective-gradient components, completion rate, entropy, KL and physical rollout
cost. A nonzero total gradient alone cannot establish a task-reward gradient.

A future native comparison uses common frozen evaluations and reports its exact
matching choice: supervised labels, response tokens, generation work and updates
are distinct budgets. Include SFT-only, frozen-parent and policy controls, raw
responses, paired uncertainty and regressions. Mechanism readiness can pass
without claiming a capability improvement; empirical claims need their own
executed evidence.

## Implementation order and effort

These are planning estimates for one contributor, including review and local
reference verification. They are not learner days or measured runtimes.

| Batch | Work | Dependency and estimated effort |
|---|---|---|
| 1 | CS33601 briefs/rubric; CS33603 corpus audit; CS33607 verifier and reward audit | Start with existing fixtures and evidence; 4–6 contributor days. |
| 2 | CS33607 balanced CPU integration; SFT/preference/RL assignment connections; capstone defense | After shared evaluation and data identities; 3–5 days. |
| 3 | CS33604 CPU systems instrument; CS33602 tokenizer contracts; CS33606 simulated scaling | Independent after relevant contracts are settled; 3–5 days. |
| 4 | CS33605 attention mechanism; complete chapter/solution/route integration and review | After systems units and shape checks; 2–4 days. |

Total proposed core/P1 source work: roughly 12–20 contributor days. Reassess after
Batch1. Native pilots, measured scaling, Triton kernels and actual cross-machine
portability have separate evidence scopes and are excluded from this estimate.
Prefer completing the post-training path to expanding the systems extension.

## Verification and execution boundaries

Before implementing each gap, recheck the current code and upstream state.
Reuse an adequate solution. Preserve original exercises, notebook learner cells,
baseline experiments and the original upgrade acceptance arrays.

Use the existing local math, prose, course-route and notebook verifiers. Add
focused independent tests for tokenizer boundaries, exact similarity oracles,
attention gradients, scaling predictions and verifier adversaries. Execute new
CPU references in an isolated environment and inspect regenerated figures.
Run the affected references first; run the full local suite and registered
notebook set once at integration closure. No hosted course CI is introduced.

New native protocols stay unexecuted until their own task scope and scientific
prerequisites cover the recipe. Declare model/data revisions, mode, seeds,
dtype, geometry, attention backend, token/update/runtime limits and memory
reserve. On Spark retain at least 20–25 GiB for the OS and interactive work under
the current platform policy. Use existing bounded runners and ordinary resource
checks. Mac evidence remains independent of Linux execution evidence.

Publication, animation rendering, public crawling and large distributed
experiments remain separate work. Completion of this proposal's source packages
does not advance learner mastery or convert unrun experiments into results.

## Pinned references and reuse

Read current handouts and interface tests rather than relying on repository
titles. A1's README still says 2025, while its pinned handout says Spring2026.
The pinned A5 main handout uses OLMo-2-0425-1B/GSM8K; the archived 2025 assignment
uses a different model/data recipe. Neither should silently replace our Qwen
checkpoint family. The current A3 API launches real jobs; the lecture overview's
description of responses based on previous runs describes a different setup.

| Source | Inspected revision | License evidence and use |
|---|---|---|
| A1 Basics | [a158843](https://github.com/stanford-cs336/assignment1-basics/tree/a158843b20107949f1a8d7df1b05cd33b9166712), handout26.0.3 | [Root MIT license](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/LICENSE). Inspect contracts; create original exercises. |
| A2 Systems | [ca8bc81](https://github.com/stanford-cs336/assignment2-systems/tree/ca8bc81a59b70516f7ebb2da4808daade877c736) | [Root MIT license](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/LICENSE). Teach mechanisms with local hardware evidence. |
| A3 Scaling | [03e9372](https://github.com/stanford-cs336/assignment3-scaling/tree/03e9372992e913061b9e78b5cfcb62ad8a87de35), handout26.0.5 | [Root MIT license](https://github.com/stanford-cs336/assignment3-scaling/blob/03e9372992e913061b9e78b5cfcb62ad8a87de35/LICENSE). Create our own simulation and local protocol. |
| A4 Data | [0555bea](https://github.com/stanford-cs336/assignment4-data/tree/0555bea66369872d912652debf10b115ca0688c8), handout26.0.1 | [Root MIT license](https://github.com/stanford-cs336/assignment4-data/blob/0555bea66369872d912652debf10b115ca0688c8/LICENSE). Author local documents and independent implementations. |
| A5 Alignment | [c2734a2](https://github.com/stanford-cs336/assignment5-alignment/tree/c2734a26308710949fe13226960a1e8cece94b7e), main handout26.0.0 | No LICENSE/NOTICE file found in the inspected tree; use conceptual references and links. |
| Current lectures | [de53a9f](https://github.com/stanford-cs336/lectures/tree/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e) | No LICENSE/NOTICE file found in the inspected tree; create original prose, code and figures. |

Additional primary anchors:
[A1 tokenizer contracts](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/tests/test_tokenizer.py),
[A4 data interfaces](https://github.com/stanford-cs336/assignment4-data/blob/0555bea66369872d912652debf10b115ca0688c8/tests/adapters.py),
[resource accounting lecture](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_02.py),
[data filtering and mixing lecture](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_14.py),
[current safety supplement](https://github.com/stanford-cs336/assignment5-alignment/blob/c2734a26308710949fe13226960a1e8cece94b7e/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf).

No Stanford prose, figures, datasets or substantial code are incorporated by
this plan. Any later source reuse must check the relevant file and retain its
attribution, including third-party material inside a repository with a root
license.
