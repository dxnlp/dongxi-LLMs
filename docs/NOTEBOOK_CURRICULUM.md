# Interactive Notebook Curriculum

This document maps the book's conceptual argument to executable learning
sessions. Every chapter receives a notebook pathway; notebooks are not optional
demonstrations added after the prose is finished.

The purpose is not to maximize notebook count. Each session should isolate one
important mathematical, architectural, optimization, data, or evaluation
mechanism that becomes substantially clearer when the learner can inspect and
change it.

Since 2026-09-09 the learner also requests real-time visual tools outside
notebooks. This file still defines the executable notebook pathway; pair it
with [the live-learning and machine workflow](LEARNING_WORKFLOW.md). Mac Studio
hosts low-latency conceptual interaction, while Spark supplies separately
approved GPU evidence. The proposed first pilot is the Day 8 Optimizer
Playground, not yet implemented. Do not replace existing worked notebooks or
equate a proposed visualization with demonstrated understanding.

Current position: completed Day 9 baseline on Spark, now interpreting results.
The three Day 8 visual notebooks remain ready, but do not implement the full
live optimizer playground. The learner found
the isolated quadratic slider boring; start from meaningful decoder/data/recipe
interventions and keep live discussion available outside notebook execution.

Full-course build,2026-10-04: all 15 chapters now have an implemented notebook
pathway,61 sessions across 28 days. The learner explicitly requested all future
material ahead of live learning; this supersedes incremental activation for
this build. The [daily route](COURSE_SEQUENCE.md) links every session index.
Fresh-kernel verification checks material, not learner mastery.

Day 9 now has three [built visual lessons](../notebooks/day-09/README.md):
saved training clocks and useful targets; padding/document masks; and story
quality versus decoding. They read the portable2026-09-14TinyStories JSON and
retain retrospective sample-selection labels. These are completed-reference
analyses, not a new controlled training comparison or independently studied work.

## Chapter contract

Reference-audit extension plan,2026-10-04: proposed sessions in
[DXI-01 through18](COURSE_IMPROVEMENT_PLAN.md) add checkpoint-interface,
math-grading, judge-bias, text-reward/process-label, critic/GAE,
sampling-support, GRPO-denominator, selection/control, response-distillation,
batching/recovery and refinement microscopes. They deepen existing chapters;
no additional mandatory day or empty notebook placeholder is created here.
The accepted inventory is76 notebooks. Fifteen verified extensions are:

- [Day1 checkpoint interface](../notebooks/day-01/04_checkpoint_interface.ipynb):
  five code cells/one mapping figure; actual local save/reload and controlled failure.
- [Day10 mathematical grading](../notebooks/day-10/04_mathematical_grading_and_response_replay.ipynb):
  six cells/four figures; original authored response replay, not live model evaluation.
- [Day15 preference/judge audit](../notebooks/day-15/03_preference_collection_and_judges.ipynb):
  eight cells/three figures; authored references and simulated judges, not human feedback.
- [Day20 probability accounting](../notebooks/day-20/03_behavior_probabilities_and_support.ipynb):
  seven cells/four figures; exact finite support/importance/KL and response masks.
- [Day16 text reward and process labels](../notebooks/day-16/03_text_reward_and_process_labels.ipynb):
  twelve cells/eight figures; preserved word-token failures, encoding-disjoint
  character intervention, calibration-only scaling and persistent held-out failures.
- [Day23 sampled reasoning controls](../notebooks/day-23/03_reasoning_tasks_and_positive_controls.ipynb):
  eight cells/five figures; actual shared decoder/EOS learning across three
  fixed seeds, constant baselines and held-out failures, not English reasoning.
- [Day18 DPO retention controls](../notebooks/day-18/02_dpo_retention_and_preference_controls.ipynb):
  eight cells/five figures; actual48-fit shared-decoder comparison, absolute
  likelihoods, noisy/long winners, extra rehearsal and weak held-out generation.
- [Day20 learned critics and frozen rewards](../notebooks/day-20/04_learned_critics_and_frozen_text_rewards.ipynb):
  seven cells/five figures; actual short actor/critic updates, TD/GAE and terminal
  versus cap boundaries; learned proxy and independent quality can disagree.
- [Day22 matched objective and filtering controls](../notebooks/day-22/03_objective_weighting_and_filtering.ipynb):
  six cells/five figures; exact coefficients and actual shared-parameter gradients,
  labeled clipping fixture and all rejected-attempt costs, not trained method gains.
- [Day10 actual candidate selection](../notebooks/day-10/05_actual_candidates_and_answer_selection.ipynb):
  seven cells/five figures;864 real tiny-decoder candidates, gold-blind decisions,
  correlated error controls and measured generation/rescoring work.
- [Day11 teacher attempts and rejection SFT](../notebooks/day-11/04_teacher_attempts_and_matched_rejection_sft.ipynb):
  eight cells/six figures; original programmatic teacher journal, actual sequence
  students and coverage/exposure controls with held-out failures.
- [Day25 ragged KV and exact recovery](../notebooks/day-25/02_ragged_kv_cache_and_exact_recovery.ipynb):
  seven cells/five figures; actual compact KV/positions/stops, tiny DPO/RLVR
  pending/completed resume and explicit typed-state identity migration.
- [Day26 actual response distillation](../notebooks/day-26/03_response_level_distillation.ipynb):
  nine cells/six figures; actual larger-tiny teacher and same-parent complete/
  answer-only students, printed-step versus final-answer contradictions.
- [Day26 critique/revision](../notebooks/day-26/04_critique_revision_and_acceptance.ipynb):
  eight cells/six figures; programmatic state transitions, retained actual source
  candidates, harmful ties, natural-stop hardening and serialized-budget boundaries.
- [Day26 student prefix/context](../notebooks/day-26/05_student_prefix_and_teacher_context.ipynb):
  nine cells/seven figures; actual neural students, finite authored teacher,
  independent prefix/KL/context factors, strict-positive tail detail and cohort gates.

The [manifest](course_manifest.json) is authoritative for machine-readable
routes and dependencies:60core, one optional recurrence and fifteen accepted
extensions. Registration by itself still cannot establish acceptance.
The [isolated CPU acceptance route](../book/appendices/d-reproduction-and-environments.md)
checks a declared subset with real kernel identities. Separately, the
[current checkpoint](../experiments/reports/2026-10-04-course-upgrade-checkpoint.md)
records all76 fresh CPU references across two batches,530 executed cells and206
figures, preserving four unfinished learner cells. This new evidence does not
rewrite the historical full61-session check or imply a Mac/hosted/model-scale pass.

Reports are indexed in [the evidence matrix](EXPERIMENT_MATRIX.md). New sessions
need fresh execution, adjacent solutions, original plots and links before entering
the count. A direct-conversation visual pilot is a separate DXI-18
deliverable, not something static notebook figures already satisfy. The
[selection pilot](../visuals/interactive/README.md) now passes local CPU-browser
control/state tests; actual inline-host and learner assessment are not inferred.

Each chapter normally contains three focused notebook sessions:

1. **Mechanism microscope** — derive and implement the smallest transparent
   version; expose shapes, intermediate values, and invariants.
2. **Perturbation and failure** — violate one assumption, compare a broken
   variant, or change a controlled input; diagnose the consequence.
3. **Integration and evidence** — connect the mechanism to a small model,
   dataset, or evaluation contract; separate observation from interpretation
   and state what the result cannot prove.

A chapter may use two sessions when the mechanism is compact or more when an
important architecture and its experiment need separation. Chapter 5 has an
explicit learner-requested layer-by-layer exception: ten core sessions across
Days 5–7 and an optional recurrent-depth extension. Sessions should be
small enough to complete interactively. Several focused notebooks are preferred
to one long notebook that mixes unrelated ideas.

Every exercise or prediction checkpoint is immediately followed by a clearly
labeled, runnable reference solution and a concise explanation. The learner
attempts the checkpoint first, but routine syntax lookup must not interrupt the
conceptual discussion.

Reusable computation belongs in `src/dongxi_llms/`. Notebook claims remain
exploratory until important results are reproduced by tests and, when empirical
claims matter, an experiment specification and report.

## Implemented pathway by chapter

| Chapter | Mechanism microscope | Perturbation and failure | Integration and evidence |
|---:|---|---|---|
| 1. Evidence Before Optimization | Build a run identity and claim ledger | Change one uncontrolled variable and expose an invalid comparison | Reconstruct a smoke-test claim from manifest, metrics, and exit status |
| 2. Text, Tokens, and Embeddings | Trace Unicode → bytes → BPE merges → IDs | Compare multilingual compression and unknown/byte fallback behavior | Trace embedding lookup, contextualization, and both embedding-gradient paths |
| 3. Learning the Next Token | Logits → stable softmax → NLL and $p-q$ | Break causal shifting and loss-mask normalization | Learn a known conditional distribution from repeated one-hot targets |
| 4. Attention and the Causal Information Boundary | Build scaled causal attention and verify its invariants | Break mask placement and score scaling; inspect gradient paths | Prove cached and uncached decoding equivalence for an unchanged prefix |
| 5. Building a Modern Decoder | Assemble residual attention and feed-forward blocks | Compare normalization, position, activation, and attention variants | Build `DongxiGPT`, account for parameters/FLOPs/memory, then test optional recurrent depth |
| [6. Pretraining as a Controlled System](../notebooks/day-08/README.md) | Document windows, budgets, correctly weighted accumulation | AdamW recurrence, schedule, clipping, dtype casts and memory ledger | Fixed validation/recovery plus three [Day 9 saved-evidence and mask lessons](../notebooks/day-09/README.md) |
| 7. Evaluation Is a Contract | Implement metrics, frozen splits, and uncertainty | Reveal sampling variance, leakage, and misleading aggregate scores | Compare fixed checkpoints with slices and qualitative error analysis |
| 8. Instruction Data as an Interface | Serialize roles and trace labels/loss masks | Break chat templates, packing boundaries, or assistant masking | Inspect a provenance-aware data mixture and its effective token weights |
| 9. Supervised Fine-Tuning | Trace SFT loss and gradient flow through one batch | Ablate masks, mixture weights, or adaptation choices | Compare base and SFT checkpoints for gains, regressions, and uncertainty |
| 10. Preferences and Reward Models | Derive Bradley–Terry probabilities and reward gradients | Explore disagreement, position bias, and reward miscalibration | Train and evaluate a tiny reward model with adversarial slices |
| 11. Direct Preference Optimization | Compute chosen/rejected sequence log-probabilities and derive DPO | Perturb $\beta$, references, masks, and length treatment | Run a controlled DPO comparison against SFT and frozen evaluation |
| 12. Language Generation as a Policy | Derive REINFORCE and inspect token/sequence credit | Compare no baseline, learned baselines, RLOO, and PPO clipping | Measure variance, KL, entropy, and reward under a small policy update |
| 13. Group-Relative Policy Optimization | Construct grouped rollouts, relative advantages, and GRPO loss | Expose zero-variance groups, verifier errors, and token-weighting choices | Run a small RLVR update and inspect reward, KL, entropy, and outputs |
| 14. When Optimization Goes Wrong | Simulate reward hacking, entropy collapse, and length bias | Break rollout freshness, synchronization, or monitoring assumptions | Diagnose a stored failure from telemetry without selecting only favorable evidence |
| 15. Distill, Evaluate, and Defend | Compare sampling, self-consistency, best-of-$N$, and distillation targets | Expose selection bias and capability regressions | Reconstruct checkpoint genealogy and defend the final comparison under uncertainty |

## Activation and maintenance

- At the start of a chapter, refine its pathway into named notebook
  sessions and add a local `README.md` under the relevant notebook directory.
- The 2026-10-04full-course request authorizes all future pathways now;
  never count an empty placeholder as a built lesson.
- Link finished sessions from the chapter, worked solutions, day artifact, and
  `notebooks/README.md`.
- Execute every reference path in the declared environment before marking a
  session ready.
- Preserve learner-entered predictions and outputs carefully; do not overwrite
  them during course maintenance without explicit permission.

## Day 4 activation

Day 4 activates three sessions under `notebooks/day-04/`:

1. `01_causal_attention_forward.ipynb` — score, scale, mask, softmax, value
   mixture, and prefix-invariance checks;
2. `02_attention_gradients_and_failures.ipynb` — autograd through Q/K/V,
   routing versus value branches, missing scaling, and incorrect mask placement;
3. `03_kv_cache_equivalence.ipynb` — prefill/decode state growth, cached versus
   uncached equality, request-local lifecycle, and the compute-memory trade.

Their detailed learning sequence is maintained in `notebooks/day-04/README.md`.

## Chapter 5 activation — revised 2026-09-06

The canonical detailed plan is `notebooks/day-05/README.md`. The learner asked
for separate hands-on treatment of important architecture layers rather than
combining residual connections, normalization, and MLPs into one session.

- Day 5: embeddings/positions; multi-head attention; residual stream;
  LayerNorm/placement; positionwise MLP; decoder assembly; one-batch learning.
- Days 6–7: RMSNorm/SwiGLU; RoPE; GQA/QK normalization/cost accounting and modern
  decoder integration, followed by an architecture defense.
- Optional Day 7 extension: fixed recurrent depth, linked to `ARCH-LOOP-001`.

On 2026-09-06 the learner explicitly requested building the entire pathway ahead
of live study. All eleven notebooks are now built and reference-verified: 93 code
cells across fresh kernels, with adjacent solutions, controlled changes, and
38 passing repository tests. This is a user-requested exception to incremental
creation, not a change to learner completion. See the Chapter 5 companion lab,
solution guide, and experiments/reports/2026-09-06-decoder-notebooks.md.

On 2026-09-07 the Day 5 foundation was synthesized into
[`Chapter 5`](../book/chapters/05-building-a-modern-decoder.md), with twelve
conceptual exercises and worked answers. Seven baseline notebooks are linked
at the point of use. The visual reference revision has 152 code cells and 48
figures across all eleven notebooks. Days 6–7 extend the same chapter; reference
execution and written prose do not complete learner practice or the defense.

The Day 6 modern-mechanism continuation is now written in Chapter 5 sections
5.11–5.22, with worked answers 13–24. The three existing Day 6 sessions were
rechecked in fresh kernels (43 code cells, 13 figures); no duplicate notebooks
were created. Guided learner sessions remain distinct from this preparation.

Day 7 now has a core architecture-defense notebook in addition to the existing
optional recurrence notebook. This brings Chapter 5 to twelve sessions. Study
`day-07/02_architecture_defense.ipynb` before `01_recurrent_depth.ipynb`; preserve
the older filename for stable links. The core session integrates actual shape
traces, budgets, gradients, controlled failures, and a comparison proposal. It
does not execute a trained recurrence comparison or certify learner mastery.
