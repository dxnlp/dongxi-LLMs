# Reference Repository Audit and Course Gaps

Current applicability correction, 2026-10-05: a historical upstream warning
is not automatically a current Dongxi gap. The DGX unified-memory hang report
[PyTorch #174358](https://github.com/pytorch/pytorch/issues/174358#issuecomment-4246197563)
was closed after the maintainer no longer reproduced it on their updated stack;
[NVIDIA's July release](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)
also reports improved GB10 OOM handling. Read-only local checks show DGX OS7.5.0,
kernel6.17.0-1031-nvidia, driver580.173.02 and Torch2.13.0+cu130. These support
retiring the assumption that the historical hang still requires a custom fix
here; they are not an intentional OOM stress-test or proof of every memory edge
case. No new watchdog, allocator replacement or recovery framework is required
by this old warning. Ordinary bounded runs remain prudent on finite shared RAM.
The current pilot's terminal-logging timeout is a separate failure, not evidence
that the old GPU OOM bug persists. Assess each proposed gap against current
upstream fixes, installed versions and actual local relevance before building.

Reviewed 2026-10-04 against Dongxi commit `ac203efca4ca4a5b42cb7e1a04dbe3a923559ab4`.
The requested comparison is an audit of public repository material, not a claim
to have read the complete commercial reasoning book.

The main finding is a missing bridge between good numerical lessons and a
complete, measured model-development workflow. Dongxi already teaches most
central objective mathematics. Adding algorithm names alone would not close
that gap. The [implementation plan](COURSE_IMPROVEMENT_PLAN.md) defines eighteen
bounded work packages and their completion checks; the
[machine-readable ledger](course_improvements.json) separates current completion
from these baseline findings. Initially all eighteen were planned. The subsequent
first CPU implementation pass completes DXI-08/13 and partially implements
DXI-01/02; their reports are in the [evidence matrix](EXPERIMENT_MATRIX.md).
This audit describes the pinned baseline, not a claim that repaired defects
still exist in current source.

## Inspected sources

| Reference | Inspected revision | Material inspected |
|---|---|---|
| [RLHF Book](https://github.com/natolambert/rlhf-book/tree/eecc49e1e1daf1be670d7242eb090240b5bb0f04) | `eecc49e1e1daf1be670d7242eb090240b5bb0f04` | Chapter sources, code pathways, objective implementations, configuration, licenses |
| [Reasoning From Scratch](https://github.com/rasbt/reasoning-from-scratch/tree/a788466dc85cfe8617b6c0b809ed4ab9084485de) | `a788466dc85cfe8617b6c0b809ed4ab9084485de` | Public notebooks, reusable code, evaluation/generation/training scripts, appendices, tests, license |
| Dongxi LLMs | `ac203efca4ca4a5b42cb7e1a04dbe3a923559ab4` | Chapters, solutions, labs, roadmap, notebook routes, runners, tests, measured reports |

References were fetched into separate temporary inspection directories. No
third-party code, prose, diagrams or benchmark results were imported into the
course. The audit did not install packages, load weights, generate model
responses, start services or run training.

## What is already strong

There are fifteen coherent chapters, worked solutions, sixty-one visual
notebooks and a full twenty-eight-day route. Their
[build report](../experiments/reports/2026-10-04-complete-course-build.md)
records fresh CPU references and independent tests, not learner mastery.

The existing lessons already cover stable likelihoods, attention and cache
invariants, assistant masks, Bradley–Terry gradients, disagreement, calibration,
DPO, REINFORCE/RLOO, clipping, KL estimator boundaries, GRPO reductions,
verifier failures, selection and distillation. The negative DPO and tiny RLVR
outcomes are useful evidence and must remain intact.

The first TinyStories learning run is actual GPU evidence. The later Qwen SFT,
DPO and RLVR entry points are prepared but unexecuted. Historical Qwen smoke
and embedding inspection are not evidence that those new campaigns have run.

## Confirmed implementation integrity issues

### SFT run identity is incomplete

The [SFT lab](../book/labs/09-supervised-fine-tuning.md), under “Output and
genealogy,” says that `config.json` records source hashes. The
[runner](../scripts/run_chapter09_spark_sft.py) configuration records arguments,
Torch/Transformers versions, data/template hashes, GPU, dtype and attention
backend, but does not add source hashes, Git identity, Python, GPU driver or a
lock-file identity.

This is a documentation/implementation mismatch, not evidence of a failed
executed Qwen run. DXI-01 must align the contract and implementation before
using the runner to support a reproducible comparison.

Inspection anchors at the audited revision: lab line55; runner configuration
lines146–154 and genealogy lines243–247.

### Parent tokenizer compatibility is not enforced

The [DPO lab](../book/labs/11-direct-preference-optimization.md) tells the reader
to validate vocabulary and revision compatibility with the actual SFT
checkpoint. The [DPO runner](../scripts/run_chapter11_spark_dpo.py) loads the
supplied tokenizer independently and restores the parent's saved template,
without comparing token mappings or actual tokenizer artifacts with the parent.
The SFT genealogy also omits tokenizer revision/fingerprint.

Matching vocabulary size and template is insufficient: the same IDs could map
to different strings. This is a confirmed missing check from source inspection;
no deliberately corrupted real checkpoint was executed during this audit.
DXI-01 needs a same-size permuted-vocabulary rejection test, legitimate save/load
checks, and a clear policy for legacy checkpoints lacking a fingerprint.

Inspection anchors at the audited revision: DPO lab line34; independently
loaded tokenizer/template restoration in the runner at lines159–161.

## Coverage comparison

“Partial” means the concept is present but a substantial implementation or
evidence layer is absent. “Unexecuted” does not mean the code is missing.

| Area | Existing Dongxi material | Gap supported by inspected source | Work packages |
|---|---|---|---|
| Model evaluation | Ch7 contracts, pass@k, paired uncertainty and slice fixtures | No shared checkpoint-to-raw-generation-to-score runner or genuine reasoning baseline | DXI-02,04 |
| Mathematical grading | General string normalization; strict integer/EOS RLVR verifier | No explicitly bounded extraction/equivalence pipeline for broader mathematical answers | DXI-02 |
| Preference collection and judges | Ch10 disagreement/calibration; feature-based synthetic labels | No text comparison collection/audit contract or offline judge-bias intervention | DXI-08 |
| Neural reward models | Ch10 linear model on known quality/length/format features | No trainable text backbone/reward head, executable outcome/process supervision | DXI-07 |
| Classical learned-reward RLHF | Reward and policy mechanisms in separate modules | No preferences→frozen learned reward→policy update→independent quality experiment | DXI-07,10 |
| PPO critic and credit | Ch12 exact detached oracle baseline and clipping | No learned critic, short-sequence returns/GAE, termination/bootstrap lesson | DXI-10 |
| Preference retention | Ch11 preserves margin gain with held-out degradation | No remedy/control experiment using chosen-NLL or rehearsal and independent retention | DXI-11 |
| Reasoning RLVR | Ch13 tiny update and optional integer-only Qwen smoke | No reasoning-capable branch, unsaturated positive control, new-template evaluation | DXI-04,03 |
| Inference-time scaling | Ch15 voting/ranking/oracle categorical simulation | No shared actual candidate pools or measured accuracy–token–latency frontier | DXI-05 |
| Self-refinement | Selection is explained | Critique→revision→acceptance mechanism and regression accounting absent | DXI-06 |
| Synthetic and selected data | Ch8 data interfaces; Ch15 rejection principles | No resumable teacher-attempt ledger, filter audit or matched random-selection→SFT comparison | DXI-09 |
| Reasoning distillation | Ch15 hard/soft distinction and three-logit forward-KL experiment | No actual response-training adapter/sequence experiment or pretrained transfer evidence | DXI-15 |
| On-policy distillation | Student prefixes mentioned | No state-distribution intervention, teacher-context boundary or approximation microscope | DXI-16 |
| GRPO variants | Normalization toggle; denominator and Dr. GRPO discussion | No identical-rollout gradient/denominator/filtering matrix or component-reward comparison | DXI-12 |
| Sampling likelihood | Ch14 correctly warns about transformed sampling; local RLVR uses full support/T=1 | No runnable mismatch/support-failure/repair lesson | DXI-13 |
| Rollout systems | Ch14 budget/version schematics; equal-prompt uncached adapter | Variable-length batched generation, per-row stopping/cache parity and resume evidence absent | DXI-14 |
| Reproduction | Spark CPU references and source/math/navigation checks | Mac execution unverified; loose portable ranges, no repository CI, some hardcoded route assumptions | DXI-17 |
| Coherent synthesis | Fifteen-chapter argument and mechanism companions | Missing explicit reasoning-compute bridge and real-checkpoint evidence/card integration | DXI-18,03 |

## Why these gaps matter

### Evaluation must precede another training campaign

The reasoning reference gives separate
[evaluation code](https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/reasoning_from_scratch/ch03.py)
and an [advanced-parser comparison](https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/ch03/03_advanced-parser/README.md).
Our Chapter7 supplies the statistical reasoning, but not that operational
bridge. Parsing, mathematical equivalence and task validity must be separate
decisions. Unsupported or unsafe inputs must be visible, not silently converted
into favorable scores.

The current SFT copy/reverse/extraction, DPO location extraction and RLVR
addition fixtures are different populations. Their scalar scores cannot form a
single capability-improvement ladder. A shared evaluation panel may contain
separate declared task slices; branches must be compared on the same relevant
items, with task-specific scores rather than an invented common average.

### Reward learning and policy learning need to meet

RLHF Book's [reward-model pathway](https://github.com/natolambert/rlhf-book/blob/eecc49e1e1daf1be670d7242eb090240b5bb0f04/code/reward_models/README.md)
and [policy-gradient chapter](https://github.com/natolambert/rlhf-book/blob/eecc49e1e1daf1be670d7242eb090240b5bb0f04/book/chapters/06-policy-gradients.md)
make the neural reward-head and critic/GAE gaps concrete. Dongxi's synthetic
feature model is valuable precisely because its shortcut is inspectable; it
does not establish text representation learning or annotation reliability.

An original small decoder lesson can expose reward-head pooling and a learned
critic without immediately requiring an expensive pretrained sweep. Connect
the fitted reward to an independently evaluated policy to show where higher
proxy reward can become worse behavior.

### Reasoning improvement includes inference, not only weight updates

The reasoning reference supplies
[inference-scaling scripts](https://github.com/rasbt/reasoning-from-scratch/tree/a788466dc85cfe8617b6c0b809ed4ab9084485de/ch04/02_math500-inference-scaling-scripts)
and a [self-refinement loop](https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/reasoning_from_scratch/ch05.py).
Dongxi should explicitly compare an unchanged model with a different
answer-producing procedure against a changed checkpoint under the same
procedure. Longer visible reasoning is not itself evidence of correct or
faithful reasoning.

Use offline candidate/transition records first. Then measure generated tokens,
latency and independent correctness under approved Spark budgets. Include
right→wrong revisions and selection failures, not only successful examples.

### Data selection and distillation need response-level experiments

RLHF Book's [rejection-sampling pathway](https://github.com/natolambert/rlhf-book/blob/eecc49e1e1daf1be670d7242eb090240b5bb0f04/code/rejection_sampling/README.md)
compares selected and random controls. The reasoning reference's
[teacher-data route](https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/ch08/02_generate_distillation_data/README.md)
and [distillation training route](https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/ch08/04_train_with_distillation/README.md)
expose the missing sequence bridge.

Reuse our SFT implementation with audited teacher-response records. Separate
teacher inference cost, selection cost and student training. A filtered corpus
changes coverage and difficulty as well as label quality. Selected responses
need a matched random control and prompt-group split integrity.

### Algorithm variations should expose mechanisms, not grow a name list

Both references provide richer objective menus:
[RLHF losses](https://github.com/natolambert/rlhf-book/blob/eecc49e1e1daf1be670d7242eb090240b5bb0f04/code/policy_gradients/loss.py),
[advanced reasoning GRPO](https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/ch07/03_rlvr_grpo_scripts_advanced/README.md).
First compare one change at a time on identical saved rollouts: reduction,
group scaling, filtering, clipping, component rewards and sampling support.
Do not call arbitrary combinations DAPO, Dr. GRPO or another named method.

References are not correctness oracles. For example, the RLHF snapshot's
generation defaults and raw-logit rescoring warrant a behavior-distribution
audit before reuse. That is an audit concern, not a reproduced upstream failure.
Our existing temperature-one/full-support RLVR convention is a strength to keep.

## Evidence and execution limits

- Current model-scale SFT/DPO/RLVR protocols have not been run. “CLI loads” and
  CPU file-contract tests are not CUDA integration or capability evidence.
- The current tiny GRPO experiment starts with saturated training greedy
  accuracy and fails its four held-out items. Preserve that negative result;
  add a separate positive control rather than relabeling it.
- Optional Qwen RLVR currently disables thinking, limits outputs to sixty-four
  tokens and uses four train/four held-out integer prompts. It is not a broad
  reasoning benchmark.
- DPO/RLVR save optimizer/RNG information but lack an independently verified
  exact-resume CLI. Sampled resource guards are not continuous monitoring or
  hard deadlines for blocking model operations.
- CPU references were executed on Spark. Mac compatibility, a fresh live
  GitHub math render and model-scale transfer remain unverified.
- Upstream reported results, runtime figures and checkpoint quality are
  upstream evidence, never Dongxi measurements or Spark runtime forecasts.

## Scope and licensing decisions

Preserve the fifteen-chapter/twenty-eight-day core. Improve existing chapters
and add clearly marked enrichment sessions; do not create another mandatory
day for each reference topic. Self-refinement and on-policy distillation are
selected extension work packages, not required detours in the learner's live
Day9 route.

Full algorithm surveys, tool-agent environments, a transparent whole-Qwen
weight loader, large-model sweeps, broad model-character studies and multimodal
post-training are candidate later projects, not prerequisites for this upgrade.
The plan explains why they are not automatically adopted.

The RLHF repository distinguishes
[chapter CC-BY-NC-SA-4.0 from other-material MIT terms](https://github.com/natolambert/rlhf-book/blob/eecc49e1e1daf1be670d7242eb090240b5bb0f04/README.md#license);
some third-party code retains its own Apache attribution, and some book images
are not licensed for reuse. The reasoning repository's public code is
[Apache-2.0](https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/LICENSE).
Prefer independently written explanations, code, fixtures and figures with
pinned references. No course license is chosen by this audit.

## Original audit handoff (planning snapshot)

At the end of the audit the goal was active and all eighteen packages were planned;
none was complete merely because the audit or plan existed. The initial route was
DXI-01, then DXI-02's offline evaluator/verifier instruments. Heavy inference,
training, external API generation, publication and new animation rendering
remain separately approved. The learner remains on Day9.

Current implementation status and the exact next production action are in the
[ledger](course_improvements.json) and [handoff](handoffs/CURRENT.md); the
verification below records the earlier planning-only revision.

The earlier build manifest is historical evidence for its recorded revision.
This planning revision does not replace its measured outcomes or re-label
unexecuted protocols as results.

## Historical verification of the planning revision

- Both external checkout identities match the pinned revisions above.
- The eighteen-item JSON ledger loads, IDs are unique, dependencies are acyclic,
  and every package is planned with an empty implementation-evidence list.
- Local links and plan coverage pass. Existing navigation reports15chapters,
 15solution guides,4appendices and61notebooks with zero issues.
- Book mathematics source checks pass:54Markdown files and1,053expressions,
  zero issues. This is not a fresh live GitHub render.
- All146existing regression tests pass on Spark CPU. No source-code behavior
  was changed and no new model-scale run was performed.
- An independent read-only review found no blocking factual/status errors.
  The current learning-memory Day9 row and handoff had stale planning-only
  statements; current summaries were reconciled with the completed-run report
  and pushed commit, while historical experiment records were preserved.
- Changes are saved locally; no commit/push is implicit in the audit request.
