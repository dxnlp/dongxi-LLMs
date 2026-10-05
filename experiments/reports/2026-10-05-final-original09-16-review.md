# Independent review of the original DXI-09–16 criteria

Review date: 2026-10-05. Reviewer: `/root/audit_internal_evidence`.

Conclusion: **all 38 original criteria are supported at their declared bounded
mechanism/CPU/material boundary**. No blocking omission or unsupported native
capability claim was found in these eight packages. This is a new read-only
acceptance mapping, not a new experiment, test pass, notebook execution or final
course-wide closure. It does not mark the eighteen-package goal complete.

The review inspected the literal criteria/dependencies, current mechanism source
and focused test definitions, registered notebook source, chapter/lab/worked
routes, retained raw-report summaries and independent acceptance/replay records.
It did not run models, rerun empirical references or tests, load checkpoints,
read/hash large weights, access the network, alter running jobs, change producer
source, or modify any package's criteria/dependencies/status. Current full CPU
verification after the sequential native queue remains a separate task.

## 1. Original baseline and dependency preservation

The retained baseline is
[original-acceptance-before-integration.json](2026-10-05-native-base-profile/original-acceptance-before-integration.json),
SHA-256 `6cb9caee1d87b36f4a1ae6a9fbb5ef36889ad9f6e198cf822aeb11ec5bf2be2b`.
Direct `jq` equality checks against
[course_improvements.json](../../docs/course_improvements.json) returned true
for **all eight complete acceptance arrays and all eight dependency arrays**.
The mapping below quotes their literal criterion strings in original order.
All 42 evidence-file references registered for DXI-09–16 exist locally; existence
alone was not treated as evidence of success.

| Package | Unchanged dependencies | Dependency/mechanism boundary inspected |
|---|---|---|
| DXI-09 | DXI-01, DXI-02, DXI-08 | Bound attempt/source/interface identities; prompt/source-group audits; gold-blind selection versus independent student grading. Actual instruction encoder/collator/SFT loss are reused. |
| DXI-10 | DXI-07, DXI-08, DXI-13 | Saved balanced/confounded character-reward fits, disjoint actual actor/reward encodings, matched conditional collection/scoring likelihoods and detached old/reference signals. |
| DXI-11 | DXI-01, DXI-02, DXI-08 | Shared decoder/SFT/DPO utility, frozen reference/cache binding, authored preference/noise/length intervention and independently graded generation. |
| DXI-12 | DXI-04, DXI-13 | Constructive symbolic correctness control, retained sampled groups, declared conditional support, matched old/current score alignment and independent derivative calculations. |
| DXI-13 | none | Standalone exact categorical probability/mask instrument; no hidden completion dependency added. |
| DXI-14 | DXI-01, DXI-02, DXI-13 | Explicit model/prefix/interface/environment identity, response/stop/work records and matched behavior support before tiny DPO/RLVR recovery. |
| DXI-15 | DXI-01, DXI-02, DXI-09 | Shared run identity and generation journal; DXI-09 digest/coordinate helpers and existing causal SFT loss; independent answer/step scorer. The new task is not forced into the earlier color-task schema. |
| DXI-16 | DXI-13, DXI-15 | Aligned support/vocabulary and EOS/cap/state masks; hard-response versus conditional soft-target objectives remain distinct. Shared attempt provenance and actual student generation, not a substituted three-logit learner. |

The prerequisite packages' scoped evidence is retained in the ledger, including
the independent native identity/evaluation review for DXI-01/02 and the
preference/reward/control receipts for DXI-04/07/08. This review does not newly
rerun or certify every prerequisite's empirical history. The inspected reusable
`sft_lab.py`, `instruction_data_lab.py`, `decoder_lab.py` and `dpo_lab.py` hashes
match their identities recorded by these package acceptances. The existing
`distillation_lab.py` still hashes to
`1bb80abc3b9135d92577dcf827c3f5237e23ceb45759d7fd155a103380bc7c8c`.

No dependency on a compulsory native SDPO/distillation campaign, extra
multi-seed model sweep, successful Mac execution, hosted CI success or repeated
14,000-update story run has been added. Those are not original DXI-09–16 criteria.

## 2. Criterion-by-criterion mapping

“Supported” below means current inspected implementation plus the cited retained
measurement/independent verification supports the **declared** objective. It
does not mean the learner completed the exercise or a pretrained capability was
demonstrated. The focused test counts are retained executed evidence, not tests
run during this review.

### DXI-09 — teacher data and matched rejection sampling

Primary source:
[teacher_data_lab.py](../../src/dongxi_llms/teacher_data_lab.py);
[tests](../../tests/test_teacher_data_lab.py).
Retained [report](2026-10-04-teacher-data.md) and
[independent root acceptance](2026-10-04-teacher-data-root-acceptance.json)
record all nine fits/outputs and an actual independent replay/regrading.

| # | Literal original criterion | Evidence and disposition |
|---|---|---|
| 1 | Retain all attempts, rejections and reasons, trace/final-answer fields, teacher/sampler/verifier identity and content terms; resumption cannot duplicate records. | **Supported.** `freeze_contract` line97, `TeacherJournal` line130 and `execute_attempt` line261 bind those fields. The retained journal paused at9 committed records, resumed to108 unique attempts and did not append duplicates on another resume. Focused controls cover duplicate/concurrent writers, changed contracts, committed corrupt lines and preserved partial tails. Exactly-once committed records are not mislabeled exactly-once physical/API execution. |
| 2 | Audit source-group split collisions, difficulty/length coverage, empty/overlength/malformed answers and retry costs. | **Supported.** `validate_protocol` line58 checks normalized prompts, actual student IDs and underlying groups; `filter_pool` line321 retains all applicable reasons. There are72 rejected/36 accepted records, including12 correct and24 wrong accepted candidates;12 errors and12 retry attempts are charged. Multi-reason counts need not partition rejections. |
| 3 | Compare identical candidate pools and fixed scorers under top-per-prompt versus seeded random controls; report global-selection coverage separately. | **Supported.** `verifier_score` line316 reads prompt/candidate, not the reference/mode fields; `select_datasets` line380 verifies the frozen pool and scorer. Top, seeded random and length-random share its identity. Global top3/6/12 coverage is0.25/0.5/1 and remains a separate audit, not extra fitted arms. Tests cover scorer isolation, pool fingerprints, order invariance and no-coverage refusal. |
| 4 | Match sample count, prompt coverage and supervised-token budget where possible; explicitly disclose when they cannot all be matched. | **Supported.** All arms have12 samples/12 prompts. Top and length-random each have42 targets/update and3360 over80 updates; unstratified random has46/3680. The primary mismatch is explicit. Every length stratum has two eligible candidates with both correctness values, so the sensitivity matches target exposure without a hidden correctness fallback. |
| 5 | Independent student held-out evaluation is separate from selection score; reuse the existing SFT stack and preserve negative outcomes. | **Supported.** `training_batch` line432 uses `encode_messages`/`collate`; `fit_student` line437 uses existing `token_loss_sum`. `evaluate_student` line470 independently generates/grades. Nine actual6120-parameter students,720 updates and31200 target presentations remain; all nine fail the greedy polite-prefix control. The root receipt regrades1080 post-training records and preserves360 unique baseline records. |

The23-test/fresh8-cell,6-figure acceptance is scoped CPU evidence. The teacher
is explicitly programmatic, with zero model forwards/API calls. Its serialized
word costs are not mislabeled LLM inference tokens. The historical spec's overly
broad statement that references are “only evaluation” is explicitly clarified:
authored-reference consistency is checked before collection, while the selector
cannot read reference labels. The original spec and failed alias receipt remain
preserved, not rewritten.

### DXI-10 — learned critics and a learned reward policy loop

Source [critic_policy_lab.py](../../src/dongxi_llms/critic_policy_lab.py),
[tests](../../tests/test_critic_policy_lab.py),
[report](2026-10-04-critic-policy.md),
[independent acceptance](2026-10-04-critic-policy-acceptance.json).

| # | Literal original criterion | Evidence and disposition |
|---|---|---|
| 1 | Verify analytically enumerable returns and declared lambda endpoints; distinguish EOS terminal states, caps/time-limit truncation and padding. | **Supported.** `gae_targets` line143 is actually decorated `torch.no_grad`; EOS terminates continuation while a nonterminal cap bootstraps. Independent closed-sum/lambda0/lambda1 controls, cap-as-EOS failure and padding/post-EOS rejection are in tests lines37–76. The task's horizon4 versus collector cap3 distinction and separate forced cap4 diagnostic remain explicit. |
| 2 | Critic has its own loss; policy advantages are detached; test policy/critic/reward gradient boundaries. | **Supported.** `actor_loss` line175 detaches advantages/old scores; `critic_loss` line186 detaches targets. `run_arm` line348 uses separate actor/critic optimizers/backbones and frozen reward parameters. Tests lines77–90 and153 verify independent gradient paths; actual first-backward reach and reference/reward hashes are retained. |
| 3 | Compare oracle, learned and noisy critic under declared rollout/update budgets, without presenting a surrogate as the complete PPO stack. | **Supported.** Twelve frozen rows: three seeds×balanced oracle/learned/noisy plus confounded learned,40 actor updates per row,4 prompts×8 paths/update. The retained totals are15360 paths,480 actor steps and240 learned critic steps. Equal actor/rollout clocks are not equal critic work. One fresh update per rollout has initial ratios1; this is explicitly not production multi-epoch PPO. |
| 4 | Fitted reward is frozen and reproducible from saved inputs; report proxy reward versus independent quality and confounded/balanced reward controls. | **Supported.** `fit_rewards` line190 fits/exports/reloads both train-only character rewards with independent expected byte identities, train-derived scaling and fixed inputs. Root acceptance reports both reloads exact and all12 reward/actor/critic states, histories, raw paths and evaluation panels replayed. The independent requested-color/natural-EOS quality rule never selects training/checkpoints; capped/style-collapse failures remain in the report. |
| 5 | Preserve the existing exact-gradient/oracle lesson as a reference, not a substitute for learned-value evidence. | **Supported.** Original `policy_gradient_lab.py` and Day19/Day20 exact-value references remain linked before the actual actor3428/separate critic3105-parameter experiment. Chapter12§12.10–12.14 and the lab distinguish the oracle microscope from learned-value errors. |

The16-test acceptance and fresh7-cell/5-figure notebook are retained. An
enumerated conditional grammar and authored independent quality rule do not
constitute pretrained English PPO or human reward evidence. Negative held-out
and stopping outcomes remain visible; no critic winner is selected afterward.

### DXI-11 — DPO retention and preference controls

Source [dpo_retention_lab.py](../../src/dongxi_llms/dpo_retention_lab.py),
[tests](../../tests/test_dpo_retention_lab.py),
[report](2026-10-04-dpo-retention.md),
[verification](2026-10-04-dpo-retention-verification.json).

| # | Literal original criterion | Evidence and disposition |
|---|---|---|
| 1 | Alpha=0 recovers the current DPO objective; verify gradients against the explicit combined loss. | **Supported.** `combined_loss` line169 reuses existing `dpo_loss`; zero auxiliary coefficients omit branches. Tests lines105–151 check exact parameter-gradient recovery and the explicit DPO+alpha chosen-token NLL+gamma rehearsal loss. The material correctly states that alpha0 alone does not remove a nonzero gamma rehearsal intervention. |
| 2 | Plot absolute chosen/rejected likelihoods as well as margins, unrelated-task retention and held-out generation scores. | **Supported.** `fit_arm` line300 stores absolute post-update pair scores/margins; `evaluate` line242 separately grades copy/parity free generation. Day18's five plots include those trajectories, unrelated-task/held-out panels and cost. Clean plain-DPO mean chosen likelihood falls from−0.001810 to−4.640873 while the margin rises to10.587275; no margin-only success claim replaces that observation. |
| 3 | Freeze reference, templates and evaluation; disclose preference data source, label noise, length mismatch and supervision/rehearsal budget. | **Supported.** Original contract validates authored sources, three seeds, fixed swapped indices1/3 and clean/chosen-long/matched-long conditions. `pair_cache_hash` line210 binds weights/tokens/template/masks/dtype/scores; reference remains frozen with no gradients. All48 fits disclose3840 preference updates, paired-target exposure, reused chosen supervision and extra rehearsal forwards/targets. |
| 4 | Keep current negative evidence; do not choose a favorable coefficient after inspecting final-test outcomes or promise a universal repair. | **Supported.** Four coefficient arms and all48 final outcomes are predeclared and retained. Noisy winner imitation, verbosity, weak warm parity and failed held-out copy transfer remain. `run_retention` line359 rejects undeclared seed choices and labels changed micro budgets nonprimary. The original negative DPO evidence digest is preserved; the chapter explicitly denies universal retention repair. |

The retained verification has16 focused checks,48-row numerical replay and a
fresh8-cell/5-figure notebook. Later native chosen/DPO recovery receipts are
additional, separate evidence; they are not necessary substitutions for this
original CPU controlled-retention criterion and do not establish quality gains.

### DXI-12 — controlled GRPO objective variations

Source [grpo_objective_controls.py](../../src/dongxi_llms/grpo_objective_controls.py),
[tests](../../tests/test_grpo_objective_controls.py),
[report](2026-10-04-grpo-objective-controls.md),
[closing verification](2026-10-04-grpo-objective-controls-closing-verification.json).

| # | Literal original criterion | Evidence and disposition |
|---|---|---|
| 1 | Compare response mean, global valid-token mean and declared fixed denominator on identical rollouts; verify analytic token gradients and padding invariance. | **Supported.** `reduction_weights` line65 defines real action weights; `objective` line87 selects valid entries before arithmetic; `analytic_logp_gradient` line116 is independent piecewise calculus. All27 seed×variant rows use unchanged first-attempt pools, analytic max error0 and unchanged parameters. Tests cover padded NaNs and fixed denominator not following padded width. |
| 2 | Test centered-only versus standard-deviation-scaled advantages, all-zero/all-one groups and component rescaling/degeneracy. | **Supported.** `component_advantages` line37 compares centered, total population-std and component population-std rules with explicit weights. Tests lines116–168 cover both constant groups, constant components, raw-component versus coefficient rescaling and invalid rewards. Actual gradient scales/directions, not optimizer gains, are measured. |
| 3 | Compare total-reward normalization with component-before-aggregation normalization using independent correctness. | **Supported.** `response_components` line151 separates answer/format/verbosity proxy components from completed correct numeral+emitted-EOS quality. The same27 objective rows/gradient cosines compare normalization orders. The test atline195 preserves proxy/independent-quality disagreements. |
| 4 | Report attempted/selected prompts and valid generated tokens for filtering/resampling; no free-data or equal-update cost claim. | **Supported.** `collect_with_filter` line208 retains every rejected group and bounded retry. Across three seeds,200 attempted responses/402 valid actions contrast with120 selected/261 valid actions. An all-constant test exhausts the cap with no selected groups. Additional full-prefix/rescoring work is separate; filtered data does not replace the fixed objective pool. |
| 5 | Name methods only when defining choices match their primary specifications; full algorithm zoo stays outside this goal. | **Supported.** Chapter13§13.5.1–13.5.2 names reductions/scalings/clipping interventions, not a complete named-algorithm reproduction. Initial ratios1 make clipping variants identical there; a separately labeled constructed off-initial-ratio fixture supplies directional clipping evidence. No optimizer/capability ranking or broad algorithm survey is claimed. |

The18-test independent replay confirms all non-timing fields for all three
seeds and the constructed clipping fixture. Fresh6-cell/5-figure evidence is
retained. No training update occurs in this matched-objective microscope.

### DXI-13 — sampling support and probability accounting

Source [sampling_likelihood_lab.py](../../src/dongxi_llms/sampling_likelihood_lab.py),
[tests](../../tests/test_sampling_likelihood_lab.py),
[initial report](2026-10-04-sampling-support.md),
[hardening report](2026-10-04-sampling-support-hardening.md) and
[hardening evidence](2026-10-04-sampling-support-hardening.json).

| # | Literal original criterion | Evidence and disposition |
|---|---|---|
| 1 | Known distributions reproduce expected ratios; identical matching behavior/target probabilities give ratio1. | **Supported.** Raw(0.55,0.30,0.15), behavior temperature0.5/k2/p0.9 and fixed-support temperature1 target have explicit finite values. Matching behavior/target gives(1,1); the deliberately different-temperature target gives(0.839572193,1.539215686). Tests lines71–92 compare independent analytic gradients with exact enumeration/autograd. |
| 2 | Store temperature/top-k/top-p transformations and behavior log probabilities; demonstrate missing-support and wrong-denominator failures. | **Supported.** `behavior_distribution` line51 and `BehaviorRecord` line85 retain temperature→top-k→renormalize→top-p→renormalize and16 seeded draws/actual selected likelihoods. Correct expected reward0.352941176 versus wrong raw denominator0.269763957 and changed gradients are preserved. Missing0.15 target mass/0.60 reward contribution raises `MissingSupportError`, not a fictitious zero ratio. |
| 3 | Repair a precisely declared objective, with masked renormalization and temperature accounted for; do not claim a retained sampling mask is universally sufficient. | **Supported.** `target_log_probabilities` line31 conditions an explicit fixed support and temperature; full-support collection is the separate repair for the original raw objective. The chapter/notebook explicitly distinguish them. Extreme-logit hardening rejects underflow/unrepresentable retained probabilities and masks excluded terms before arithmetic, without inventing support floors or universal autoregressive correction. |
| 4 | Test response-only/EOS/padding masks and reference versus old-policy detachment; distinguish true terminal from budget truncation. | **Supported.** `response_surrogate` line175 detaches old/advantage/reference and selects valid positions before likelihood arithmetic. `termination_masks` line204 includes a generated EOS, excludes prompt/post-stop/padding, rejects unexplained early ends and labels caps separately. Tests lines131–173 and218–231 cover those contracts, including EOS-as-padding. Bootstrap is explicitly a continuing-task convention. |
| 5 | Keep the current temperature-one/full-support model adapter as the simple baseline. | **Supported.** Test atline18 preserves the utility baseline; the current `qwen_rlvr_lab.py` collection/scoring still uses unfiltered softmax/log-softmax/full-support temperature1 (lines350–390/472 and contract line528). This CPU lesson did not replace it with filtered sampling. |

The old17-test results and failures remain historical; the later21-test
hardening and fresh7-cell/4-figure reference bind the inspected current source.
Equal KL estimator values and unequal frozen-sample gradients are a finite-state
objective lesson, not a complete trajectory/off-policy theorem.

### DXI-14 — batched generation, cache stopping and recovery

Source [batched_cache_lab.py](../../src/dongxi_llms/batched_cache_lab.py),
[tests](../../tests/test_batched_cache_lab.py),
[report](2026-10-04-batched-cache-recovery.md),
[root verification](2026-10-04-batched-cache-recovery-root-verification.json),
[later tensor-byte acceptance](2026-10-05-recovery-tensor-bytes-verification.json).

| # | Literal original criterion | Evidence and disposition |
|---|---|---|
| 1 | Variable-length greedy batched and sequential outputs agree to the declared tolerance up to each row's stop; reordering and finished-row handling are correct. | **Supported.** `Generation` line161 implements real ragged positions/key masks/compact caches and active-row compaction. Three prefixes of lengths2/4/6 agree across cached/uncached single/batch paths, observed selected-score error0 within predeclared1e−10. Reordered request-keyed sampled streams and a separate forced-stop control are tested atlines178–198. Natural greedy rows cap; forced EOS is not presented as learned stopping. |
| 2 | Cache identity includes prefix/model/template/position semantics; padded work and useful output tokens are separately counted. | **Supported.** `Generation.contract/_check/_account` bind those semantics and actual active geometry. For12 useful outputs, cached batch forwards27 positions with6 pads versus uncached90 with24 pads; compact retained cache bytes and valid input counts are separate from whole-process/attention temporary memory. Tests atlines159–206 inspect actual tensors and accounting. |
| 3 | Old-policy/rollout versions cannot silently mix; stale-weight and resume/data-cursor/RNG failures have tests. | **Supported.** `Generation._check` and `PolicySession.update/restore` line349 onward check model/version/prefix, pending old-policy identity, source cursor and likelihood alignment. Stale weights, changed interface, resealed cursor/RNG/cache corruption, wrong references/data and mixed versions are retained rejection controls/tests. |
| 4 | Implement and independently test declared DPO/RLVR recovery before describing checkpoint state as resumable evidence. | **Supported.** Both objectives/two seeds/six updates have actual completed3 and pending-fourth-batch interruption recovery. Next data/rollout, complete history, final policy/reference/Adam/cursor/RNG bytes agree; pending applies without resampling. Independent root v2 migration rechecks actual states. The later30-test tensor-byte receipt preserves13 archived identities/eight continuations while adding BF16 byte fixtures; it is not BF16 neural recovery. |
| 5 | Measured throughput/latency/memory remains pending until an authorized profile; no assumed batching speedup. | **Supported at its explicit boundary.** The bounded tiny CPU protocol authorizes its own five-repeat descriptive timings and cache payload measurements. Cached batch median0.004383510s is slightly slower than uncached0.004216790s; this negative remains. The report denies optimized engine/Spark throughput/peak-memory/large-model inference claims. Later native recovery is not silently reused as a batching-performance profile. |

The v1 nested-container encoding collision, failed geometry expectation,
registry preflight failure, corrected v2 encoding and later raw-BF16 extraction
follow-up are preserved. Old contracts remain rejected, not silently migrated
into a current resume interface. The original fresh notebook had7 cells/5 images;
the current source has10 code cells and additional independently scoped reader
exercises. This review does not relabel the old notebook execution as execution
of the added exercises or a new all-notebook pass.

### DXI-15 — response-level reasoning distillation

Source [response_distillation_lab.py](../../src/dongxi_llms/response_distillation_lab.py),
[tests](../../tests/test_response_distillation_lab.py),
[report](2026-10-04-response-distillation.md),
[verification](2026-10-04-response-distillation-verification.json),
[independent root review](2026-10-04-response-distillation-root-review.json).

| # | Literal original criterion | Evidence and disposition |
|---|---|---|
| 1 | Prompt loss excluded; response/EOS included; template/tokenizer compatibility, truncation and malformed examples explicitly checked. | **Supported.** `batch_examples` line94 leaves prompt/pad labels−100, includes real response EOS, refuses absent/embedded EOS/invalid IDs/context overflow and uses existing one-shift SFT loss. `adapt_teacher` line173 verifies exact contract/token/template/model/parent serialization. Tests lines47–87 and133 cover rehashed incompatibilities, shift/mask/EOS gradients and actual backbone reach. |
| 2 | Use source-group splits and preserved teacher provenance; do not assume teacher-generated rationales are correct/faithful. | **Supported.**18 authored items/14 groups prevent source/problem/encoded-prompt train/test overlap; alias siblings stay together in test. The actual teacher response journal retains144 data attempts and18 selected parent traces. Selection is first naturally ended well-formed trace, not semantic correctness. Actual teacher outputs include80 wrong-step/right-answer and24 valid-step/wrong-answer evaluations; neither printed correctness nor final success is labeled causal faithfulness. |
| 3 | Compare original student, response-SFT student and selection procedure under a common appropriate evaluation contract; keep teacher and student costs separate. | **Supported.** Original student, teacher, same-initial full-response and answer-only students each receive the common eight-pool/free-generation contract. `select` line232 removes gold from selector views and replays first/majority/mean-logp on identical pools. Teacher360 fit steps, full240 and answer-only240;10800/7200/4320 target presentations and collection/evaluation costs remain separate. An availability/selection failure is retained, not oracle-selected away. |
| 4 | End-to-end CPU fixture produces actual generated responses; Spark transfer remains unexecuted until its own run/report. | **Supported.** Actual6552-parameter neural teachers generate from all15 outcomes, and3088-parameter causal students fit actual bodies. Three campaigns retain840 updates/1872 generated responses including300 evaluation caps. Root independently replayed all non-timing runs. Local-only Spark-transfer source preparation uses dummy-file controls and stays explicitly unexecuted; no native distillation run is inferred. |
| 5 | Keep exact forward-KL/T-squared gradient lesson alongside sequence supervision, with their different objectives explicit. | **Supported.** Original Day26/01 and `distillation_lab.py` remain unchanged; Chapter15§15.5–15.7 and Chapter9§9.12–9.13 distinguish realized response NLL from distribution KL/T². The actual response lesson is additive, not a three-logit replacement. |

Seventeen focused checks and a fresh9-cell/6-figure notebook are retained;
artifact-map mismatch count in root review is zero. Four-item held-out slices
and teacher step/final contradictions are not broad natural-language reasoning
or deployment-speed evidence.

### DXI-16 — on-policy distillation and teacher context

Source [student_prefix_lab.py](../../src/dongxi_llms/student_prefix_lab.py),
[tests](../../tests/test_student_prefix_lab.py),
[report](2026-10-04-student-prefix-distillation.md),
[hardening](2026-10-04-student-prefix-hardening.md),
[independent hardening review](2026-10-04-student-prefix-hardening-root-review.json).

| # | Literal original criterion | Evidence and disposition |
|---|---|---|
| 1 | Teacher detached and token vocabulary aligned; evaluation excludes privileged teacher context. | **Supported.** `aligned_kl` line91 enforces the exact eight-ID mapping; `score_states` line269 detaches teacher targets. Hints enter only teacher target queries, never `question`/student input. `evaluate` line325 records zero teacher queries and no privileged context. Tests lines51–93 and152 cover actual gradient boundaries and hint-free evaluation. |
| 2 | Retain all generation/skipped-prompt costs; contrast state distribution change with loss-direction change. | **Supported.** Factorial source/direction/context are independent: three seeds×eight arms/720 updates. Journals retain10656 responses and all generation costs; ordinary-error group skips retain attempts, while actual main-run skips are0. Fixed offline targets use25200 states with3840 wrong-prefix uses; student-prefix targets use23331/8212. `states_from_pool` line197 enforces homogeneous cohort and all four attempts/source; conditional KL does not differentiate occupancy/sampling. |
| 3 | Exact full distributions conserve mass; bucketed top-k retains tail mass and labels lost token-level detail/approximation. | **Supported.** `validate_probabilities` line67, `kl_probabilities` line76 and `topk_tail` line99 verify full mass, detached teacher and exact tail decomposition. The same top-two/total-tail vectors have bucket KL0 but full forward0.211166908232/reverse0.339322921201; logit-gradient difference is explicitly checked. Fullk restores detail; hardening rejects zero-support tail-detail inf−inf while full KL separately represents genuine support infinities without floors. |
| 4 | No external teacher API or large-model production is implicit in this CPU extension; do not relabel it a full SDPO reproduction. | **Supported.** The finite teacher's0.8/0.2÷7 rule and engineered unhinted wrong-prefix failure are explicit, alongside actual2864-parameter student generation. All24 arms/poor held-out outcomes/caps remain. Report, notebook and Chapter15§15.9 deny a pretrained/self-teacher/API/full-SDPO or full occupancy-gradient result. |

The current27-test hardening receipt independently replays all24 arms/720
updates and10656 response records, excluding only measured seconds and their
duration-derived payload hashes. Fresh9-cell/7-figure evidence is retained.
Offline-forward-hint held-out5/24,6/24,7/24 versus student-prefix-forward-hint
1/24,0/24,3/24 is a preserved negative comparison, not universal on-policy
superiority. Mixed campaign/phase/update/arm/checkpoint and zero-support defects
were repaired with original source/results preserved and recipe unchanged.

## 3. Registered material, worked failure and plot pathways

All eight routes are registered as `extension`/`cpu-offline`, not future
empty placeholders or inferred learned mastery. The current manifest remains
15 chapters,28 days,76 notebooks, with exactly one route for each of these eight
IDs. Notebook source has adjacent predictions/reference answers and executable
mechanism/failure controls. Existing previews and retained fresh-run manifests
are evidence of their recorded revisions, not proof of current host rendering.

| Package | Registered day/chapter/notebook | Book/worked/lab integration and plot/failure |
|---|---|---|
|09|Day11/Ch8 — [04 teacher attempts](../../notebooks/day-11/04_teacher_attempts_and_matched_rejection_sft.ipynb)|[Ch8§8.11–8.14](../../book/chapters/08-instruction-data-as-an-interface.md#811-a-teacher-response-is-an-attempt-before-it-is-a-demonstration), [solutions11–16](../../book/solutions/08-instruction-data-as-an-interface.md#11-acceptance-is-not-ranking), [teacher lab](../../book/labs/08-instruction-data-as-an-interface.md#audit-a-teacher-before-teaching-from-its-answers);6 previews show pipeline, reasons/candidate matrix, matched units, global coverage, fit and all held-out failures. Ch15§15.4 bridges selection.|
|10|Day20/Ch12 — [04 learned critics](../../notebooks/day-20/04_learned_critics_and_frozen_text_rewards.ipynb)|[Ch12§12.10–12.14](../../book/chapters/12-language-generation-as-a-policy.md#1210-a-critic-predicts-the-return-before-an-action), [solutions13–17](../../book/solutions/12-language-generation-as-a-policy.md#13-lambda-endpoints-with-a-cap), [critic lab](../../book/labs/12-language-generation-as-a-policy.md#a-neural-critic-and-an-actual-text-reward);5 previews: actual gradient topology, EOS/cap GAE control, proxy/quality trajectories, every negative panel, confounded/balanced rewards.|
|11|Day18/Ch11 — [02 retention](../../notebooks/day-18/02_dpo_retention_and_preference_controls.ipynb)|[Ch11 controlled-retention discussion](../../book/chapters/11-direct-preference-optimization.md), [solutions11–13](../../book/solutions/11-direct-preference-optimization.md#11-explicit-auxiliary-gradients), [lab](../../book/labs/11-direct-preference-optimization.md);5 previews include absolute pair scores/margins, retention/generation, noise/length failures and unequal cost.|
|12|Day22/Ch13 — [03 objective weighting](../../notebooks/day-22/03_objective_weighting_and_filtering.ipynb)|[Ch13§13.5.1–13.5.2](../../book/chapters/13-group-relative-policy-optimization.md#1351-compare-objectives-before-comparing-training-runs), [solutions14–17](../../book/solutions/13-group-relative-policy-optimization.md#14-changing-scale-versus-changing-direction), [lab](../../book/labs/13-group-relative-policy-optimization.md);5 previews show exact coefficients, measured gradients/cosines, constructed clipping and rejected collection costs.|
|13|Day20/Ch12 — [03 probabilities/support](../../notebooks/day-20/03_behavior_probabilities_and_support.ipynb)|[Ch12 probability bridge](../../book/chapters/12-language-generation-as-a-policy.md), [Ch14§14.3](../../book/chapters/14-when-optimization-goes-wrong.md#143-entropy-collapse-certainty-can-mean-several-things), [worked11–13](../../book/solutions/14-when-optimization-goes-wrong.md#11-three-probability-distributions-and-a-support-boundary), [bridge lab](../../book/labs/14-when-optimization-goes-wrong.md);4 previews expose denominator/support/KL-gradient and stop-mask failures.|
|14|Day25/Ch14 — [02 ragged cache/recovery](../../notebooks/day-25/02_ragged_kv_cache_and_exact_recovery.ipynb)|[Ch14§14.7–14.8](../../book/chapters/14-when-optimization-goes-wrong.md#147-generation-can-dominate-the-budget), [worked16–18](../../book/solutions/14-when-optimization-goes-wrong.md#16-saved-work-and-elapsed-time-are-different-measurements), [lab](../../book/labs/14-when-optimization-goes-wrong.md);original5 previews cover real geometry, pad/positions, work/stops, snapshot boundary and exact/broken history. Current additional reader/RNG exercises remain separately scoped. Ch5 provides the architecture bridge.|
|15|Day26/Ch15 — [03 response distillation](../../notebooks/day-26/03_response_level_distillation.ipynb)|[Ch15§15.5–15.7](../../book/chapters/15-distill-evaluate-and-defend.md#155-hard-responses-and-soft-distributions), [worked15–18](../../book/solutions/15-distill-evaluate-and-defend.md#15-one-realized-response-is-not-a-distribution), [lab](../../book/labs/15-distill-evaluate-and-defend.md), [Ch9§9.12–9.13 bridge](../../book/chapters/09-supervised-fine-tuning.md#912-a-teacher-response-is-a-supervised-sequence);6 previews show supervision, real fits, held-out/step/final contradictions, selection failure and separate costs.|
|16|Day26/Ch15 — [05 student prefixes](../../notebooks/day-26/05_student_prefix_and_teacher_context.ipynb)|[Ch15§15.9](../../book/chapters/15-distill-evaluate-and-defend.md#159-what-histories-does-the-teacher-teach-on), [worked20–22](../../book/solutions/15-distill-evaluate-and-defend.md#20-separate-coverage-from-target-geometry), [lab](../../book/labs/15-distill-evaluate-and-defend.md);7 previews show private teacher/student boundary, changed states/fixed-state gradients, poor held-out behavior, exact/lost tail detail and every work unit.|

The manifest prerequisite paths for all eight notebooks exist. Pedagogical
prerequisites are not silently reinterpreted as requiring additional native
experiments. Registered chapter/day routing is preserved; the earlier corrected
Day25/Ch5 mismatch is historical rather than a current misplaced notebook.

## 4. Current small-source binding and verification limits

Read-only `sha256sum` of the eight mechanism modules and eight focused test
files matched their **latest relevant** recorded acceptances. The SHA-256 pairs
below identify the inspected source, not a new executed shared-suite revision.

| ID | Mechanism source SHA-256 | Focused test SHA-256 | Binding receipt |
|---|---|---|---|
|09|`d28fa327dbf6a50a7f50985217bd721c6c45012eaf44081aa163a1f0583086c2`|`5d969c4dea3c37050cdb46573539fe1610a7cff2498bf4595338cd89054b70d6`|teacher-data root acceptance|
|10|`fbd9c062bdc45008fae5bc968e5811f1d5ca6302054dddbb1a7520b7e930ccff`|`3f75b3e7a2b144ac9d1a3adac0d34a2533ccb688e1e3b36b2e9ee8f9c4266b47`|critic-policy acceptance|
|11|`37635d971493b7ecf4a8575474d3daf17009a929e26295a98b62d8e9e3beaa9b`|`4af53d79520b70c564c84a44449597884f5e4fec9a382d65b545b9ecf5332150`|DPO-retention verification|
|12|`9d599929cc17408f70d8f4cec088055adde873c43b355dfa9a877628afea8259`|`aca739330de23b3e1cf3c8ea78451e54c94ae17e3386770eb4b9e84add9f8f90`|GRPO-objective closing verification|
|13|`94b739b009d83c387735bf91d4abf4ba108d01df255520fff2943ae7e7ce4c32`|`b543f8a18635ed906fe0b5fb59911f4919a244bbb8d04c7d295c7e2df476cc6a`|sampling-support hardening|
|14|`564381546c250065aee5d5245bad0607493348d449552e68cb02a4917c0c6a04`|`ca9b361a4a305f7831462d4e46f72ff23e56cb29ef55d8485214f2e05307b403`|2026-10-05 tensor-byte follow-up, not old v2-source receipt|
|15|`0d41a4788aa64f622febd134da8e9607f4c9764bb85fff91abe54e88a13b4588`|`a35279aac69fbcff21ec6f9993914900787567755866b2fb088690a7b36c27da`|response-distillation verification|
|16|`022ec2957c1a42cb867a3f51e5f21190dd971a43b97ce52f147d83655653d828`|`6897fb06e13893973c42831f7ecda902a32fc90a1e575e36a374eb6e652da464`|student-prefix hardening root review|

Historical shared test counts such as218/366/443 apply only to their recorded
tree and time. This source mapping neither reruns them nor promotes their
hashes to every sibling source in the evolving tree. Similarly, preserved fresh
notebook reference copies/figures do not prove actual Mac execution or live
GitHub math rendering. The current all-source/full-notebook closure after the
native queue belongs to DXI-17, not an invented rerun inside this review.

## 5. Actionable omissions and remaining boundaries

**Blocking original DXI-09–16 omissions: none found.** Their original completion
scope is retained, including measured failure, explicit objective boundaries
and real end-to-end neural response/student references where required.

Two maintenance follow-ups remain useful but are not additional acceptance
gates:

1. The Chapter12 lab table lists PPO and learned critics but leaves the registered
   Day20/03 support instrument discoverable through Day20 README, Chapter12's
   prose link and Chapter14's bridge lab. Adding a direct table row would improve
   discoverability; the lesson/solution/failure/plot pathway already exists.
2. When recording final current-source closure, bind the then-current Day25
   ten-cell reader extension and all later chapter/prose edits to its own manifest.
   Do not cite the historical seven-cell receipt as fresh execution of those
   additions. The need for that integrated closure is already tracked outside
   these eight original bounded packages.

Retained edge-case repairs must stay visible: teacher journal execution/commit
and reference-consistency scope, sampling underflow/excluded arithmetic,
cache structural identity and BF16 byte extraction, and student-prefix mixed
cohorts/zero-support tail detail. No report was overwritten by this review.
Native response distillation, API/pretrained teachers, full SDPO, generalized
off-policy trajectory correction, production multi-epoch PPO, optimized serving
speedups and broad reasoning/retention superiority remain **unestablished** by
these CPU packages. The independently accepted native chosen/DPO recovery
reports answer their own later questions and do not retroactively turn the
original symbolic outcomes into model-scale quality evidence.

The learner remains **Day9 on Spark**. The coherent book remains15 chapters
over28 learning days. This review changes no learning progress or package status,
does not assert the queued100-update pilot has finished and does not complete
DXI-03, DXI-17, DXI-18 or the active improvement goal.
