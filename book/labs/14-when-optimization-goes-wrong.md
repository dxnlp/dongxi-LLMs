# Chapter 14 lab route — failures with discriminating evidence

Prerequisites: Chapter 13 objective and verifier. CPU on Mac or Spark; no rollout
server or CUDA process is needed. Simulated system budgets are labeled projected.

| Day | Session | Main question | Intervention |
|---|---|---|---|
|16 bridge|[Text reward and vocabulary](../../notebooks/day-16/03_text_reward_and_process_labels.ipynb)|Can the reward distinguish inputs, and does it generalize afterward?|Preserved word collisions versus a separately specified character representation|
|18 bridge|[DPO retention](../../notebooks/day-18/02_dpo_retention_and_preference_controls.ipynb)|Did a lower preference loss preserve an unrelated task?|Chosen-response likelihood and rehearsal; clean/noisy/length controls|
|20 bridge|[Behavior probabilities and support](../../notebooks/day-20/03_behavior_probabilities_and_support.ipynb)|Which distribution generated this action, and which objective are we updating?|Wrong denominator, absent support, KL value/gradient mismatch and EOS/padding boundaries|
|20 bridge|[Learned critics and reward](../../notebooks/day-20/04_learned_critics_and_frozen_text_rewards.ipynb)|Can a bootstrap amplify a mistaken future-value prediction?|Oracle/learned/noisy values; learned frozen rewards; terminal/cap separation|
|22 bridge|[Matched objective controls](../../notebooks/day-22/03_objective_weighting_and_filtering.ipynb)|Did the loss or the selected data change the gradient?|Fixed rollout pool, analytical gradients, clipping controls and all retry costs|
|24|[Reward hacking](../../notebooks/day-24/01_reward_hacking.ipynb)|Can reward rise while accuracy falls?|Broken proxy versus strict reward from equal initialization|
|24|[Length and entropy](../../notebooks/day-24/02_length_entropy_diagnostics.ipynb)|What does the objective weight?|Response means versus token means; entropy concentration|
|25|[Versions and budget](../../notebooks/day-25/01_rollout_versions_and_budget.ipynb)|When is an old response safe to reuse?|Strict freshness; changed identities; generation bottleneck|
|25|[Ragged cache and exact recovery](../../notebooks/day-25/02_ragged_kv_cache_and_exact_recovery.ipynb)|Which computations and states can really be reused?|Actual compact KV, row positions/EOS/caps, completed and pending updates, identity migration|

The reward experiment performs real SGD over a finite action policy. It is
independent of Monte Carlo/GRPO estimator noise. The version exercise is a
metadata contract; it does not certify a production engine. The budget exercise
is an analytical scenario, not a measured Spark throughput claim.

Write an incident record: observation, first failed invariant, two possible
causes, intervention, outcome and remaining uncertainty. Keep the losing
outputs. Read [worked explanations](../solutions/14-when-optimization-goes-wrong.md)
before attempting a larger repair. Completion requires defending why the chosen
intervention distinguishes causes rather than merely changing several settings.

For the DXI-13 bridge, predict whether identical raw logits give ratio1 after
temperature and filtering, then test the exact finite identity. A legitimate
repair must name its objective: full-support collection preserves the raw target;
conditioning on a fixed retained support changes it. The four plots show actual
finite probabilities/gradients and a constructed token mask, not LLM benchmarks.
Read [worked explanations11–13](../solutions/14-when-optimization-goes-wrong.md#11-three-probability-distributions-and-a-support-boundary)
and [the specified CPU report](../../experiments/reports/2026-10-04-sampling-support.md).

The additional bridges do not replace the failure notebooks or advance learner
mastery automatically. Use one observed negative result to write a discriminating
incident record. A character vocabulary can remove indistinguishable inputs
without improving ranking; a critic can bootstrap at a cap without receiving
terminal reward; rehearsal can continue an imperfectly learned task. Explain
which observation each control supports and what remains unidentified. Compare
objective gradients only on the same pool; count discarded attempts when
discussing filtering. These are measured tiny CPU mechanisms, not confirmation
of pretrained capability or Spark-scale stability.

The cache session supplies the missing systems bridge: real tiny decoder
actions/likelihoods agree across four paths, while work and elapsed time differ.
Its recovery evidence tests these isolated CPU interfaces. Read the preserved
[digest-hardening/recovery report](../../experiments/reports/2026-10-04-batched-cache-recovery.md)
before describing old state files as current resume artifacts. The new loader
rejects obsolete identity contracts rather than silently relabeling them.

For a source-readiness extension, read the
[owned-worker/disk report](../../experiments/reports/2026-10-05-owned-workers-and-disk-guards.md)
and predict which fields differ when the leader exits0 but its worker continues.
Inspect actual leader exit, stop reason, retained worker identities and final
live/unknown states together. Compare real per-file SIGXFSZ with the sampled
aggregate stop; keep injected shortages and mocked ownership faults labelled.
The fixed standard-library collector is reproducible in a new evidence directory
on Linux, but supplies no arbitrary/model launcher. This optional audit does not
replace the notebook route or authorize a production stage.

The next systems reading is section14.8's whole-job boundary. Sketch the four
DPO score branches before comparing update geometry with total logical work.
Then consider a crash after a durable snapshot but after another work reservation:
which numerical state restores, and which spending must remain? Read worked
answers21–24 and the [nonexecuting preflight spec](../../experiments/specs/2026-10-05-production-preflight.md).
Predict refusals for stale membership, absent quota and a clean leader exit with
a remaining worker. These are CPU source-contract exercises, not an invitation
to create cgroups, quotas, services or model runs.

Continue with worked answers25–28. Trace a pending-pool likelihood check and its
temporary sampling replay: what executes even though the optimizer cursor does
not move? Compare cached SFT observations with uncached DPO/RLVR, retaining their
original generation paths. In the [actual snapshot report](../../experiments/reports/2026-10-05-dpo-snapshot-artifacts.md),
explain why six completed updates, ten reserved update allowances and a retained
partial file can all be correct. Then list outputs that are still outside the
snapshot ledger. This is a source-reading/CPU control extension, not a new claim
of learner mastery or approval to run a model-scale campaign.

For worked answers29–30, trace the live training sampler beside the temporary
validation generator. Predict which one advances in a history check, then locate
the actual load-callback and restore calls in the DPO runner. Repeated validation
is repeated work even when it produces the same numerical answer. Trace the
shared reader next: identify which byte/tensor checks happen before the callback.
Read the [inspection-budget plan](../../docs/SNAPSHOT_INSPECTION_BUDGET_PLAN.md)
and explain why a trusted pre-read receipt is necessary. Exercise7 in the same
notebook now makes actual shared save/inspect/load calls with a separate I/O
journal. Predict the fourth-load refusal before viewing the operation-count
plot. All repeated visits stay charged; identical weights do not erase them.

For worked answers31–32, find where identity collection first reads a resume
payload, then where the independently retained receipt binds its physical
journals. Explain why save completion belongs in later journal history, not
inside its own payload. Change the declared load allowance only as a new
microscope configuration, not a refill of the old journal. Read the
[source verification boundary](../../experiments/reports/2026-10-05-snapshot-io-readiness.md)
before describing it as production readiness. No pretrained, physical quota,
service or cross-machine experiment is authorized by this lesson.

For worked answer33, predict Exercise8's source cursor and first resumed action
before running the native random-Qwen CPU microscope. It compares two applied
updates with/without a retained next pool, keeps a rejected read charged and
finishes the original four-update trajectory. Explain why both saved prefixes
can precede later journal charges without refunding them. Then derive the full
native lifecycle's pending/completed publication counts separately from the
microscope's measured one-save/two-load trace. Do not use completed-only schedule
geometry or identical weights as permission to recollect a pending group.

For worked answers 34–35, trace three lanes: the model operation, external clock and
resource/logging workers. Identify which lane can hang without preventing the
clock from requesting owned shutdown. Then inspect the fixed-profile
[source contract](../../experiments/specs/2026-10-05-native-profile-watchdog.md)
and its [inert controls](../../tests/test_native_profile_supervisor.py).
The focused CPU route is:

~~~bash
PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest -v test_native_profile_supervisor
~~~

This tests the controller, not a model or the full course. Keep successful exits,
nonzero exits, refused preflights, ignored TERM, stalled/failed observations and
failed logs in the same record. Explain why an actual leader exit, worker
cleanup and retained final receipt can disagree. Do not invoke the native
adapter's execution option as part of this reading exercise.

Now read the retained [actual DPO incident and replay record](../../experiments/reports/2026-10-05-native-dpo-recovery-deadline.md)
and its run03 [acceptance](../../experiments/reports/native-dpo-replay-20261005-run-03/acceptance.json)
and [comparison](../../experiments/reports/native-dpo-replay-20261005-run-03/comparison.json).
This is a metadata-reading exercise, not permission to reload payloads or launch
a model. First predict which evidence would be sufficient for each claim:
durable update2, whole resumed-child success, same numerical trajectory, and
independent preference quality. Then explain why run02 can have the first
without the second, and why run03's ten matching component identities do not
establish the fourth.

Locate actual CPU8 and hash-worker-4 stdout witnesses. Check that four workers
change independent-file scheduling rather than omitting body hashes, that the
default remains serial, and that all 600-second invocation guards and scientific
settings remain unchanged. Keep child, supervisor and owner-transcribed outer
timings separate. A shorter complete run after two capped failures is not by
itself a causal speedup measurement.

Finally trace the shared work journal: source2 plus fresh completed1→2 spends
three actual optimizer updates but ends at numerical cursor2. Identify the
additional admitted CPU comparison inspect/load charges and the separately
named inventory/component-hash exclusions. Do not refund later work, erase
the 01/02 failure/null-exit records or turn this two-update replay into a 100-update
pilot or capability result. Use the same incident-record structure above;
source readiness, actual recovery and quality require different evidence.
