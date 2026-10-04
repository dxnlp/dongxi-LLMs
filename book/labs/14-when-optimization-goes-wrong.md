# Chapter 14 lab route — failures with discriminating evidence

Prerequisites: Chapter 13 objective and verifier. CPU on Mac or Spark; no rollout
server or CUDA process is needed. Simulated system budgets are labeled projected.

| Day | Session | Main question | Intervention |
|---|---|---|---|
|24|[Reward hacking](../../notebooks/day-24/01_reward_hacking.ipynb)|Can reward rise while accuracy falls?|Broken proxy versus strict reward from equal initialization|
|24|[Length and entropy](../../notebooks/day-24/02_length_entropy_diagnostics.ipynb)|What does the objective weight?|Response means versus token means; entropy concentration|
|25|[Versions and budget](../../notebooks/day-25/01_rollout_versions_and_budget.ipynb)|When is an old response safe to reuse?|Strict freshness; changed identities; generation bottleneck|

The reward experiment performs real SGD over a finite action policy. It is
independent of Monte Carlo/GRPO estimator noise. The version exercise is a
metadata contract; it does not certify a production engine. The budget exercise
is an analytical scenario, not a measured Spark throughput claim.

Write an incident record: observation, first failed invariant, two possible
causes, intervention, outcome and remaining uncertainty. Keep the losing
outputs. Read [worked explanations](../solutions/14-when-optimization-goes-wrong.md)
before attempting a larger repair. Completion requires defending why the chosen
intervention distinguishes causes rather than merely changing several settings.
