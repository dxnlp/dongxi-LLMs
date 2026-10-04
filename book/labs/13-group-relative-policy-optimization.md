# Chapter 13 lab route — grouped rewards to a decoder update

Prerequisites: Chapter 12 policy gradients and Chapter 5 decoder tensor shapes.
Machine: CPU on Mac or Spark; no server/model download required. Use the course
Torch/Matplotlib environment. Each notebook includes prediction prompts,
adjacent executable answers, measured plots and a controlled modification.

| Day | Session | Main question | Intervention |
|---|---|---|---|
|22|[Group advantages](../../notebooks/day-22/01_group_advantages.ipynb)|What signal survives reward normalization?|Population versus sample std; constant group|
|22|[Ratios and KL](../../notebooks/day-22/02_token_ratios_kl.ipynb)|Which probability movements receive pressure?|Advantage sign; clipping; exact KL derivative|
|23|[Autoregressive RLVR](../../notebooks/day-23/01_decoder_rlvr.ipynb)|How do sampled answers reach decoder parameters?|Group 4 versus8 with cost recorded|
|23|[Verifier contract](../../notebooks/day-23/02_verifier_contract.ipynb)|What exactly is being rewarded?|Substring exploit and strict parsing|

Start with a prediction about all-failure groups. Inspect reward/advantage arrays,
then ratios and gradients. Run the real tiny decoder update only after explaining
the detach boundary. Read every held-out row, including failures. The
[bounded report](../../experiments/reports/2026-10-04-grpo-diagnostics-distillation.md)
records actual CPU execution; its result is not a Qwen score.

The [Spark model-scale contract](../../experiments/specs/day-23-qwen-rlvr.md)
specifies the optional locally loaded Qwen pathway, smoke budget and acceptance
gates. Model files and platform resources are required for that separate run.

Completion evidence: explain the objective's reduction/std conventions; prove
old/reference parameters receive no gradient; reproduce the aligned first-token
log probability; identify at least one verifier false positive; interpret
zero-variance groups and one held-out failure. Material execution alone does not
establish that the learner has demonstrated those explanations.
