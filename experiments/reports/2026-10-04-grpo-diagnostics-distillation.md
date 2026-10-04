# Chapters 13–15 CPU mechanism results

Measured report; source hashes and every row are in the [JSON](2026-10-04-grpo-diagnostics-distillation.json).

Interpreter: `/home/dongxi/dgx-spark-dongxi/.venv/bin/python`; Python 3.12.14; Torch 2.13.0+cu130; aarch64; one CPU thread.
Command: `/home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/run_grpo_capstone_experiments.py`. Elapsed: 2.944 seconds.

## Autoregressive decoder GRPO

One 16-wide layer, 12-token vocabulary; 60 SFT warm-start updates; 12 fresh RL updates per variant. Original 12 training/four held-out arithmetic prompts; frozen split, not natural-language reasoning.

|Group|Valid response tokens|Initial train greedy|Final train greedy|Initial held-out greedy|Final held-out greedy|
|---:|---:|---:|---:|---:|---:|
|4|192|1.000|1.000|0.000|0.000|
|8|385|1.000|1.000|0.000|0.000|

The comparison holds update count fixed, so G8 consumes more samples. It is not an equal-token algorithm ranking. Every held-out response is retained. Zero-variance groups and negative results are evidence, not discarded rows.

## Reward misspecification

- Broken proxy: expected proxy 1.000000 → 1.966561; strict accuracy 0.333333 → 0.016816.
- Strict-reward restart: expected proxy 0.333333 → 0.964428; strict accuracy 0.333333 → 0.964428.

Both policies start from zero logits. This is actual exact-expectation SGD over three actions; repairing a previously hacked language model is not measured.

## Distribution distillation

80 SGD updates at T=2: scaled forward KL 0.657134229 → 0.000000098. Teacher/student probabilities are retained. This establishes three-logit distribution fitting, not transferred reasoning.

## Selection simulation

Original categorical Monte Carlo, 1200 trials, seed 2628. Candidate cost is an illustrative 8 tokens each. Oracle availability assumes independent correctness; the imperfect ranker favors a wrong class.

|Candidates|Majority accuracy|Proxy best-of-N accuracy|Oracle availability|Illustrative tokens|
|---:|---:|---:|---:|---:|
|1|0.4458|0.4458|0.4458|8|
|2|0.4458|0.3192|0.6892|16|
|4|0.5083|0.1217|0.8992|32|
|8|0.5608|0.0208|0.9917|64|
|16|0.5683|0.0000|1.0000|128|

## Boundary

No model downloads, Qwen execution, CUDA measurements, long training run, inference service, animation render or external publishing. Invariant/gradient tests and notebook execution have separate verification logs. Learner mastery is not inferred from this generated material.
