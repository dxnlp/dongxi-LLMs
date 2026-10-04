# Chapter 12 — Policy Gradient Laboratories

The CPU route enumerates small distributions before sampling a procedural task.
It needs PyTorch, Matplotlib, and an existing notebook environment. Mac Studio
is the ordinary learning lane; Spark CPU can execute the same code. No downloaded
model, GPU, or verifier service is required.

| Day | Session | Question |
|---|---|---|
| 19 | [Exact REINFORCE](../../notebooks/day-19/01_exact_reinforce_gradient.ipynb) | Can one sample point the wrong way while its expectation is correct? |
| 20 | [Baselines and RLOO](../../notebooks/day-20/01_baselines_and_rloo.ipynb) | Which centering operations preserve the expectation? |
| 20 | [PPO and KL](../../notebooks/day-20/02_ppo_clipping_and_kl.ipynb) | What does clipping flatten, and when is a KL estimate unbiased? |
| 21 | [Algorithm comparison](../../notebooks/day-21/01_policy_algorithm_comparison.ipynb) | How do rollout budget and gradient-work budget change a comparison? |

The source module `policy_gradient_lab.py` computes exact categorical gradients,
sample-estimator covariance traces, all iid groups up to a bounded size, the
PPO surrogate, finite-support KL estimators, and actual sampled policy updates.
Plots expose signed gradients and policy trajectories. References follow every
prediction; each notebook ends with an explicit limit on its evidence.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/run_preference_policy_cpu.py
python scripts/verify_course_notebooks.py --days 19 20 21 --kernel dgx-spark-native --export-figures
PYTHONPATH=src python -m unittest discover -s tests -p test_preference_policy_labs.py -v
```

The [specification](../../experiments/specs/2026-10-04-preference-policy-cpu.md)
declares sampling seeds and budgets before the recorded run. The
[report](../../experiments/reports/2026-10-04-preference-policy-cpu.md) keeps
negative outcomes and both work clocks. PPO uses an exact detached value in this
finite control; it has no learned critic, so the experiment cannot establish
critic quality or neural PPO performance. Reward/KL gradients and KL-monitoring
estimators have separate contracts. Chapter 13 supplies the model-scale grouped
rollout path; Chapter 14 supplies failure diagnosis and system accounting.
