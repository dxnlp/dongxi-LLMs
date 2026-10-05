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
| 20 | [Behavior probabilities and sampling support](../../notebooks/day-20/03_behavior_probabilities_and_support.ipynb) | Which distribution produced this response, and what fails when its probability or support is misreported? |
| 21 | [Algorithm comparison](../../notebooks/day-21/01_policy_algorithm_comparison.ipynb) | How do rollout budget and gradient-work budget change a comparison? |
| 20 | [Learned critics and frozen text rewards](../../notebooks/day-20/04_learned_critics_and_frozen_text_rewards.ipynb) | Can fitted rewards and bootstraps hide delivered-quality failures? |

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

## A neural critic and an actual text reward

The exact-value control above stays unchanged. `critic_policy_lab.py` trains a
causal token actor and separate causal value head. Trace reward/TD/GAE tensors,
break the cap mask, then run the adjacent small actor/critic reference with
verified saved rewards. Inspect every fixed balanced/confounded seed. The
[specification](../../experiments/specs/2026-10-04-critic-policy.md) precedes
fitting; the [report](../../experiments/reports/2026-10-04-critic-policy.md)
retains all sampled paths, value errors and quality failures.

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m unittest discover -s tests -p test_critic_policy_lab.py -v
CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m dongxi_llms.critic_policy_lab --report /tmp/NEW-critic-policy.json --exports /tmp/NEW-frozen-rewards
```

Use unused output paths. Conditional syntax/cap-four continuation are explicit,
not pretrained critic evidence, production PPO, human feedback or demonstrated
color-rule transfer. Saved plots are measured previews; rerun for interventions.
