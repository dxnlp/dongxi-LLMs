# Study path S1–S7: preference learning and RL, hands on

Seven optional Spark GPU workbooks, one per stage of the study path approved on 2026-10-29
(learning note D40, "★ Study path"). They go along two axes: the **feedback source** (human pairs,
a reward model, a verifier) and the **optimizer** (PPO, GRPO, and DPO as the RL-free loss).

Each notebook runs a mechanism first on a toy (tiny tensors or a tabular policy: exact, seconds)
and then once on the real Qwen3-0.6B. In every exercise you write 1–4 key lines where you see `...`,
run the check (✅/❌ with a hint, never a crash), then compare with the reference. Later cells use
your version if its check passed, otherwise the reference, so one stuck exercise never blocks the rest.

| Stage | Notebook | Course day | You write | Real-model part (reference run, one seed) |
|---|---|---|---|---|
| S1 | [Reward models](s1_reward_models.ipynb) | 16 · Ch 10 | BT loss and its gradient, a linear RM fit, label noise, ECE, last-token pooling, pair accuracy ± SE, length audit, outcome vs process labels | RM on 1,200 HH pairs: held-out 0.52 → 0.65; the gain is on pairs where the chosen answer is also longer |
| S2 | [KL optimum → DPO](s2_kl_optimum_and_dpo.ipynb) | 18 · Ch 11 | KL, π* ∝ π_ref·exp(R/β), the objective, implicit rewards, DPO loss and gradient, masked sequence log-probs, a DPO step | DPO on 512 pairs: implicit-reward accuracy 50% → 67% ± 3%; larger β moves the policy less |
| S3 | [REINFORCE → baselines → PPO](s3_policy_gradient_to_ppo.ipynb) | 21 · Ch 12 | exact gradient, REINFORCE, RLOO, V/Q/A on a toy, GAE, KL-shaped rewards, clipped loss, value loss | PPO with a rule reward (finish within 48 tokens): 0.12 → 0.50; KL to ref 6.4 nats |
| S4 | [GRPO](s4_grpo.ipynb) | 22 · Ch 13 | group z-scores, zero-signal probability, three loss normalisations, k1/k3 KL, the GRPO loss | GRPO with a length rule: 0.27 → 0.89; peak 17 GiB with no value model |
| S5 | [RLVR](s5_rlvr.ipynb) | 23 · Ch 13 | answer extraction, normalisation, verifier reward, pass@k, pool filter, format/arithmetic split | GRPO + verifier on synthetic arithmetic: 0.30 → 0.80, about ¾ of it from writing the answer line |
| S6 | [Combinations](s6_combinations.ipynb) | 23 · Ch 13 | best-of-n, rejection-sampling SFT, RM-labelled pairs, RM + verifier mixing, SFT → DPO → RLVR on a toy | one round of RM-labelled online DPO: both RMs' scores rise, answers get ~50% longer |
| S7 | [Failure modes](s7_failure_modes.ipynb) | 24 · Ch 14 | overoptimization curve, best-of-n KL, length-controlled score, entropy, conservative RM ensemble, a training-log diagnostic | GRPO against a learned RM with β = 0: every answer grows to the 128-token cap |

**Environment.** The `Python (DGX Spark Native)` kernel with CUDA, cached Qwen3-0.6B, and the cached
HH-RLHF helpful-base files (S1, S2, S6, S7). Nothing is downloaded. Each notebook takes about 2–12
minutes and peaks at 17–23 GiB of GPU memory. Run one notebook's kernel at a time.

**Precision.** Every trained model keeps fp32 master weights and runs forward passes under bf16
autocast. In bf16, AdamW steps of 1e-6 to 1e-5 round back to the same weight for most parameters
(measured: 2–10% of weights change instead of nearly all). Frozen models stay bf16.

**Evidence.** The toy results are exact and asserted. The Qwen3 runs use one seed and small budgets.
They verify the mechanics and measure effects on the stated rewards, with standard errors. None of
them is a claim about answer quality.

Shared code: `src/dongxi_llms/study_path/` (`common.py` checks, `qwen.py` data and sampling,
`sN_*.py` references and checks; tests in `tests/test_study_path_*.py`).
