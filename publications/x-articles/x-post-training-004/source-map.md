# Source map — X-POST-004

The public body has no links. Every factual claim maps to a source here.

| Article claim | Source |
|---|---|
| DeepSeek-R1-Zero: RL directly on the base model without SFT; rule-based accuracy and format rewards (answer check; reasoning between think tags); AIME 2024 pass@1 from 15.6% to 71.0%, 86.7% with majority voting; average response length grows during RL; GRPO | DeepSeek-AI (2025), "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", arXiv:2501.12948v1: abstract, §2.2.2 (reward modeling), §2.2.4 and Figure 3. Text checked 2026-10-28. |
| Problems with process reward models: hard to define a fine-grained step; hard to judge intermediate steps (automated annotation unsatisfactory, manual annotation does not scale); a model-based PRM leads to reward hacking | Same paper, §4.2 "Unsuccessful Attempts: Process Reward Model (PRM)" (checked 2026-10-28). The paper also says no neural reward model was used for R1-Zero because of reward hacking (§2.2.2). |
| R1-Zero's poor readability and language mixing; DeepSeek-R1 adds cold-start data and a language-consistency reward (proportion of target-language words in the CoT) | Same paper, abstract and §2.3.2 (checked 2026-10-28) |
| gpt-oss-120b and gpt-oss-20b are open-weight models under Apache 2.0; flagship models through API and apps | OpenAI gpt-oss model pages on Hugging Face (checked 2026-10-27). Stated as of the time of writing. |
| Agent, environment, policy; the judge as part of the environment; reward model frozen during RL | Learning note D39 glossary and D20–D23 (`learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md`); book Chapter 12 §12.1 |
| State = prompt + tokens so far, action = next token, trajectory/rollout/episode | Learning note D39; book §12.1 |
| Sparse reward (0 per token, score at the end); return; γ = 1 for LLM RL and why | Learning note D39; book Chapter 12; review discussion recorded in `plan.md` |
| Outcome-only rewards can pass right answers with wrong reasoning; statistical tendency toward sound reasoning | Review discussion recorded in `plan.md` (stated qualitatively, no external citation) |
| V, Q, A = Q − V; probability-weighted A averages to 0 | Learning note D24–D25, D39; arithmetic on the running example (`plan.md`) |
| PPO learns V with a critic; GRPO uses the group average | Book Chapter 12 §12.10–12.11; learning note D21 and the method table. DeepSeek-R1 paper §2.2.1 for GRPO in R1-Zero. |
| Baseline: unchanged expected update, lower variance; R − V averages to Q − V | Book §12.4; arithmetic and simulation of the running example (`plan.md`: 200,000 samples, seed 0; variance 20.8 with b = 0, 3.7 with b = V, reported as about five times smaller) |
| TD error δ = r + V′ − V; GAE λ; λ = 0 vs 1 on the example; λ near 0.95 in PPO practice | Book §12.11 (λ = 0 is one-step TD, λ = 1 leaves G_t; worked example with λ = 0.95); learning note glossary row 15 (`lam=0.95`); λ = 0.95 is also the PPO paper's default (Schulman et al., 2017; not re-checked for this article); arithmetic (`plan.md`) |
| GRPO gives every token of an answer the same advantage | Learning note glossary, Credit assignment row (uniform per answer for REINFORCE/GRPO; D37–D38) |
| Policy gradient: Δ logit ∝ π·A; two exact steps .50/.30/.20 → .15/.36/.49 → .04/.11/.86, V 4.3 → 6.5 → 8.3 | Arithmetic (`visuals/animations/projects/post-training-rl-terms/scenes.py`, `pg_steps`); validated at render time |
| Exploration: greedy decoding sees only open; sampling at temperature 1 | Learning note D39; book Chapter 12 |
| On-policy data, π_old, several update steps per batch, importance ratio, clipping window [1 − ε, 1 + ε]; π(partly) 0.2 → 0.24 at ε = 0.2 | Book §12.6–12.8; arithmetic |
| KL penalty β·KL(π‖π_ref); flattery example: 9 vs 9.6, π_ref 5%, drift to 90%, judge 9.03 → 9.54, KL 2.38, objective 9.03 → 7.16 at β = 1; real systems use per-token KL and smaller β | Book §12.8; learning note D35 and the KL row (per-position KL summed over the answer; `BETA = 0.05` in the notebook); arithmetic (`scenes.py`, `kl2`); stated as illustrative |
| Reward hacking | DeepSeek-R1 paper (neural reward models); book Chapter 12; learning note D39 |
| Next article: reward model from preference pairs; PPO, GRPO, DPO | Series plan (`../README.md`); SLP3 Chapter 8 |

## Evidence boundary

Only the DeepSeek-R1-Zero numbers are published results. Every other number comes from the
illustrative running example, with exact arithmetic that the render script validates. The baseline
variance figure is from a small simulation of that example and is stated approximately. No results
from our own model runs appear in the article.
