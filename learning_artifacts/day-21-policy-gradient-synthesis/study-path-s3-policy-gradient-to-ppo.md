# S3 study notes: From REINFORCE to PPO: policy gradients, baselines, advantages and clipping

- Notebook: [`notebooks/study-path/s3_policy_gradient_to_ppo.ipynb`](../../notebooks/study-path/s3_policy_gradient_to_ppo.ipynb) (learner copy in
  `learner_work/notebooks/study-path/`)
- Study path stage 3 of 7 · Chapter 12 · Day 21
- Status: not started
- Main preference note (Key concepts D1–D55, RL glossary, S1 log, plan table P1–P12):
  [learning-from-preferences.md](../day-15-preference-data-and-bradley-terry/learning-from-preferences.md); RL terms:
  [glossary](../day-15-preference-data-and-bradley-terry/learning-from-preferences.md#-rl-terms-in-the-llm-setting-glossary-d39)

One learning note per study-path notebook from S2 on (learner, 2026-11-03). Questions raised while
working through **S3** are recorded here as they happen, tagged `[S3 · Part X]`. **Discussion numbers
continue one sequence across all notes** (D1–D55 in the main note, D56 onward in the stage notes): the
next number is one more than the highest in any of them, so a D-number points to exactly one place.
Animation and article requests also go into the main note's plan table and
`visuals/animations/PROPOSALS.md`. ★ = marked important by the learner.

## Key concepts

| Concept | Description | Status |
|---|---|---|

## Knowledge gaps

- none recorded yet

## S3 notebook study log

| Part | Question | Status | Exercises / predictions | Discussions and requests |
|---|---|---|---|---|
| A · One decision | What is the exact gradient of expected reward, and how does REINFORCE estimate it from samples? | not started | — | — |
| B · Baselines and RLOO | Why may we subtract a baseline, and which baselines lower the variance? | not started | — | — |
| C · Credit over tokens | How do return, value, Q and advantage split one terminal reward across tokens? What does GAE's λ trade? | not started | — | — |
| D · PPO's pieces | KL-shaped rewards, the clipped ratio loss, the value loss; REINFORCE vs RLOO vs PPO on a toy | not started | — | — |
| E · PPO on Qwen3 | Policy, reference, reward and value in one real PPO loop: what moves, and by how much? | not started | — | — |

## Discussions
