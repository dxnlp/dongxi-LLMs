# S4 study notes: GRPO: the group replaces the value model

- Notebook: [`notebooks/study-path/s4_grpo.ipynb`](../../notebooks/study-path/s4_grpo.ipynb) (learner copy in
  `learner_work/notebooks/study-path/`)
- Study path stage 4 of 7 · Chapter 13 · Day 22
- Status: not started
- Main preference note (Key concepts D1–D55, RL glossary, S1 log, plan table P1–P12):
  [learning-from-preferences.md](../day-15-preference-data-and-bradley-terry/learning-from-preferences.md); RL terms:
  [glossary](../day-15-preference-data-and-bradley-terry/learning-from-preferences.md#-rl-terms-in-the-llm-setting-glossary-d39)

One learning note per study-path notebook from S2 on (learner, 2026-11-03). Questions raised while
working through **S4** are recorded here as they happen, tagged `[S4 · Part X]`. **Discussion numbers
continue one sequence across all notes** (D1–D55 in the main note, D56 onward in the stage notes): the
next number is one more than the highest in any of them, so a D-number points to exactly one place.
Animation and article requests also go into the main note's plan table and
`visuals/animations/PROPOSALS.md`. ★ = marked important by the learner.

## Key concepts

| Concept | Description | Status |
|---|---|---|

## Knowledge gaps

- none recorded yet

## S4 notebook study log

| Part | Question | Status | Exercises / predictions | Discussions and requests |
|---|---|---|---|---|
| A · Group advantages | What does the per-group z-score do, and how often does a group give no signal at all? | not started | — | — |
| B · Loss normalisation | Per-answer mean, token mean or a constant: how much is each token pushed, and why does length matter? Why divide by the std? | not started | — | — |
| C · KL estimators | k1 and k3: what does each estimate, with what variance, and what gradient does k3 give in the loss? | not started | — | — |
| D · The GRPO loss | Clipped surrogate + β·k3, written once; what happens with one update vs several on one batch? | not started | — | — |
| E · Toy GRPO vs REINFORCE | Does the group baseline beat no baseline on a 0/1 reward, and when does a baseline matter? | not started | — | — |
| F · GRPO on Qwen3-0.6B | Reward, KL, zero-variance groups and length over 40 real updates; memory without a value model | not started | — | — |

## Discussions
