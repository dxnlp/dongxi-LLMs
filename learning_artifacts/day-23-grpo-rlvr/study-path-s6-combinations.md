# S6 study notes: Combinations: feedback sources × optimizers

- Notebook: [`notebooks/study-path/s6_combinations.ipynb`](../../notebooks/study-path/s6_combinations.ipynb) (learner copy in
  `learner_work/notebooks/study-path/`)
- Study path stage 6 of 7 · Chapter 13 · Day 23
- Status: not started
- Main preference note (Key concepts D1–D55, RL glossary, S1 log, plan table P1–P12):
  [learning-from-preferences.md](../day-15-preference-data-and-bradley-terry/learning-from-preferences.md); RL terms:
  [glossary](../day-15-preference-data-and-bradley-terry/learning-from-preferences.md#-rl-terms-in-the-llm-setting-glossary-d39)

One learning note per study-path notebook from S2 on (learner, 2026-11-03). Questions raised while
working through **S6** are recorded here as they happen, tagged `[S6 · Part X]`. **Discussion numbers
continue one sequence across all notes** (D1–D55 in the main note, D56 onward in the stage notes): the
next number is one more than the highest in any of them, so a D-number points to exactly one place.
Animation and article requests also go into the main note's plan table and
`visuals/animations/PROPOSALS.md`. ★ = marked important by the learner.

## Key concepts

| Concept | Description | Status |
|---|---|---|

## Knowledge gaps

- none recorded yet

## S6 notebook study log

| Part | Question | Status | Exercises / predictions | Discussions and requests |
|---|---|---|---|---|
| A · Best-of-n and rejection sampling | What does selecting with a reward model do to the answer distribution, and what does SFT on the selections converge to? | not started | — | — |
| B · Offline vs online DPO | DPO has no reward model inside. What changes when a reward model labels pairs that the current policy samples? | not started | — | — |
| C · Reward model + verifier | How do you mix a learned score with a 0/1 verifier so that a wrong answer can't win? | not started | — | — |
| D · SFT → DPO → RLVR | What does each stage of a Tülu 3-style pipeline change, and what keeps a later stage from undoing an earlier one? | not started | — | — |
| E · Qwen3: reward-model-labelled online DPO | After one round, does the labelling reward model score fresh answers higher, and does an independently trained reward model agree? | not started | — | — |

## Discussions
