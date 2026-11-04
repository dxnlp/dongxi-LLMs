# S7 study notes: When optimization goes wrong: reward hacking, length and entropy

- Notebook: [`notebooks/study-path/s7_failure_modes.ipynb`](../../notebooks/study-path/s7_failure_modes.ipynb) (learner copy in
  `learner_work/notebooks/study-path/`)
- Study path stage 7 of 7 · Chapter 14 · Day 24
- Status: not started
- Main preference note (Key concepts D1–D55, RL glossary, S1 log, plan table P1–P12):
  [learning-from-preferences.md](../day-15-preference-data-and-bradley-terry/learning-from-preferences.md); RL terms:
  [glossary](../day-15-preference-data-and-bradley-terry/learning-from-preferences.md#-rl-terms-in-the-llm-setting-glossary-d39)

One learning note per study-path notebook from S2 on (learner, 2026-11-03). Questions raised while
working through **S7** are recorded here as they happen, tagged `[S7 · Part X]`. **Discussion numbers
continue one sequence across all notes** (D1–D55 in the main note, D56 onward in the stage notes): the
next number is one more than the highest in any of them, so a D-number points to exactly one place.
Animation and article requests also go into the main note's plan table and
`visuals/animations/PROPOSALS.md`. ★ = marked important by the learner.

## Key concepts

| Concept | Description | Status |
|---|---|---|

## Knowledge gaps

- none recorded yet

## S7 notebook study log

| Part | Question | Status | Exercises / predictions | Discussions and requests |
|---|---|---|---|---|
| A · Overoptimization, exactly | If π maximizes E[proxy] − β·KL exactly, how do proxy and gold reward change as β falls? | not started | — | — |
| B · Best-of-n | How much KL does best-of-n spend, and does it overoptimize too? | not started | — | — |
| C · Length hacking | What does a reward that credits length do to the optimum, and how is the length channel removed? | not started | — | — |
| D · Entropy | What does entropy collapse look like in a policy-gradient run, and what do a KL penalty and an entropy bonus change? | not started | — | — |
| E · Mitigations | KL strength, reward-model ensembles (mean vs min), length control: which gold outcomes improve, and which errors remain? | not started | — | — |
| F · Diagnostics | Which training-log patterns should make you stop and read samples? | not started | — | — |
| G · Qwen3 | Does a 1,000-pair reward model reward length, and does a short GRPO run exploit it with β = 0 vs β > 0? | not started | — | — |

## Discussions
