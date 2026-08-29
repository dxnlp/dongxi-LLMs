# Progress and Handoff Ledger

This file is the operational source of truth for resuming work. Update it at the end of every learning day and whenever responsibility passes to another person or agent.

## Current position

- Active release: `v0.1`
- Active day: Day 0
- Status: planning complete; ready to begin Day 1
- Current focus: establish the laboratory and repository artifact structure
- Next action: execute Day 1 from `ROADMAP.md`
- Last updated: 2026-08-29

## Four-week tracker

Status values: `pending`, `in progress`, `complete`, `blocked`.

| Day | Topic | Status | Primary evidence |
|---:|---|---|---|
| 1 | Laboratory and reproducibility | pending | — |
| 2 | Tokenization and embeddings | pending | — |
| 3 | Probabilities and next-token loss | pending | — |
| 4 | Attention from first principles | pending | — |
| 5 | Decoder-only Transformer | pending | — |
| 6 | Modern architecture design | pending | — |
| 7 | Architecture synthesis | pending | — |
| 8 | Pretraining data and recipe | pending | — |
| 9 | Pretraining run and diagnosis | pending | — |
| 10 | Evaluation before training | pending | — |
| 11 | Instruction-data engineering | pending | — |
| 12 | SFT mechanics | pending | — |
| 13 | Qwen3-0.6B full SFT | pending | — |
| 14 | SFT recipe defense | pending | — |
| 15 | Preference data and Bradley–Terry | pending | — |
| 16 | Reward models | pending | — |
| 17 | DPO derivation and implementation | pending | — |
| 18 | DPO experiment | pending | — |
| 19 | Language generation as a policy | pending | — |
| 20 | Baselines, RLOO, and PPO | pending | — |
| 21 | Policy-gradient synthesis | pending | — |
| 22 | GRPO derivation | pending | — |
| 23 | Qwen3-0.6B GRPO/RLVR | pending | — |
| 24 | Reward hacking and instability | pending | — |
| 25 | Rollout systems and monitoring | pending | — |
| 26 | Distillation and inference scaling | pending | — |
| 27 | Capstone evaluation | pending | — |
| 28 | Technical defense and release | pending | — |

## Durable decisions

| ID | Decision | Reason |
|---|---|---|
| D001 | English is the canonical course language. | Precise technical communication and global accessibility. |
| D002 | Chinese is a selective social/localization layer. | A large portion of the existing audience is Chinese-speaking without requiring two canonical courses. |
| D003 | Swedish work lives in `Dongxi_LLMs_Swedish`. | It is a distinct applied research journey toward Swedish reasoning capability. |
| D004 | The first learner is a capable Python user with deep-learning foundations. | This matches Dongxi and the intended public audience. |
| D005 | SFT and RL receive the greatest depth; pretraining remains concise but empirical. | The objective is modern post-training mastery. |
| D006 | Qwen3 is the central model family. | It connects readable architecture code, 0.6B reasoning experiments, and 1.7B post-training. |
| D007 | Qwen3-0.6B is the glass-box model; Qwen3-1.7B is the flagship validation model. | Fast iteration and meaningful validation need different scales. |
| D008 | Evaluation is frozen before a training run. | Training reward or loss alone cannot establish capability improvement. |
| D009 | The 28-day deliverable is a public beta, not a claim of completed lifelong mastery. | Four weeks provides urgency while preserving technical honesty. |
| D010 | `dgx-spark-dongxi` remains the independent platform layer. | Hardware compatibility and course content should evolve independently. |

## End-of-day update template

Copy this block below the daily log heading after each session:

```markdown
### Day NN — YYYY-MM-DD

- Status: complete | partial | blocked
- Questions investigated:
- Derivations completed:
- Code or content produced:
- Experiments executed:
- Evidence and results:
- Failures or surprises:
- Claims not yet validated:
- Decisions made:
- Public artifacts produced:
- Exact next action:
```

## Daily log

No learning days have been executed yet.

## Open questions

- Which exact dataset and capability mix should anchor the general-assistant SFT branch?
- Which verifiable task mixture should anchor the GRPO branch beyond initial arithmetic smoke tests?
- What parameter count and corpus budget should the first `DongxiGPT` pretraining run use after Spark profiling?
- Which evaluation examples can remain private or held out to reduce contamination?
