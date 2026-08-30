# Progress and Handoff Ledger

This file is the operational source of truth for resuming work. Update it at the end of every learning day and whenever responsibility passes to another person or agent.

## Current position

- Active release: `v0.1`
- Active day: Day 2
- Status: in progress
- Current focus: embedding lookup, batching and masks, and converting the completed multilingual tokenizer exploration into a reproducible lab and book chapter
- Next action: complete the embedding-and-batching lesson, then implement and record the Day 2 tokenizer exploration lab
- Last updated: 2026-08-30

## Four-week tracker

Status values: `pending`, `in progress`, `complete`, `blocked`.

| Day | Topic | Status | Primary evidence |
|---:|---|---|---|
| 1 | Laboratory and reproducibility | complete | Chapter: `book/chapters/01-evidence-before-optimization.md`; evidence: `experiments/reports/2026-08-29-qwen3-0.6b-sft-smoke.md` |
| 2 | Tokenization and embeddings | in progress | — |
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
| D011 | Course material is developed book-first rather than as chronological daily notes. | Each learning day must strengthen a coherent reader-facing narrative; logs and reports provide evidence but do not replace structured chapters. |
| D012 | Maintain `LEARNING_MEMORY.md` as the cross-session and cross-machine learning and production ledger. | Deep discussions, corrected mental models, public-content ideas, and portable task packets must survive beyond a chat transcript without turning the book into daily notes. |

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
- Book-facing contribution:
- Public artifacts produced:
- Exact next action:
```

## Daily log

### Day 01 — 2026-08-29

- Status: complete
- Questions investigated: What identifies an experiment? What do smoke tests prove? How should unified memory be monitored? How do observations differ from interpretations?
- Derivations completed: decomposed training memory into fixed and shape-dependent contributors; distinguished overlapping system, cgroup, and CUDA accounting domains.
- Code or content produced: book architecture; Chapter 1; laboratory appendix; exercise solutions; course charter; mastery rubric; repository structure; experiment specification/report templates; environment-manifest collector; two environment manifests; a completed smoke-test specification; and its report.
- Experiments executed: native PyTorch GPU-stack verification; pinned open-instruct GPU-stack verification; three-step Qwen3-0.6B BF16 full-SFT smoke profile.
- Evidence and results: smoke test passed with status 0; three finite losses (`0.360384`, `0.264207`, `0.357005`); 118.84 GiB starting and 109.74 GiB minimum `MemAvailable`; 4.03 GiB cgroup peak; 6.42 GiB CUDA-visible peak; zero cgroup memory events; model saved. See `experiments/reports/2026-08-29-qwen3-0.6b-sft-smoke.md`.
- Failures or surprises: loss was non-monotonic, which did not violate the specification; PyTorch emitted an `sm_121` capability-range warning despite the validated operations passing; the compatibility SFT trainer is deprecated upstream.
- Claims not yet validated: capability improvement, long-run convergence and stability, exact cross-host reproduction, larger batch/sequence safety, and recipe optimality.
- Decisions made: use immutable revision identities and lock hashes; retain all three non-additive DGX Spark memory views; require precommitted success and failure criteria; keep raw platform outputs separate from durable course reports.
- Book-facing contribution: `book/chapters/01-evidence-before-optimization.md`, supported by `book/appendices/a-laboratory-setup.md` and `book/solutions/01-evidence-before-optimization.md`.
- Public artifacts produced: Chapter 1 draft, Appendix A draft, and Chapter 1 exercises with solutions; not yet released externally.
- Exact next action: begin Day 2 by predicting token-count differences for fixed English, Chinese, and Swedish strings before inspecting Qwen3 tokenization.

## Learning memory and production queue

Detailed learner understanding, the X article backlog, animation storyboards, and
cross-machine task packets are maintained in `LEARNING_MEMORY.md`.

## Open questions

- Which exact dataset and capability mix should anchor the general-assistant SFT branch?
- Which verifiable task mixture should anchor the GRPO branch beyond initial arithmetic smoke tests?
- What parameter count and corpus budget should the first `DongxiGPT` pretraining run use after Spark profiling?
- Which evaluation examples can remain private or held out to reduce contamination?
