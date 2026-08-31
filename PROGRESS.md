# Progress and Handoff Ledger

This file is the operational source of truth for resuming work. Update it at the end of every learning day and whenever responsibility passes to another person or agent.

## Current position

- Active release: `v0.1`
- Active day: Day 2
- Status: complete
- Current focus: Day 2 chapter, exercises, implementations, and reports complete; ready to transition
- Next action: begin Day 3 with the conceptual bridge from relative logits to softmax probabilities, then derive next-token cross-entropy and causal label shifting
- Last updated: 2026-08-31

## Four-week tracker

Status values: `pending`, `in progress`, `complete`, `blocked`.

| Day | Topic | Status | Primary evidence |
|---:|---|---|---|
| 1 | Laboratory and reproducibility | complete | Chapter: `book/chapters/01-evidence-before-optimization.md`; evidence: `experiments/reports/2026-08-29-qwen3-0.6b-sft-smoke.md` |
| 2 | Tokenization and embeddings | complete | Chapter: `book/chapters/02-text-tokens-and-embeddings.md`; reports: `experiments/reports/2026-08-30-qwen3-multilingual-tokenization.md`, `experiments/reports/2026-08-31-tokenizer-mechanics.md`, `experiments/reports/2026-08-31-embedding-gradient-paths.md`, `experiments/reports/2026-08-31-qwen3-embedding-inspection.md` |
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
| D012 | Maintain `LEARNING_MEMORY.md` as the cross-session and cross-machine artifact index and production ledger. | Topic artifacts, public-content ideas, and portable task packets must remain discoverable beyond a chat transcript without turning the book into daily notes. |
| D013 | Create and actively update `learning_artifacts/day-NN-topic/` during each lesson. | Deep discussions should be durable and reusable by topic and day, while `LEARNING_MEMORY.md` remains a compact index and production queue. |

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

### Day 02 — 2026-08-30 to 2026-08-31

- Status: complete
- Questions investigated: How do bytes, characters, BPE pieces, IDs, and embeddings differ? How are BPE vocabularies trained and frozen? Why does multilingual token efficiency differ? Where does contextual meaning emerge? How does next-token loss train repeated and tied embedding rows? How do visibility and supervision masks differ?
- Derivations completed: byte-level fallback and merge progression; `[B,T] → [B,T,d]` lookup shapes; repeated-row gradient accumulation; tied output gradient `(p_i-1[i=y])h`; causal contextualization boundary; prompt-gradient flow under response-only loss; distinction between tokenizer entry count and model row count.
- Code or content produced: multilingual tokenizer lab, transparent embedding-gradient lab, pinned Qwen3 embedding-interface inspector, four unit tests, Chapter 2, ten exercises with worked solutions, topic-organized learning artifacts, and portable article/animation packets.
- Experiments executed: pinned Qwen3 English/Chinese/Swedish tokenizer comparison; repeated lookup and tied/untied gradient-path verification; response-only masking verification; pinned Qwen3 tokenizer/embedding boundary and runtime tying inspection.
- Evidence and results: exact tokenizer round trips with 9 Chinese, 11 English, and 20 Swedish tokens in the fixed example; repeated row gradient doubled exactly; untied nonzero input rows `[1,3]` versus all six tied rows; masked prompt rows received nonzero gradients; Qwen3 input/output matrices verified as the same `[151936,1024]` parameter, with 267 model rows beyond the tokenizer's 151,669 entries.
- Failures or surprises: the predicted token-efficiency order was reversed; GPT-SW3 remained gated despite authentication; byte-level vocabulary pieces required source-offset spans for readable reporting; serialized input/output tensor names did not imply untied runtime parameters; the model/tokenizer row mismatch was larger than expected and its rationale remains unverified.
- Claims not yet validated: general language-level tokenizer efficiency, causal attribution to the tokenizer-training corpus, semantic quality from token count, exact Qwen3 training gradient magnitudes, the design reason or performance effect of 267 extra rows, and masking behavior of every trainer API.
- Decisions made: distinguish tokenizer size `V_t` from model vocabulary dimension `V_m`; say “direct lookup-path gradient” when output tying may add other paths; treat loss masking, detaching, and freezing as separate mechanisms; keep the full logits/softmax/loss derivation in Chapter 3.
- Book-facing contribution: `book/chapters/02-text-tokens-and-embeddings.md` and `book/solutions/02-text-tokens-and-embeddings.md`, supported by three experiment reports and reusable source modules.
- Public artifacts produced: BPE and cross-entropy preview animations; `X-BPE-001` and `X-EMB-001` are ready for Mac drafting; final embedding and cross-entropy animations retain their Day 3 dependency.
- Exact next action: begin Day 3 with a mechanism-level discussion of softmax as normalized competition, then derive cross-entropy, perplexity, causal shifting, and a tiny next-token model.

### Day 02 enrichment — 2026-08-31, local Mac

- Status: complete with one retained failed criterion.
- Questions investigated: How can BPE pair counts and tie handling be made
  executable? How do grapheme clusters differ from code points and bytes? What
  do normalization, leading spaces, and chat-template packaging change before
  model computation?
- Predictions recorded before execution: three BPE merges and counts; NFC/NFD
  unit counts; one family-emoji grapheme; leading-space token-identity change;
  chat packaging larger than raw text; exact decode of both normalization forms.
- Code or content produced: `src/dongxi_llms/tiny_bpe.py`,
  `src/dongxi_llms/tokenizer_mechanics_lab.py`, six focused new unit tests, a
  specification, raw JSON output, report, new learning artifact, expanded Chapter
  2 treatment, and an eleventh exercise with solution.
- Experiment executed: CPU-only pinned Qwen3-0.6B tokenizer mechanics on the Mac;
  no model weights or DGX Spark resources used.
- Evidence and results: BPE selected `h+u`, `hu+g`, `hug+s` with counts 10, 10,
  3; NFC/NFD `café` used 4/5 code points but four graphemes each; the family emoji
  used seven code points, 25 bytes, and one grapheme; `token` and ` token` mapped
  to different IDs; raw `Hello` used one token and the chat form nine.
- Failure or surprise: NFD exact source round trip failed because decode returned
  NFC; normalized offsets also omitted the original combining mark. The first BPE
  maximum was tied and required the predeclared tie policy.
- Interpretation boundary: the trace establishes generic BPE mechanics, not
  Qwen's training history; the tokenizer examples do not generalize to other
  revisions, languages, or chat templates.
- Book-facing contribution: Chapter 2 now includes graphemes, concrete
  preprocessing, tested BPE training, tokenizer-family scope, and chat packaging.
  Chapter 1 receives only a short evidence-discipline bridge; tokenizer mechanics
  remain in Chapter 2.
- Public-content contribution: the evidence gaps for expanded `X-BPE-001` are
  closed; the canonical English X Article can now be drafted for review.
- Exact next action: retain the course transition to Day 3; the independent Mac
  content lane may draft `X-BPE-001` from the approved outline and evidence.

## Learning memory and production queue

Detailed learner understanding is maintained by day and topic under
`learning_artifacts/`. Its index, the X article backlog, animation storyboards,
and cross-machine task packets are maintained in `LEARNING_MEMORY.md`.

## Open questions

- Which exact dataset and capability mix should anchor the general-assistant SFT branch?
- Which verifiable task mixture should anchor the GRPO branch beyond initial arithmetic smoke tests?
- What parameter count and corpus budget should the first `DongxiGPT` pretraining run use after Spark profiling?
- Which evaluation examples can remain private or held out to reduce contamination?
