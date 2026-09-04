# Dongxi LLMs Book Architecture

This document is the narrative map for the `v0.1` book. `ROADMAP.md` defines the
28-day learning and production schedule; this file defines how that work becomes a
coherent reader-facing book. A learning day may contribute to several book
locations, and a chapter may synthesize several learning days.

## Reader journey

The book moves through one continuous argument:

1. establish what counts as trustworthy evidence;
2. understand how text becomes next-token predictions;
3. build and pretrain a modern decoder;
4. evaluate behavior before changing it;
5. teach desired behavior with demonstrations and preferences;
6. optimize language generation as a policy;
7. diagnose failures and defend a complete model-development process.

## Planned contents

### Front matter

- Preface — From API User to Model Developer
- How to Use the Book and Companion Repository
- Notation and Experimental Conventions

### Part I — From Text to a Modern Decoder

1. **Evidence Before Optimization**
   Experiment identity, hypotheses, controls, reproducibility, observations versus
   interpretations, smoke tests, and the first Qwen3 case study. Primarily Day 1.
2. **Text, Tokens, and Embeddings**
   Tokenization, vocabularies, multilingual efficiency, and learned vector lookup.
   Primarily Day 2.
3. **Learning the Next Token**
   Logits, softmax, likelihood, cross-entropy, perplexity, causal shifting, and a
   tiny next-token model. Primarily Day 3.
4. **Attention and the Causal Information Boundary**
   Queries, keys, values, scaling, masks, attention distributions, gradients, and
   deliberately broken variants. Connect causal immutability to request-local KV
   caching and inference optimization, while reserving detailed cache accounting
   and modern attention variants for the decoder-design chapters. Primarily Day 4.
5. **Building a Modern Decoder**
   Transformer blocks, residual streams, normalization, feed-forward layers,
   RMSNorm, RoPE, SwiGLU, GQA, QK normalization, KV caching, and `DongxiGPT`.
   After the standard decoder is secure, a time-stamped frontier section uses
   recurrent depth and looped Transformers to separate stored parameters,
   effective depth, and per-token compute; it contrasts fixed stack reuse,
   variable recurrence, adaptive token-level routing, and visible token-space
   chain-of-thought. Synthesizes Days 5–7.

### Part II — From Base Model to Assistant

6. **Pretraining as a Controlled System**
   Data and token budgets, batching, AdamW, schedules, clipping, precision,
   checkpoint recovery, scaling, and diagnosis. Synthesizes Days 8–9.
7. **Evaluation Is a Contract**
   Capability definitions, frozen splits, metrics, uncertainty, contamination,
   sampling variance, and error analysis. Primarily Day 10.
8. **Instruction Data as an Interface**
   Messages, roles, chat templates, loss masks, packing, mixtures, provenance, and
   data cards. Primarily Day 11.
9. **Supervised Fine-Tuning**
   Objective, implementation, base-to-assistant experiment, ablations, capability
   gains, and regressions. Synthesizes Days 12–14.

### Part III — Learning from Preferences and Rewards

10. **Preferences and Reward Models**
    Bradley–Terry modeling, preference data, disagreement, calibration, bias, and
    adversarial cases. Synthesizes Days 15–16.
11. **Direct Preference Optimization**
    KL-regularized optimization, reference policies, sequence log-probabilities,
    the DPO derivation, implementation, and controlled experiment. Days 17–18.
12. **Language Generation as a Policy**
    Trajectories, REINFORCE, baselines, RLOO, PPO, importance ratios, clipping,
    KL estimation, and bias–variance trade-offs. Synthesizes Days 19–21.

### Part IV — Verifiable Rewards and Complete Systems

13. **Group-Relative Policy Optimization**
    Grouped rollouts, relative advantages, zero-variance groups, verifiers,
    token-level losses, derivation, and Qwen3 RLVR experiment. Days 22–23.
14. **When Optimization Goes Wrong**
    Reward hacking, KL growth, entropy collapse, length bias, exploration,
    rollout architecture, memory, synchronization, and monitoring. Days 24–25.
15. **Distill, Evaluate, and Defend**
    Rejection sampling, self-consistency, best-of-N, distillation, checkpoint
    genealogy, capstone evaluation, technical defense, and release. Days 26–28.

### Appendices

- Appendix A — Laboratory Setup and Reproducible Runs
- Appendix B — Mathematical and Tensor Notation
- Appendix C — Evaluation and Experiment Templates
- Appendix D — Reproduction Commands and Environment Locks

## Chapter design contract

Each chapter should form an argument rather than a pile of artifacts. Use the
following elements when they serve the material:

1. motivating question;
2. explicit learning outcomes and prerequisites;
3. intuition followed by precise definitions;
4. derivations with symbols and shapes defined;
5. a transparent implementation;
6. predictions made before experiments;
7. measurements, representative outputs, and limitations;
8. exercises that require explanation, calculation, or modification;
9. a summary that connects to the next chapter.

Every chapter has a first-class interactive notebook pathway, normally two to
three focused sessions rather than one monolithic notebook. Mechanism-heavy
chapters should progress from a transparent mechanism microscope to a deliberate
perturbation or failure and then to integration and evidence. Notebooks should
be linked from the relevant chapter and solutions, remain self-contained with
adjacent runnable reference answers, and reuse importable logic rather than
becoming disposable scratch work. The course-wide plan is
`docs/NOTEBOOK_CURRICULUM.md`. Notebooks support the narrative argument; they do
not replace it.

Detailed logs, manifests, and exhaustive telemetry remain in the companion
repository. The main prose includes only enough evidence to support its claims and
links to the complete record.

## Day 1 placement

Day 1 contributes to:

- the course charter and mastery rubric in `docs/`;
- Chapter 1, `book/chapters/01-evidence-before-optimization.md`;
- Appendix A, `book/appendices/a-laboratory-setup.md`;
- exercises and solutions in `book/solutions/`;
- the specification, report, and manifests in the companion experiment tree.

## Day 2 placement

Day 2 contributes to:

- Chapter 2, `book/chapters/02-text-tokens-and-embeddings.md`;
- exercises and worked solutions in
  `book/solutions/02-text-tokens-and-embeddings.md`;
- transparent tokenizer, embedding-gradient, and Qwen3 interface implementations
  under `src/dongxi_llms/`;
- three specifications and reports covering multilingual tokenization, gradient
  routing and masking, and the pinned Qwen3 tokenizer/embedding boundary;
- reusable BPE and embedding article and animation packets in
  `LEARNING_MEMORY.md`.

## Day 3 placement

Day 3 contributes to:

- Chapter 3, `book/chapters/03-learning-the-next-token.md`;
- worked solutions in `book/solutions/03-learning-the-next-token.md`;
- three first-class interactive lessons under `notebooks/day-03/`;
- the reusable two-logit distribution lab and tests under `src/` and `tests/`;
- a precommitted specification and report demonstrating $p-q$ gradient agreement
  and empirical 70/30 distribution learning;
- refinements in Chapter 2 on categorical token IDs, BPE atoms, and the
  vocabulary-versus-sequence-length trade-off;
- approved probability, cross-entropy, and next-token animation packets in
  `LEARNING_MEMORY.md`, with production retained for the Mac Studio.
