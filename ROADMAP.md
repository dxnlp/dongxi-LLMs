# Dongxi LLMs v0.1 — Four-Week Learning and Build Roadmap

This is the authoritative plan for the first 28 days. If scope pressure appears, protect mathematical clarity, executable code, controlled experiments, and honest evaluation before adding breadth.

## Target outcome

By Day 28, release a coherent public beta covering:

- tokens, probabilities, attention, and a modern decoder architecture;
- brief from-scratch pretraining;
- evaluation-first model development;
- instruction data and supervised fine-tuning;
- preference data, reward modeling, and DPO;
- policy gradients, PPO/RLOO concepts, and GRPO/RLVR;
- training monitoring, failure analysis, and a complete checkpoint comparison.

The capstone checkpoint genealogy is:

```text
Qwen3-0.6B Base
├── general instruction SFT
│   └── preference/DPO checkpoint
└── reasoning-oriented SFT or Base
    ├── GRPO/RLVR checkpoint
    └── optional distilled checkpoint
```

Important conclusions are selectively validated on Qwen3-1.7B. The release must not wait for every larger-model experiment.

## Daily learning loop

Each normal day follows:

1. Learn and derive — 60–90 minutes.
2. Work through the active chapter's focused interactive notebook session and
   implement its reusable mechanism — 60–90 minutes.
3. Run or analyze an experiment — approximately 60 minutes.
4. Write the durable learning artifact — 30–60 minutes.
5. Record open questions and the next hypothesis — 10 minutes.

As part of the durable-artifact step, perform a brief animation-opportunity
check. Explicit mathematics important to an LLM mechanism automatically becomes
an animation candidate, even when the learner does not request one. This includes
objectives, probability transformations, gradients, tensor operations, masking
rules, optimization updates, and multi-step derivations. Consolidate overlapping
ideas rather than creating one candidate per equation. For non-mathematical
material, record a candidate only when motion would make the mechanism materially
clearer than prose or a static figure. Candidates enter
`visuals/animations/PROPOSALS.md`; animation production remains a separate,
user-approved task and does not silently expand the active learning day.

Long reference runs should start near the end of a day and be analyzed the following day. Days 7, 14, 21, and 28 are lighter consolidation days, but they remain learning days.

Every chapter receives a notebook pathway, normally a mechanism microscope, a
deliberately broken or perturbed variant, and an integration/evidence session.
The complete map is `docs/NOTEBOOK_CURRICULUM.md`. Refine and create each set as
its chapter becomes active; do not fill the repository with empty future
notebooks.

## Experiment modes

Every experiment declares one mode:

| Mode | Expected duration | Purpose |
|---|---:|---|
| Smoke | 1–15 minutes | Validate code, shapes, data, and metrics |
| Learning | 30 minutes–4 hours | Produce and understand a visible learning signal |
| Reference | 8–24+ hours | Produce a publishable comparison |

## Week 1 — Understand and build the model

### Day 1 — Establish the laboratory

Learn:

- reproducible environments, seeds, configurations, and checkpoint identity;
- experiment hypotheses and controlled comparisons;
- unified-memory monitoring on DGX Spark;
- observation versus interpretation.

Build:

- repository structure;
- course charter and mastery rubric;
- experiment specification and report templates;
- first machine/environment manifest.

Evidence of completion:

- reproduce one existing Qwen3-0.6B smoke test;
- explain exactly what it proves and what it does not prove.

### Day 2 — Tokenization and embeddings

Learn tokenization, vocabularies, embeddings, and multilingual token efficiency.

Build a tokenizer exploration lab comparing English, Chinese, and Swedish examples.

### Day 3 — Probabilities and next-token loss

Derive logits, softmax, likelihood, negative log-likelihood, cross-entropy, perplexity, causal shifting, and labels.

Build a manual numerical calculation, verify it in PyTorch, and train a tiny next-token model.

### Day 4 — Attention from first principles

Derive queries, keys, values, scaled dot products, causal masking, and attention distributions.

Build attention without `nn.MultiheadAttention`; inspect its weights and gradients; intentionally break scaling or masking.

Use the completed mechanism to derive why past per-layer keys and values remain
reusable during causal decoding, why queries are transient, and where the
Transformer architecture ends and optional inference-time KV caching begins.
Reserve detailed cache sizing, GQA, and serving benchmarks for the modern decoder
and inference-systems material.

### Day 5 — The decoder-only Transformer

Learn multi-head attention, residual streams, normalization, feed-forward networks, Transformer blocks, initialization, and output projection.

Build a minimal decoder-only Transformer and make it overfit one batch.

### Day 6 — Modern architecture design

Learn RMSNorm, RoPE, SwiGLU, GQA, QK normalization, KV caching, and parameter/FLOP/memory calculations.

Evolve the minimal model into `DongxiGPT`; define candidate 50M, 100M, and 150M
configurations; map the design to Qwen3. Then introduce recurrent depth as a
frontier design axis: distinguish fixed shared-stack loops, variable recurrence,
adaptive token-level routing, and visible chain-of-thought; account separately
for stored parameters, effective block applications, compute, latency, and
memory. Treat vendor architecture reports as unverified until primary evidence
supports them.

### Day 7 — Architecture synthesis

Explain and defend every important tensor shape and architecture choice.

After the ordinary `DongxiGPT` baseline works, specify a small optional
shared-block recurrence comparison. Keep parameter-matched, compute-matched, and
wall-clock-matched claims separate; record effective depth, measured latency,
peak memory, validation loss, and fixed samples. The experiment demonstrates a
mechanism and trade-off, not the general superiority of recurrent depth.

Publishable outputs:

- Week 1 architecture article;
- first signature Manim animation;
- architecture design review.

## Week 2 — Pretraining, evaluation, and SFT

### Day 8 — Pretraining data and recipe

Learn token budgets, shuffling, batches, AdamW, warmup, decay, clipping, BF16, validation, and checkpoint recovery.

Write a bounded `DongxiGPT` pretraining specification.

### Day 9 — Run and diagnose pretraining

Pretrain `DongxiGPT`; inspect loss, samples, activations, gradients, throughput, and memory. Compare at least one controlled change in scale, data, or recipe.

### Day 10 — Evaluation before training

Learn capability definitions, frozen splits, exact match, pass@k, confidence, contamination, sampling variance, and qualitative error analysis.

Build fast development and slower publication evaluation suites. Establish Qwen3-0.6B and 1.7B baselines where practical.

### Day 11 — Instruction-data engineering

Learn messages, roles, chat templates, BOS/EOS, padding, assistant-only loss masks, sequence packing, mixtures, deduplication, provenance, and licenses.

Build and inspect a compact instruction dataset and write its data card.

### Day 12 — SFT mechanics

Derive the SFT objective and implement a readable training loop with BF16, gradient accumulation, checkpointing, and fixed-prompt generation.

### Day 13 — Full SFT on Qwen3-0.6B

Run the base-to-assistant transition. Measure instruction following, termination, format behavior, capability gains, and regressions.

### Day 14 — Design and defend the SFT recipe

Compare full SFT with LoRA or run a learning-rate/data-mixture ablation. Start a Qwen3-1.7B validation run if the 0.6B recipe is healthy.

Publishable outputs:

- Week 2 SFT article;
- assistant-loss-mask animation;
- SFT recipe and experiment report.

## Week 3 — Preferences and policy gradients

### Day 15 — Preference data and Bradley–Terry modeling

Learn pairwise preferences, latent rewards, score margins, preference probability, annotator disagreement, and common biases.

Derive and numerically verify the Bradley–Terry loss.

### Day 16 — Reward models

Train a small preference, outcome, or process reward model. Inspect score distributions, calibration, length bias, format bias, and adversarial cases.

### Day 17 — Direct Preference Optimization

Learn KL-regularized policy optimization, reference policies, sequence log-probabilities, and the DPO derivation.

Implement DPO first on tiny tensors and then in the training stack.

### Day 18 — DPO experiment

Run DPO on Qwen3-0.6B. Compare against SFT and an appropriate control. Inspect chosen/rejected margins and independent task evaluation.

### Day 19 — Language generation as a policy

Formalize prompts as contexts, tokens as actions, completions as trajectories, and correctness as reward.

Derive REINFORCE and verify its analytical gradient on a tiny categorical policy.

### Day 20 — Baselines, RLOO, and PPO

Learn advantages, leave-one-out baselines, value functions, importance ratios, PPO clipping, KL estimators, and bias–variance trade-offs.

Compare algorithms on the same small procedural task.

### Day 21 — Policy-gradient synthesis

Explain the relationship among SFT, DPO, REINFORCE, RLOO, and PPO.

Publishable outputs:

- Week 3 preference/RL article;
- reward-margin or PPO-clipping animation;
- controlled algorithm-comparison report.

## Week 4 — GRPO, failures, and capstone

### Day 22 — GRPO derivation

Learn grouped rollouts, group-relative advantages, normalization, zero-variance groups, verifiable rewards, and token- versus sequence-level losses.

Implement a complete numerical GRPO example.

### Day 23 — Qwen3-0.6B GRPO/RLVR

Run GRPO with a verifiable task. Compare rollout count or response-length settings and inspect advantage distributions.

### Day 24 — Reward hacking and instability

Study KL growth, entropy collapse, reward hacking, length bias, insufficient exploration, and reward/evaluation divergence.

Create an exploitable reward deliberately, observe the failure, and test a repair.

### Day 25 — Rollout systems and monitoring

Learn vLLM integration, policy-weight synchronization, batching, colocated memory, checkpoint recovery, and throughput bottlenecks.

Compare the readable course implementation with the Open Instruct stack validated in `dgx-spark-dongxi`.

### Day 26 — Distillation and inference-time improvement

Study teacher-generated reasoning, rejection sampling, self-consistency, best-of-N, offline/on-policy distillation, and accuracy-versus-token-cost trade-offs.

Run one bounded comparative lab.

### Day 27 — Capstone evaluation

Evaluate the checkpoint genealogy with the frozen panel. Complete error analysis, ablations, training reports, reproducibility instructions, and model cards.

### Day 28 — Defense and release

Conduct a technical model-development defense. Complete the editorial and executable-code checks, publish `v0.1`, and record the retrospective and `v0.2` roadmap.

Publishable outputs:

- Week 4 GRPO/capstone article;
- GRPO rollout-to-advantage animation;
- public release thread and selected Chinese social assets.

## Signature visual plan

The first release targets four excellent animations rather than one animation per lesson:

1. Tokens → logits → probabilities → target loss.
2. Attention and causal masking.
3. Chat template → tokens → assistant-only SFT mask.
4. Prompt → rollout group → rewards → advantages → GRPO update.

A fifth PPO-clipping or Bradley–Terry animation is optional if time permits.

## Publishing cadence

- Private learning log: daily.
- Concise public posts: two or three per week.
- Substantial English article: one per week.
- Major visual or animation: one per week.
- Selected Chinese localization: derived from validated canonical material.
- Release article/thread: Day 28.

## Definition of ready on Day 28

`Dongxi_LLMs v0.1` is ready when it contains:

- a clear README and navigable curriculum;
- a reproducible DGX Spark environment path;
- approximately 12–14 coherent core lessons;
- importable architecture and training code;
- tested architecture, SFT, DPO, and GRPO labs;
- a frozen evaluation panel;
- experiment specifications and reports;
- a checkpoint genealogy and model cards;
- at least four high-quality visual explanations;
- four substantial English articles;
- selected Chinese social-media packages;
- exercises and diagnostic questions;
- an explicit `v0.2` backlog.

## Deferred beyond v0.1

- exhaustive preference-optimization variants;
- comprehensive MoE and hybrid-architecture implementation;
- large distributed pretraining;
- broad 4B/8B/32B sweeps;
- full Chinese or Swedish course translations;
- production deployment and serving as a major course section;
- the complete Swedish reasoning-model research program.
