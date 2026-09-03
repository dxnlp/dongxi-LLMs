# Dongxi LLMs

**From equations to experiments: architecture, supervised fine-tuning, and reinforcement learning for language models.**

`Dongxi_LLMs` is an English-first, experiment-driven learning course for technically capable Python users with a foundation in deep learning. Its first learner is Dongxi; its public purpose is to help others move beyond calling LLM APIs and understand how models are designed, trained, evaluated, and diagnosed.

## Current status

- Release target: `v0.1` public beta
- Schedule: 28 consecutive learning days
- Current position: Day 4 in progress — attention and the causal information
  boundary from first principles
- Primary machine: NVIDIA DGX Spark
- Primary model family: Qwen3

## Start here

1. Read [`BOOK.md`](BOOK.md) for the reader-facing book architecture.
2. Read [`ROADMAP.md`](ROADMAP.md) for the authoritative four-week production plan.
3. Read [`PROGRESS.md`](PROGRESS.md) for the current position and next action.
4. Read [`LEARNING_MEMORY.md`](LEARNING_MEMORY.md) for the artifact index, content ideas, and cross-machine task packets.
5. Read [`learning_artifacts/`](learning_artifacts/) for deep discussions organized by day and topic.
6. Use [`notebooks/`](notebooks/) for the book's interactive mechanism lessons.
7. Contributors and coding agents must read [`AGENTS.md`](AGENTS.md) before changing the project.

## Learning promise

At the end of the course, the learner should be able to:

- design and defend an LLM architecture under explicit constraints;
- determine what data is needed for a target capability;
- design and justify pretraining, SFT, preference, and RL recipes;
- measure model capability with a frozen evaluation contract;
- monitor training health and diagnose common failure modes;
- explain the mathematics, implementation, evidence, and trade-offs clearly.

Expert answers should follow this structure:

> Choice → rationale → evidence → trade-off → failure risk → next experiment

## Project principles

1. Predict before running an experiment.
2. Evaluation is designed before training begins.
3. Every public empirical claim links to reproducible evidence.
4. Notebooks explain experiments; reusable logic lives in importable modules.
5. Failed experiments and negative results are retained and explained.
6. English is canonical. Chinese is used selectively for social publishing. Swedish reasoning research lives in the sibling `Dongxi_LLMs_Swedish` project.
7. Existing repositories are references, not material to concatenate or paraphrase.

## Local reference projects

- `../LLMs-from-scratch`: architecture and from-scratch implementation reference
- `../reasoning-from-scratch`: Qwen3-0.6B reasoning, evaluation, GRPO, and distillation reference
- `../rlhf-book`: broad post-training and instrumentation reference
- `/home/dongxi/dgx-spark-dongxi`: validated DGX Spark platform and memory-safety layer

## Planned model ladder

| Tier | Model | Role |
|---|---|---|
| Numerical microscope | 1–10M parameters | Derivations, gradients, and mechanism tests |
| From-scratch model | `DongxiGPT`, approximately 50–150M | Architecture design and brief pretraining |
| Glass-box model | Qwen3-0.6B | Frequent SFT, DPO, RM, and GRPO experiments |
| Flagship model | Qwen3-1.7B | Validate and publish important post-training results |
| Scale-transfer model | Qwen3-4B/8B | Selected LoRA and transfer experiments |
| Teacher/judge | Approximately 14–32B | Inference, synthetic data, and evaluation assistance |

## Version scope

The four-week target is a high-quality public beta, not the end of the mastery journey. Advanced architecture variants, broad algorithm surveys, large-model sweeps, full translations, and extensive distributed training are candidates for later releases.
