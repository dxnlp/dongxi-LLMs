# Evaluation, instruction data and SFT course experiment contract

Authored 2026-10-04 for Chapters 7–9 / Days 10–14. This is a course verification contract, not an externally preregistered capability study.

## Executed CPU scope

Hypotheses:

1. The pass@k product equals finite combinatorial subset counting.
2. Supervision masks align to next-token targets exactly once and preserve the assistant end marker.
3. Token-weighted accumulation reproduces a joint batch gradient within numerical tolerance.
4. Zero-initialized LoRA B preserves the initial base function and creates the expected first-step gradient asymmetry.
5. A fixed 100-update full-SFT microscopic recipe lowers the fixture answer NLL.

Inputs are original teaching fixtures. The random decoder has width24, one block, three heads, learned positions, vocabulary22, hidden48 and tied input/output weights. Full mode trains6,120 scalars. LoRA adds rank-four adapters to Q/V, freezes the6,120 base scalars and trains384 adapter scalars.

Freeze seed1212, AdamW0.015, weight_decay0, clipping1.0, four seen copy requests and eight supervised targets/update. Run100 updates, CPUFP32, one Torch thread. Record every loss and gradient norm; show all four predetermined completions rather than selecting successful outputs.

Failure conditions are nonfinite state, incorrect mask/shift, missing termination supervision, accumulation mismatch, failed LoRA equivalence or restart mismatch. Fixture quality outcomes are measurements, not required implementation invariants.

The actual report records the results and source hashes. It cannot establish pretrained transfer, unseen instruction competence, calibration or Spark throughput.

## Proposed real-model Spark scope

Objective: determine whether a pretrained base gains narrow copy/reverse/extraction interface behavior on held-out values while preserving format and termination.

Model starts:

- Qwen/Qwen3-0.6B-Base @ da87bfb608c14b7cf20ba1ce41287e8de496c0cd.
- Optional later Qwen/Qwen3-1.7B-Base @ ea980cb0a6c2ae4b936e82123acc929f1cec04c1.

These exact official snapshots were retrieved on2026-10-04. They are base checkpoints; the corresponding models without the Base suffix are already post-trained.

Data: deterministic original generator in scripts/prepare_chapter09_instruction_fixture.py. Train240 examples/80 value groups, development60/20 groups, publication120/40 groups. All three task families for one value remain together. Shared task templates limit transfer claims to new values. Record JSONL hashes and license/card status.

Interface: explicit experiments/data/instruction_interface_v1.jinja, existing im_start/im_end vocabulary markers, completed assistant demonstrations, body/end/separator supervision, no user/system/header loss, one explicit shift, right padding, no packing, reject overlength.

Smoke/profile:20 updates, microbatch1, accumulation4, context256, BF16weights/autocast, FP32cross-entropy, SDPA, gradient checkpointing, AdamW2e-5, no weight decay, clipping1, seed1212, 900-second invocation cap and sampled host reserve>=25GiB.

Proposed learning comparison:400 updates and3,600-second invocation cap, identical original data order and supervised exposure, full versus rank8 Q/V LoRA. The learning rate is a fixed-recipe control; a tuned-method claim would need separately declared searches. Report actual target presentations, updates, runtime, memory and parameter counts.

Evaluation: full-development answer NLL plus a fixed first-eight-ID greedy diagnostic grid before/after,64new tokens, raw IDs/text/ending/answer exact match. The publication suite is untouched by the training runner; run it only after recipe selection. Item group is the uncertainty unit. Add independent general-capability regression data before claiming broad assistant improvement; this narrow suite alone cannot measure it.

Larger transfer gate: finite state, preserved target alignment, successful restart, memory reserve, verified ending IDs, improved held-out task slices and separately declared regression tolerance. No larger validation run is authorized merely by creation of these materials.

## Reproduction and outputs

Use the commands in book/labs/09-supervised-fine-tuning.md. The optional CUDA runner refuses unpinned revisions, nonempty fresh output directories and incompatible template prefixes. It records config/data/template identities, per-update metrics, fixed-grid outputs and full restart state.

Full mode exports an HF policy directory. LoRA exports an adapter requiring the recorded base; merge/save must be an explicit derived step for downstream full-checkpoint trainers.

Runtime and memory checks run between operations; a hard external deadline requires a job supervisor. GPU execution was not performed for this writing task. The existing TinyStories results remain separate historical evidence.
