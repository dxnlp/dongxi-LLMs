# DPO CPU activation checkpointing recovery

This protocol is saved before new measurements. It closes only the CPU
activation-checkpointing gap explicitly retained in the released DPO recovery
report. The released runner, earlier tests and reports remain unchanged. No GPU,
pretrained weights, acquisition, installation, environment mutation, service or
Git action is included.

## Fixed recipe and actual production mode

Reuse the original seed 1818, authored train/validation branches, vocabulary 16,
width 16, intermediate width 32, one-layer Qwen3, two query heads, one KV head,
head dimension 8, context 32 and zero dropout. The model is randomly initialized
locally. Use CPU FP32, six updates, accumulation 2, beta 0.2, AdamW learning rate
0.008, weight decay 0.01, clipping 1 and a 16 MiB snapshot envelope. Keep the
original frozen reference, unequal response lengths, real EOS supervision,
EOS-valued padding, single causal shift and summed response DPO objective.

Set `model.config.use_cache=False` and
`reference.config.use_cache=False`; call `model.gradient_checkpointing_enable()`
exactly as production does. The reference remains frozen/eval and does not enable
activation checkpointing. Actual updates use the released
`completed_dpo_update` and `train_completed_updates`, not a separate learner.
The effective contract binds the observed checkpointing flag and cache settings.
Record the new test/fixture source identities alongside the unchanged release.

## Predeclared acceptance

Compare the actual first checkpoint-on update with independently written original
response likelihood and softplus DPO equations. Assert finite gradients, retained
EOS and unchanged reference, including Adam moments. Compare update 3 restoration
through update 6 in the same process and a separately launched Python process.
Within the checkpoint-on mode require exact equality of policy/reference/Adam,
sampler/global RNG, selected indices, cumulative work and numerical history,
explicitly including the first post-restore loss. Timing, invocation identity,
parent receipt and output paths are not equality targets.

Also compare checkpoint-off versus checkpoint-on under the otherwise identical
recipe and explicit cache-disabled settings. Before measurement, fix the floating
comparison at absolute tolerance 1e-6 and relative tolerance 1e-6 for losses,
margins, gradient norms, policy/reference tensors and Adam moments. Integer work,
sample indices and RNG must match exactly. Report observed maximum differences;
do not infer bitwise equality from tolerance acceptance. Wrong-mode restore must
fail against the independently recomputed expected contract before state
application. Existing files and earlier evidence remain untouched.

Retain the completed update 0, 3 and 6 snapshots. Inject a metric interruption
after durable update 3, retain the actual error and row history, then resume in a
new output directory. Record actual commands, interpreter/package/lock/source
identities, exit/runtime, all checkpoint identities and before/after source
hashes. Preserve a failed test or collection as a distinct diagnostic; do not
retune the recipe or widen tolerances after observing results.

## Evidence boundary

Passing establishes CPU FP32 activation-checkpointing execution and recovery
through the actual runner functions. It does not establish pretrained-model
compatibility, CUDA/BF16 parity, model-scale memory savings or overhead,
cross-machine determinism, whole-job containment or language quality. The earlier
checkpoint-off report remains historically correct and is not rewritten.
