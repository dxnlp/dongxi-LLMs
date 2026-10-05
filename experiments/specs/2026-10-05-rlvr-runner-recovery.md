# RLVR runner completed and pending recovery

This protocol is saved before new numerical measurements. It extends the actual
`qwen_rlvr_lab.py` update path, not the separate tiny recovery exercise. Only
original randomly initialized local HF CPU models and authored arithmetic-token
fixtures may execute. No pretrained model, download, GPU, installation, service
or Git operation is authorized. Existing source changes and reports are retained.

## Numerical objective and durable phases

Split complete collection from application while preserving `update_model` as a
backward-compatible wrapper. Collection remains full-support temperature 1:
sample the rectangular group, fill already finished rows with EOS, and retain a
contiguous validity mask that includes the first declared stop. Cap-only answers
receive zero reward. Decode the pre-stop token IDs and use the existing strict
integer verifier and population-standardized group advantages. Application
retains response-mean clipped policy loss, exact full-vocabulary reference KL,
AdamW with zero weight decay and clipping at 1. No reference refresh, reward
repair, reweighting or sampling-support change is introduced.

The original frozen reference is saved with the policy and optimizer. A pending
pool binds the pre-update policy version and typed digest, original reference
digest, actual source/prompt/response IDs, stop-inclusive masks, detached old
selected likelihoods, raw decoded texts, rewards, advantages and stop IDs. It
also retains sampler RNG before and after collection and actual rectangular
generation/score work. Full reference logits need not be persisted: the original
saved reference is rescored at application. Dropout remains disabled by eval
mode during collection, validation and ratio computation.

A completed boundary `(k, None)` has policy/Adam at update k and a cursor after k
collections. A pending boundary `(k, pool for k+1)` has the same policy/Adam but
a cursor and collection RNG after the retained complete collection. Restoring
pending must consume it exactly once without a new draw. Both phases retain
global Torch, applicable CUDA and Python RNG, rollout-generator state, cumulative
collection and application work, complete committed numerical/raw history and
parent invocation. Integer counters/IDs must reject boolean aliases.

An interrupted generation is not a resumable pool. Retain completed sampled
steps and known attempted costs, identify the incomplete stage and replay from
the last durable boundary; unknown failed-call work is explicitly unavailable.
An interrupted backward or partially executed optimizer operation poisons the
live loop until a durable snapshot is restored. Such a state cannot be saved.

## Scientific identity and publication

Use the shared trusted-local snapshot API with independently retained contract,
payload SHA256, exact byte count and an explicit per-file bound. Compare observable
source/lock/parent/input-role bytes, encoded train/evaluation prompts, tokenizer
semantics/template/stops, objective, seed, geometry, dtype/backend and fixed
horizon before loading a model. After construction, also compare the effective
model configuration, policy layout, actual ordered optimizer parameter bindings
and original-reference typed digest before loading/applying snapshot state.
Only explicitly operational config path fields are excluded; model configuration
changes remain scientific. Paths, output directory, invocation time and deadline
are recorded separately. A new deadline does not authorize more scientific work.

Resume uses a new evidence directory. Commit initial, full-collection pending
and completed states exclusively before publishing a durable metric; never
overwrite an earlier snapshot. Preserve the latest durable receipt and metric
history across save, header, journal, evaluation and export failures. Final
training state is committed before final evaluation/export. Rehash actual
source/input/lock/parent files at closure, and obtain the redundant command field
from captured invocation evidence rather than rewritten `sys.argv`.

## Frozen CPU recipe

Use the existing isolated Python with offline flags, CUDA hidden and one CPU
thread. Construct Qwen3 locally: vocabulary 16, hidden width 16, intermediate
width 32, one layer, two query heads, one KV head, head dimension 8, context 32,
attention dropout 0 and FP32 weights. Original token IDs represent an authored
small integer alphabet and two declared stops. Three unequal prompt lengths
cycle deterministically; evaluation sources are independently authored. These
tokens verify recovery, not language quality.

Freeze seeds 2323 and 2324, four updates, group size 3, generation cap 4, AdamW
learning rate 0.008, zero weight decay, beta 0.02, clip 1 and a 16 MiB snapshot
envelope. Do not select a seed/checkpoint based on reward. Completed interruption
is after update 2; pending interruption is after collecting update 3 with policy
still at 2. Compare uninterrupted tails with restored tails, including exact
same-environment CPU parameters, original reference, Adam moments, RNG, cursor,
sampled IDs/masks/texts/rewards/advantages, work and numerical history. Execute
fresh-process completed and pending restarts, with a no-collection spy on the
first pending application. Timings and producing invocation IDs are not numeric
equality targets.

Compare the wrapper with the original monolithic equations using the same
initial weights, reference and generator. Include separately authored forced
stop, zero-signal and cap controls; do not report them as naturally sampled
training outcomes. Validate shape/type/content/cursor/source/reference/pool
tampering before state application, including reordered same-shaped optimizer
bindings, stale pools, shifted masks and inconsistent work/history. Exercise
save/header/metric/baseline/final/export failures and refusal of existing outputs.
Each failed test/collection command is retained as a distinct initial observation.

## Evidence and limits

New reports retain actual commands, exit status, source/spec/test/lock identities,
environment, all fixed arms and failures, fresh-process results and latest durable
receipts. Source is frozen before the final panel; owner changes require another
distinct measurement rather than rewriting earlier output. The approved command
uses `/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python`, `PYTHONPATH=src`,
offline HF flags and one CPU thread.

Acceptance establishes bounded runner-owned CPU recovery only. Mid-generation
continuation, model-scale save overhead, OS containment, total disk/work quotas,
pretrained capability, cross-machine reproducibility and CUDA/BF16 numerical
parity remain separate gates. No external campaign row is filled by these tests.
