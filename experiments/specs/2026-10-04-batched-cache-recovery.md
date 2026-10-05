# Ragged cached generation and exact CPU recovery

This original specification is saved before the first measurements. The lab
reuses the course decoder's learned modules, without changing its historical
implementation. It is a tiny CPU correctness experiment, not a serving-engine
benchmark, pretrained result or Spark DPO/RLVR resume implementation.

## Fixed model and generation protocol

Use float64 CPU, one Torch thread, seed2501, vocabulary12, width16, two blocks,
four query heads, two KV heads, head width4, hidden32, context24, untied output,
RMSNorm/RoPE/SwiGLU and no dropout. Original prompt rows are short `[1,3]`,
medium `[1,4,5,6]`, and long `[1,7,8,9,10,11]`. EOS and PAD both have ID0;
validity is determined by masks, never by testing the ID alone. Left padding
uses position0 for dummy entries and positions0..length-1 for real entries.

Compare four fixed paths: single/uncached, single/cached, batch/uncached and
batch/cached. Greedy IDs and every selected-token raw log probability must
agree up to each row's stop, with absolute and relative tolerances1e-10. Full
vocabulary temperature1 is the primary behavior. Also compare sampled paths
using independent row-keyed generators, seed2510. Reordering must preserve each
row's trajectory; changing batch layout is not guaranteed bitwise numerical
equivalence. Repeat the four greedy paths five times for small descriptive
runtime samples; do not select a favorable repetition or claim a speedup.

A separate explicit stopping control restricts the short row to EOS on its
first draw, the medium row to tokens3/4 until its fourth-draw cap, and the long
row to token3 followed by EOS. These are intentionally forced conditional
controls, not spontaneous learned stopping. Preserve raw-model and transformed
behavior likelihoods separately. Active rows are physically compacted after
stopping. Count actual prefill/decode/full-prefix forwards, valid input positions,
forwarded rectangular positions, padding, useful generated tokens including
EOS, and real compact KV tensor payload/peak bytes. Payload excludes allocator,
attention expansion, snapshots, parameters and general process memory.

## Generation interruption and identity

Interrupt sampled generation after two global draws, save the active cache,
next logits, prefixes, draw cursors, every generator state and raw records, then
resume. Saved-cache continuation must match bitwise under this same environment.
An explicitly requested full-prefix cache rebuild is checked to1e-10, including
selected IDs; it is not described as bitwise. Prefix, ordered row identity,
policy parameter bytes/version, token/template/position/stop/support contract,
precision and environment are validated before reuse. Changing weights,
interface, a cursor or RNG state must reject or demonstrably change continuation.

Snapshots are data-only trusted local Torch state loaded with `weights_only=True`,
a pre-load external file digest, bounded file size and an independently expected
contract. Metadata alone is not an authenticated checkpoint. Interrupted and
rejected cases remain in the report; no arbitrary external pickle is accepted.

## Actual tiny DPO and RLVR recovery

Seeds2521/2522 each receive six completed AdamW updates per objective, split after
three completed updates and again after collecting the fourth update's batch.
Use lr0.008, weight decay0, foreachFalse and gradient clip1. DPO scores chosen
and rejected atom+EOS branches with beta0.5 and a frozen initial reference.
RLVR samples four responses, cap3, conditional support `{EOS,3,4}`, temperature1,
per-response-mean clipped objective epsilon0.2, exact conditional KL beta0.02 and
detached population-standardized group advantages. Reward is the independently
fixed rule: first answer equals the requested token3/4 and the second token is
actual EOS. No warm-up, best seed or outcome-dependent recipe selection.

Three original source rows use the fixed short/medium/long prompts; alternate
targets3/4/3. Shuffle order, data cursor/RNG, rollout seed generator, global Torch
RNG, policy/reference/optimizer bytes, objective contract, completed version,
history and any pending collected batch are checkpoint state. A post-collection
interruption retains the pending rollout and resumes its same update before any
new draw. Completed-boundary replay must reproduce the next batch, next rollout,
loss history and final parameters exactly. Omitted optimizer/data-cursor/RNG
controls are explicitly broken illustrative branches, not supported resume modes.
Version/interface/stale-policy mismatches must be rejected. These interfaces do
not retrofit or validate the optional pretrained Spark runners.

## Evidence and lesson

Retain all four generation comparisons, forced controls, interruption records,
rejections, all two-seed DPO/RLVR histories and pending rollout records. Export
five explanatory figures from a fresh isolated CPU notebook. Independently test
unpadded agreement with the original decoder, causal ragged masks, GQA storage,
reordering/stopping, exact serialization and all failure controls. Record source
identities, actual invocation, environment and notebook manifest separately from
claims about transfer, production performance or learner mastery. No downloads,
GPU/Mac/API/server, installs, Git changes or animations are authorized here.
