# Persistent logical work accounting in full and LoRA SFT

This premeasurement protocol extends only the actual Chapter 9 SFT runner and
new CPU controls. The shared work/artifact/snapshot implementations and DPO source
are not edited. No pretrained bytes, GPU run, acquisition, installation, service
or Git action is authorized. Earlier SFT evidence retains its historical source
identity; a new exclusive reference records the current source.

## Operation and recovery contract

SFT selects a fixed seeded cyclic order, not random draws per update. Before
advancing that cursor or entering a forward, reserve the complete known next
accumulation window. Its exact padded forward geometry and shifted valid-target
counts can be derived from the retained order/cursor and ragged encoded records.
Never shorten accumulation, targets, data or tokenizer outputs to fit a cap.
Keep sampled_examples as selected cyclic examples; selector_steps counts those
selections, not random RNG draws. Dropout remains part of the original model's
training computation and is preserved by its existing Torch RNG snapshot.

The work schema is dongxi-sft-logical-work-v1. Its twelve independent, overlapping
dimensions are train_updates, sampled_examples, selector_steps, valid_targets,
logical_sequence_tokens, policy_forward_calls, policy_forward_positions,
evaluation_calls, evaluation_positions, generation_calls,
generation_position_upper_bound and generation_tokens. Logical sequence tokens
exclude padding while forward positions include actual padded input positions.
Valid targets count surviving shifted labels, not successful training quality.
None of these dimensions is a FLOP, backward/checkpoint-recomputation or physical
quota counter.

Reserve a complete NLL evaluation panel and a complete generation panel before
their first forward. Preserve the original SFT cached greedy generation path
(use_cache=True); use the growing uncached-prefix geometry as a conservative
upper envelope and record the actual cached input positions separately. Explicit
single-sequence greedy settings prevent an implicit beam expansion. Cap is four
for the tiny CPU observer fixture and remains the production runner's fixed64.
Retain stop-inclusive output IDs, raw stopping/decoding errors, entered and
successful forwards, and unknown failed-call internal costs separately. Returned
IDs remain known even if text decoding fails. An unexpected geometry is refused
before its backend call.

Attach an optional existing WorkLedger to the actual SFTLoop after scientific
contract construction. A budgeted snapshot includes an exact prefix; validate
limits, contract and the same physical journal before any model/Adam/RNG state
application. Later failed or open reservations remain permanently charged when
restoring an older numerical cursor into a new output. Production requires
explicit --work-limits and --work-journal-max-bytes plus the same --work-journal
on resume. Cap-file input is bounded, nonblocking, regular and no-follow, including
ancestor traversal. The old unbudgeted CPU API remains backward compatible.

The ledger's existing private-file, lock, hash-chain, byte-bound, retained-tail
and no-refund rules are reused without modification. This is cooperative
trusted-local accounting, not resistance to privileged rollback or an arbitrary
hostile worker. Artifact/log/export byte containment is a separate integration.

## Frozen CPU recipe and controls

Reuse the original randomly initialized tiny Qwen3 SFT model, tokenizer, ragged
three-record instruction suite and masks. Use seeds1212 and1213, full and LoRA
rank2 modes, four updates, microbatch1, accumulation2, AdamW lr0.003 and weight
decay0, clipping1, attention dropout0.1, SDPA, actual activation checkpointing,
CPU FP32 and the existing input-gradient requirement for LoRA. Preserve exact
assistant-body/template-ending supervision, a single causal shift, EOS/padding
distinction and original full/LoRA numerical equations.

Normal caps are declared before fitting: train_updates16, sampled_examples32,
selector_steps32, valid_targets160, logical_sequence_tokens640,
policy_forward_calls128, policy_forward_positions2048, evaluation_calls96,
evaluation_positions2048, generation_calls32,
generation_position_upper_bound1024 and generation_tokens128. The work journal
bound is1MiB; each tiny numerical snapshot remains bounded by16MiB. Separate
deliberately exhausted caps are negative refusal controls, not successful-recipe
retuning.

Compare budgeted and unbudgeted full/LoRA original-equation updates and complete
four-update numerical states exactly, including original untrained weights in
LoRA, trainable weights, Adam, seeded order/cursor, Python/Torch RNG and numerical
history. Test update2 same/fresh-process recovery while the journal retains an
entered failed forward charged after that checkpoint. Baseline observers charge
again per invocation while preserving the original training RNG. Verify whole
training/observer-panel refusal before cursor change, active selection or model
forward. Include saved-boundary, metric/observer/export and ledger persistence
failures without blessing a partial update.

Check new/copy journal and changed-cap refusal before any model application,
cached generation's actual call geometry, returned-ID decode failures, malformed
geometry rejection, and no-peer FIFO/symlink limits-file refusal before hardware
or output creation. Record exact commands, exits, environment, raw authored
failures, current source hashes before/after, immutable numerical receipts and
journal states. Passing establishes this actual CPU SFT source integration only;
pretrained/CUDA/BF16, physical containment, model-scale overhead, other runners,
external campaign outcomes and capability claims stay pending.
