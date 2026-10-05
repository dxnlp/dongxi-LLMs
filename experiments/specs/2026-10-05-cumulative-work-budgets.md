# Cumulative logical work reservation in the DPO runner

This protocol precedes new measurements. It extends only a generic cooperative
work ledger and the actual DPO runner. No pretrained model, GPU, download,
installation, environment change, service or Git action is authorized. Historical
source and reports retain their original identities. This is not whole-campaign,
FLOP, activation-recomputation, physical-quota or arbitrary-worker containment.

## Reservation and recovery contract

Reserve an entire known next operation before its first draw/forward. DPO reserves
the complete accumulated update from componentwise maxima of the frozen encoded
training branches, rather than sampling first or shortening its objective when a
cap is reached. Evaluation reserves a whole panel. Greedy uncached generation
uses the declared prompt lengths and cap to reserve calls, stop-inclusive output
tokens and growing-prefix positions conservatively. Enforce actual generation
forward geometry before each call so an unexpected decoding path cannot silently
exceed its reservation. The original masks, loss, reference, optimizer and
sampling stream remain unchanged.

Each reservation permanently consumes capacity. Successful measured work and
known partial work are separate; unused conservative reservation is not refunded.
Open/crashed and failed attempts retain uncertain upper bounds. Refuse a crossing
operation before work. A failed ledger write poisons its live handle. Retain raw
bytes and reject incomplete/invalid journals instead of truncating away attempts.
The summary separates reserved, completed, known_partial, attempted_upper and
uncertain_upper. A finalized failed call records entered logical input geometry
separately from successful returns; unknown crash entry retains the full upper.
The permanent reservation does not shrink to that later observation.

Use an exclusive, bounded, hash-chained fsynced JSONL journal with a retained
single-writer flock, regular-file/no-follow check and device/inode anchor. Limits,
schema and journal bound belong to the scientific contract. Completed snapshots
retain the ledger identity and exact prefix summary. Resume must reuse that
anchored journal, validate the prefix before model state application and include
all later charged attempts. A new output directory, copied journal or changed
limits cannot refill capacity. Explicit filesystem rollback/malicious privileged
mutation is not prevented by a cooperative trusted-local ledger; invalid tails
fail closed and need a separately approved reconciliation, not an automatic reset.
The journal must be owner-UID, mode 0600, single-link and regular; its final
directory must be owner-UID and private. Open every ancestor no-follow via retained
directory descriptors, refuse FIFO targets nonblocking, and check live pathname,
ancestor identity, mode and same-inode raw bytes before append/snapshot. Refuse
accidental mutation of public accounting maps before it can refill a limit.
Pinned paths plus repeated checks do not eliminate a hostile check/syscall race.

Production DPO must supply explicit work limits and journal-byte bound; old
unbounded CPU references may pass no ledger and retain their original numerical
interfaces. Charge baseline evaluation again on every new invocation. Final
evaluation and export failures leave training snapshots and spent work anchored.
Export/storage resource accounting belongs to the separate artifact package.

The DPO dimensions are train_updates, sampled_examples, sampler_draws,
valid_targets, logical_sequence_tokens, policy_forward_calls,
policy_forward_positions, reference_forward_calls, reference_forward_positions,
evaluation_calls, evaluation_positions, generation_calls,
generation_position_upper_bound and generation_tokens. Role/global and
evaluation/generation dimensions overlap intentionally: apply each cap
independently, never sum them as FLOPs. Backward recomputation is not measured.

## Frozen CPU controls

Reuse the original random Qwen3 seed 1818, six updates, accumulation 2, beta 0.2,
learning rate 0.008, weight decay 0.01 and clipping 1. Enable actual production
activation checkpointing and disable caches; weights remain FP32 on CPU. Keep
the same authored unequal branches, EOS/prompt/padding masks and original frozen
reference. Compare budgeted versus existing unbudgeted numerical history, policy,
reference, Adam and RNG exactly. Test same/fresh-process update 3 recovery under
the same physical ledger, preserving a deliberately failed/retried reservation.

Normal fixture limits are fixed before collection: train_updates 12,
sampled_examples 24, sampler_draws 24, valid_targets 160,
logical_sequence_tokens 512, policy_forward_calls 128,
policy_forward_positions 1024, reference_forward_calls 96,
reference_forward_positions 1024, evaluation_calls 64, evaluation_positions 512,
generation_calls 16, generation_position_upper_bound 512 and generation_tokens
64. The journal bound is 1 MiB. Separately declared tiny caps intentionally test
refusal; they are negative controls, not a retuned successful training recipe.

Shared tests cover strict integer/key/overflow gates, atomic multidimensional
refusal, success/partial/uncertain separation, crash-open tickets, concurrent
writer refusal, copied/new-ledger/changed-limit rejection, snapshot-prefix replay,
corrupt/torn tails, bounded writes and persistence failures. Actual DPO controls
must show no draw/forward on a crossing update, full accumulated-objective
reservation, charged baseline/final panels and retries, generation bound checks,
before-application ledger binding, and preserved recovery after evaluation failure.
Record exact commands, exits, environment/source/spec identities, raw failures,
ledger vectors, immutable snapshots and before/after source hashes.

Passing establishes cooperative logical reservation in this DPO source only.
Other runners, pretrained/CUDA compatibility, model-scale overhead, whole-job
physical resource enforcement and the separately authorized platform/pilot gates
remain pending. Do not fill any external campaign outcome from these CPU controls.
