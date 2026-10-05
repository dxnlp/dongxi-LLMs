# SFT runner completed update recovery contract

This specification is saved before new recovery measurements. It deepens the
actual Chapter 9 runner rather than substituting the separate microscopic SFT
lab. The public entry point remains CUDA and BF16 gated. Only independently
constructed random tiny HF models may execute on CPU during this source check;
no pretrained checkpoint, tokenizer download, GPU job, service or environment
installation is authorized.

## Numerical path and durable boundary

Extract a device-neutral loop from the existing runner. Preserve the fixed
`random.Random(seed)` permutation reused cyclically, microbatch accumulation,
right-padding attention mask, assistant body and end-token labels, exactly one
explicit next-token shift, FP32 cross-entropy sum divided by the total surviving
answer targets in the update, AdamW settings and clipping at 1.0. CPU tests use
FP32 weights without autocast; this is not proof of CUDA or BF16 equivalence.

A completed snapshot must bind policy weights, optimizer moments, parameter
layout, full applicable RNG, order/cursor, completed updates, cumulative shifted
answer targets, cumulative physical training positions, the latest completed
numeric metric and parent invocation to the stable scientific contract. SFT has
no separate KL reference; do not fabricate one. Saved metric counters must agree
with the saved data cursor and actual deterministic target/padded-length sums.
Resume goes into a new evidence directory, not the prior journal or exports.

Use the shared versioned trusted-local snapshot API after its interface is
confirmed. Validate independently expected bytes and size, schema and stable
source/lock/input/base/interface/recipe identity before data-only deserialization;
use `weights_only=True`. Validate cursor/order/counters/RNG and state layout before
applying any model or optimizer state. Keep immutable payloads with committed
headers, rather than overwriting an earlier recovery point. The independent
expected digest is evidence supplied by the caller, not automatically accepted
from the same unchecked resume payload.

Changing source, lock, data, base, token mapping, template, stop IDs, objective,
precision, optimizer, geometry, seed or declared update horizon rejects resume.
Operational output paths and invocation timestamps are not scientific identity.
Moving inputs can retain their verified bytes identity; changing an operational
deadline is a separately recorded new invocation, not permission to extend the
scientific horizon. Legacy unrestricted snapshots require a separate audited
migration; no implicit compatibility is claimed here.

Failures during backward or a partially executed optimizer operation are not a
completed boundary. Restart only from the preceding durable snapshot. A failure
writing a later snapshot, appending a metric, evaluating or exporting must not
erase the last committed training state. Retain the snapshot-carried completed
metric so a crash between commit and journal append does not hide that update.
Unsaved attempted updates and retried numbers remain distinct invocation
evidence; do not sum them as unique completed updates.

## Frozen CPU model and recipe

Use the approved isolated Python with CUDA hidden, offline HF flags, one Torch
thread and no installs. Construct Qwen3 from a local configuration only:
vocabulary 32, hidden width 16, intermediate width 32, one layer, two attention
and KV heads, head dimension 8, context 64 and attention dropout 0.1. The small
nonzero dropout makes Torch RNG restoration observable; the production model's
actual configuration is a separate source identity. Enable the runner's
gradient checkpointing and cache-disabled training behavior where supported.

Use original tiny message fixtures and an authored local tokenizer/template,
with three records of unequal encoded lengths and assistant target counts.
Include multi-turn answer ownership, a real end marker, and padding that aliases
EOS by ID while remaining label-masked. Freeze seeds 1212 and 1213; four optimizer
updates per arm; microbatch 1, accumulation 2, AdamW 0.003, no weight decay and
clip 1.0. Compare full parameters and rank-2 Q/V LoRA with dropout 0 and alpha 2.
The adapter rank is chosen for the tiny control, not retuned pretrained LoRA.
Resume at completed update 2 and compare updates 3 and 4 without selecting a
favorable checkpoint. Use a 16 MiB test snapshot cap.

For all four seed/mode arms, require exact same-environment CPU equality of
policy/Adam tensors, RNG, order/cursor, counters and numeric update metrics
between uninterrupted and restored continuation. Verify at least one fresh
process reload through the actual runner-local loop, not a model-weights-only
load. Compare the extracted uncapped loop with the original update equations.
Preserve timing as measured telemetry, not part of numerical equality.

## Failure controls and evidence

Independently test changed byte digest/size/schema/contract, source/lock/data/
base/interface/template/stops, cursor/update/target/position mismatch, invalid
RNG and incompatible model/optimizer layout before loading or updating. Keep
refused outputs and partial failure evidence. Test deliberate backward
interruption and recovery from the prior durable state; failed save/header or
metric observer; final evaluation/export failure; and refusal to overwrite an
existing evidence directory. Any simplified injected failure is labelled as a
control, not an unobserved production incident.

Retain raw outcomes, actual test/reload commands and exits, source/spec/test/lock
hashes, exact environment, fixed budgets, parent/snapshot identities, latest
committed metrics and all failures in new evidence. Do not rewrite existing
SFT identity or experiment reports. This package establishes bounded same-loop
CPU recovery readiness only. Pretrained serialization, full/merged interfaces,
current-source Spark recovery, practical save memory/disk/runtime, production
supervision and the staged comparison remain separately gated.

## Source-validation refinements before final collection

The numerical recipe above is unchanged. During preflight review, strengthen
the recovery contract with the full completed metric history, not only its last
row; bind optimizer parameter objects and ordered names; reject boolean aliases
for IDs and malformed Adam steps or negative second moments. Validate CUDA RNG
list/type/layout and bytes using temporary generators before applying model or
optimizer state. CUDA rejection fixtures use mocks only, with no GPU execution.
Ignore only the model configuration's operational `_name_or_path`, while
retaining actual architecture, dropout, precision and implementation settings.

Preserve the initial installed-tokenizer API failure and the subsequent
malformed-fixture serialization preflight failure. The API repair explicitly
requests a list from `apply_chat_template`; it does not alter labels or shift.
The malformed-key control must expect rejection at serialization rather than
pretend a forbidden mapping key can be saved. These source/preflight fixes are
not changes to seeds, model dimensions, objective, optimizer or the endpoint.

Final numerical collection keeps every seed/mode arm, its initial and committed
state identities, all four update metrics, both resumed metrics and a fresh
process continuation per arm. Sixteen unique recipe updates plus eight same
process and eight fresh-process replay updates are distinct from two independent
one-update original-equation parity controls. Do not count duplicate replay
work as new training progress. Retain immutable receipt-backed snapshots in a
new bounded temporary evidence directory; write new non-overwriting reports.

### Observed environment hardening after the first completed reference

The first 17-test verification and four-arm numerical reference stay unchanged
as historical records. Final review found that the production contract named
Torch/Transformers and lock bytes, but did not bind all observed Python/platform/
PEFT/tokenizer package identities. Add `numerical_environment` from the collected
environment identity: Python version, platform, machine and observed package
versions, excluding the operational interpreter and lock paths. The tiny fixture
uses the corresponding actually observed environment. Test changed versions and
platform before deserialization, and reject an invalid creation bound greater
than the shared API's `2**63-1` maximum before any allocation or acquisition.

This changes source/contract identities, not numerical seeds, update budgets,
fixtures, objective, parameters or checkpoint selection. Save a separately
named environment-hardening verification and all-arm reference; do not overwrite
earlier receipts or claim their old science contracts are current-source resume
inputs. Recheck unchanged update metrics and numerical state against the first
reference while retaining both derived identity generations.
