# Separate checkpoint IO schedule

Freeze this protocol before executing its new arithmetic controls. This task
adds a pure schedule module and tests, not a runner hook, checkpoint reader,
production approval or physical quota. Preserve the existing nineteen DPO work
dimensions and all validation-stage v2 records unchanged.

## Independent envelope and dimensions

The caller supplies exactly `max_payload_bytes`, `max_tree_nodes`,
`max_tensor_elements`, `max_tensor_bytes` and `max_primitive_bytes`. Values are
exact signed-64-bit nonnegative integers, except payload bytes and tree nodes
must be positive. Tensor and primitive byte bounds must not exceed payload
bytes. These supplied logical envelopes do not authenticate a tokenizer, model,
checkpoint or supplier. Unknown fields, bool aliases and oversized integers
refuse before arithmetic or canonical hashing.

Use a separate requirements schema `dongxi-snapshot-io-requirements-v1` and the
nine exact dimensions agreed with the shared-reader owner:
`snapshot_inspect_operations`, `snapshot_load_operations`,
`snapshot_save_operations`, `snapshot_hash_bytes`, `snapshot_tree_nodes`,
`snapshot_tensor_elements`, `snapshot_primitive_bytes`, `snapshot_clone_bytes`
and `snapshot_serialization_bytes`. Do not extend an old DPO cap file or silently
migrate an old requirements record.

For one inspect operation, reserve one inspection, the independently declared
exact positive payload byte count, the node envelope and primitive-byte envelope.
For one load, additionally reserve the tensor-element envelope. A save reserves
one save, maximum payload bytes for hashing and serialization, the node,
tensor-element and primitive-byte envelopes, and maximum tensor bytes for
cloning. These are the shared core's logical units. Payload hash bytes do not
claim to measure every canonical-contract hash, filesystem read, CPU instruction,
allocation, elapsed second or model FLOP. Generic metadata, header and repeated
contract visits must fit the declared whole-operation node/primitive envelope;
phase changes do not refill it. Admission and enforcement belong to the shared
core and runner, not to this pure calculator.

## Actual call schedule

Require explicit updates from one through 1000, checkpoint cadence from one
through 1000, diagnostic loads from one through sixteen and complete-attempt
capacity from one through four. Commit cursors are unique sorted zero, cadence
multiples through the final update, and the final update. A fresh attempt has
one save per cursor and the declared number of final full diagnostic loads.

A resumed attempt at durable cursor k has one pre-model inspection, one full
load, a new invocation save at k, saves at later commit cursors and the declared
final full diagnostic loads. DPO restore's repeated semantic check is not a
second shared load. Read costs in the schedule use maximum payload bytes;
individual calls may use their exact independently retained positive size.

Compute a componentwise maximum across resumed cursors. Total requirements are
the fresh vector plus that maximum multiplied by capacity minus one. This is
conservative capacity, not a measured trajectory, retry permission or enforced
invocation count. A rehashed omitted call must still fail recomputation.

## Frozen arithmetic fixtures and controls

The primary authored envelope is payload 16777216 bytes, 4096 visited nodes,
21646 tensor elements, 56248 tensor bytes and 262144 primitive bytes. Tensor
figures follow the already measured small DPO architecture only; the node and
primitive bounds are declared examples, not observations of the shared reader.
Keep the original two updates, accumulation four, generation cap64 and all
scientific ceilings unchanged. The primary cadence is two, one diagnostic load
and two complete attempts. Expected operation capacity is one inspection,
three loads and four saves. A separately declared cadence-one control expects
one inspection, three loads and six saves.

Test each operation independently, exact shorter read sizes, malformed envelopes,
all nine insufficient cap dimensions when positive, missing/extra dimensions,
old schema, changed/rehashed schedules, final deduplication, zero remaining
updates at a resumed final cursor, componentwise arithmetic, overflow and pure
import without Torch. Exhaustive small schedule enumeration is independent of
the production algorithm. Preserve any first failing command, then collect an
exclusive report with actual exits and unchanged source hashes using the existing
isolated CPU interpreter. No numerical fit, model acquisition, GPU, installation,
services, Git mutation, route edit or learner advancement is in this task.
