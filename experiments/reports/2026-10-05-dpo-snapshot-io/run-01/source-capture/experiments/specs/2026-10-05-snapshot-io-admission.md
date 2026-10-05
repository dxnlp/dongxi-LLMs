# Shared snapshot I/O admission — bounded CPU source protocol

Declared before implementation. This adds a separate cooperative I/O journal;
the existing SFT16/DPO19 scientific work limits remain unchanged. No model-scale,
GPU, acquisition, installation, service, Git mutation or external run is allowed.

Use strict `dongxi-snapshot-io-work-v1` with nine explicit dimensions:
`snapshot_inspect_operations`, `snapshot_load_operations`,
`snapshot_save_operations`, `snapshot_hash_bytes`, `snapshot_tree_nodes`,
`snapshot_tensor_elements`, `snapshot_primitive_bytes`, `snapshot_clone_bytes`,
`snapshot_serialization_bytes`. A supplied whole-operation envelope independently
bounds payload bytes, visited nodes, tensor elements/bytes and primitive bytes.
No file-size-to-logical-tensor inference or automatic missing-cap migration.

Reserve each full operation before contract walking, payload hashing, restricted
loading, finite/tree checks or cloning/serialization. Count actual entered and
completed logical units separately; permanently retain failed reservations.
`hash_bytes` means payload bytes fed to SHA256, not every metadata/journal hash.
Tree/tensor/primitive units are declared visits, not CPU instructions/FLOPs.
The whole operation shares counters across its contract/state phases.

A bounded, no-follow, independently retained receipt binds exact payload
SHA/size/header, scientific and I/O contracts, the checkpoint's pre-save I/O
prefix and its runner-work prefix. Bind both physical journals before inspection.
The saved payload carries the same I/O prefix, not its own future save completion.
Recovery replays later failures/spending and refuses copied/replaced journals.
This is trusted local evidence, not adversarial authentication or cross-machine
recovery. Preserve unhooked CPU reference v1 APIs; hooked payloads use explicit
v2 and must not silently accept v1.

Controls: refusal before payload read, `torch.load`, finite scan or clone;
repeated inspect/load charges; cumulative nodes and expanded-view elements;
malformed/changed receipts, copied journals, later failed charges; save failures
and partial files; artifact receipt coexistence; fresh original-equation SFT
full/LoRA and DPO checkpoint-on interrupted replay. Mapper schedules are separate
pure fixture requirements, never launch authority. Actual CLI tests must catch
pre-admission identity hashing and model allocation.

Explicit exclusions: bounded receipt/cap/journal processing, metadata hashing,
caller state capture/history copy, runner semantic panels (already separately
charged), application, artifact inventory hashes, other logs/exports/scratch,
deserializer internals and physical CPU/memory/deadline/storage containment.
The restricted loader remains trusted-local, not a hostile-checkpoint sandbox.
Keep Day9, thirteen completed packages and all45 external outcomes unchanged.
