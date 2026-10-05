# V2 trusted local recovery states

These thirteen data-only Torch states and JSON headers were collected after
repairing the original structural identity encoding. The fixed numerical
recipe was unchanged: one interrupted generation state, eight three-update
completed/fourth-batch-pending states, and four final six-update states.

The external byte identities, current source hashes, actual commands and
direct migration comparisons are in
`experiments/reports/2026-10-04-batched-cache-recovery-hardening.json`.
The current bounded loader requires the independently expected v2 contract
and external file digest before `weights_only=True` deserialization. It does
not accept old v1 contracts or arbitrary downloaded checkpoint files.

The two older fixture directories, their tensor/header bytes and their raw
reports remain historical evidence. Their earlier loader instructions describe
the old implementation; they are not an instruction to bypass current v2
validation. Migration compares actual tensors, pending actions and numerical
histories without replacing historical identities. New identities are expected
to differ because typed container boundaries and byte-length framing changed.

This tiny float64 CPU exercise does not add pretrained Spark runner recovery,
authenticate arbitrary external metadata or establish cross-device bitwise
replay, serving performance or held-out model quality.
