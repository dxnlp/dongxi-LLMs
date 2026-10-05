# Original trusted local cache and policy states

These data-only Torch states and JSON headers are outputs of the predeclared
tiny CPU experiment, not pretrained checkpoints or arbitrary external files.
The initial raw report records independently expected file digests, contracts,
actual pending collections and all original primary trajectories. Use the
module's safe bounded loader with those expected identities; never generalize
the trusted-local boundary to downloaded pickle.

The generation snapshot follows two global draws. Each DPO/RLVR completed
snapshot follows three updates; the paired pending snapshot also retains the
fourth batch after collection and before its optimizer step. Model/reference,
Adam state, data cursor, rollout/Torch RNGs and any pending likelihoods belong
to the state. These examples do not add resume support to the optional Spark
runner implementations. Their source/numeric identity is preserved in
`experiments/reports/2026-10-04-batched-cache-recovery.json`; later verification
records actual current-source replay separately.
