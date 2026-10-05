# Exact-base LoRA merge and local reload

After the fixed rank-8 Q/V native replay passes, verify its original DXI-01
handoff criterion using the20-update adapter, not a fabricated400-update parent.
Bind the recorded Base revision/localweightbytes, actual adapter config/tensors,
native template/tokenizer mapping/stops and raw development prompt bytes. Reject
an adapter with different original base, revision, rank, targets or vocabulary.
No model download or third-party adapter substitution is allowed.

At the first eight original development generation prefixes, compare FP32-view
BF16/SDPA logits from Base+adapter with the merged full model. Predeclare
absolute tolerance0.125 and relative tolerance0.015625 (two BF16 epsilons),
and report actual maximum/RMS discrepancies and argmax agreement. This is a
declared numerical comparison tolerance, not a theoretical error guarantee or
bitwise-equivalence claim: merging rounds updates into BF16 base weights.
Failure is retained rather than increasing tolerance after measurement.

Save the merged full HF model and original tokenizer to a new private directory,
with actual base/adapter genealogy. Delete only in-memory model references,
reload the actual saved bytes in the same child, and require exact logits at all
eight prefixes. Record output inventory hashes and reconstructed interface.
Never present an adapter directory alone as a full-policy handoff.

One owned local-only child has900-second external deadline,25GiB reserve and
4GiB output planning space. This is merge/reload identity proof, not training,
new pilot, improved behavior, cross-machine parity or a selected SFT parent.
Its hash/export/diagnostic forward cost is outside earlier SFT16/I/O9 journals;
record it separately. It remains goal-aligned bounded acceptance work.
