# Completed update recovery in the DPO runner

This protocol precedes implementation measurements on 2026-10-05. It tests the
actual update functions used by `scripts/run_chapter11_spark_dpo.py`, rather than
inferring runner readiness from a separate recovery lab. Only an original tiny
local CPU model, authored token sequences and the existing isolated interpreter
are authorized here. No pretrained checkpoint, GPU, acquisition, installation,
service, publication or Git operation is included.

## Recovery boundary and preserved objective

A recoverable boundary follows a complete AdamW update. It binds the updated
policy, the original frozen reference, optimizer moments, data sampling generator,
global Torch and applicable CUDA RNG, completed update number, cumulative work
and the committed numerical metric history. A failed backward or interrupted
optimizer operation is not a new boundary. Restart from the last durable state
in a new output directory; do not accept a partly mutated live model as saved.
There is no pending rollout or mid-accumulation guarantee in this DPO package.

Preserve sampling with replacement, one pair per accumulation microbatch, mean
pair loss across microbatches, summed chosen and rejected response likelihoods,
the single causal target shift, valid termination tokens, prompt and padding
exclusion, the original reference, AdamW weight decay0.01 and gradient clipping1.
CUDA keeps FP32 weights and BF16 autocast; the CPU verification uses FP32 and no
autocast. No reward, chosen NLL, rehearsal, reference refresh or objective repair
is introduced.

Scientific identity and invocation evidence are separate. The stable contract
contains actual parent, source, environment lock, data and encoded branch bytes;
the observed tokenizer, template and stop interface; model and optimizer setup;
seed, accumulation, beta and the planned update endpoint. The current extension
does not authorize changing that endpoint or silently migrating old snapshots.
Output paths and a fresh invocation deadline do not change the scientific recipe.
Expected snapshot digest, size and contract must be independently supplied and
checked through the shared trusted local data-only loader before deserialization.
Supply `--resume`, `--resume-sha256`, `--resume-bytes` and the separately retained
`--resume-contract` together. Observable data/source/parent/lock/interface/recipe
fields are compared before model allocation; the original-parent reference,
effective architecture, functional dropout, attention backend and ordered
optimizer layout are recomputed before any saved tensor is applied. File paths
may move when their role-bound actual bytes are identical. The CPU snapshot
envelope is16MiB. Production must explicitly supply its bounded envelope.
The shared API is being coordinated before runner edits; its final source identity
will be frozen with the new evidence, not substituted into historical reports.

The sampler draw count must equal completed updates times accumulation. Cumulative
pair, completion token, input token and forward-position counters must agree with
the retained selected indices and encoded masks. Restore the original reference,
not a copy of the updated policy. Reject inconsistent counters, changed data,
source, parent, interface, optimizer or reference state before applying a resume.

## Durable publication and failure retention

Save completed update0 before baseline evaluation. Commit periodic completed
snapshots before publishing their associated durable metric row. Include complete
committed numerical history so a metric-write interruption can be reconciled on
the next invocation. Rows after the last committed boundary remain diagnostic
attempts; a new invocation identifies any retried update numbers rather than
counting both as one uninterrupted trajectory. Snapshot cadence is declared.

Commit the final completed training state before final evaluation and HF export.
An evaluation, export, journal or metric failure must leave that state usable.
Keep existing output directories and immutable committed snapshots untouched.
Partial files are failed attempts, never completion markers. Evaluation runs in
an RNG-preserving context and cannot change the next training draw. Retain every
raw generated continuation, IDs, natural stop versus cap and error; known token
counts are not a claim that internal generation forward work has been measured.

This integration also distinguishes raw prompt separation from observed encoded
prompt separation. Explicit source groups must be disjoint where provided;
missing legacy group provenance must be rejected or explicitly labeled as an
unverified adoption, never inferred from a unique row ID. Gold evaluation text
does not select training rows, hyperparameters or checkpoints.

## Frozen CPU comparison

Use seed1818, six completed updates and two sampled one-pair microbatches per
update. Use AdamW learning rate0.008, weight decay0.01, beta0.2 and gradient
clipping1. The local random Qwen3 microscope has vocabulary16, width16,
intermediate width32, one layer, two query heads, one KV head, head dimension8,
context32 and no dropout. Its weights are randomly initialized locally; no Hub
weights or tokenizer are loaded. Authored branches include unequal completion
lengths, a retained EOS, EOS also used for padding and prompt-excluded masks.
Store the exact authored records and fixture configuration in the test/evidence
sources before fitting. These sequences verify recovery, not language quality.

Compare uninterrupted training with completed update3 serialization and remaining
updates, plus a separately launched fresh-process replay using the same actual
runner functions. Require exact same-environment CPU equality of policy and
reference tensors, optimizer states, sample indices, sampler/global RNG, update
and work counters, and numerical metrics. Record elapsed time separately; it is
not an equality target. Check the next sampled batch and the very next loss,
not only a favorable final parameter value. Retain the update0 and final boundaries.

Independent tests cover masks and the unchanged loss; file/schema/contract and
reference tampering; inconsistent draw and work counters; existing-output refusal;
source-group and actual encoded-prompt collisions; and failed save, metric,
evaluation and export paths. A failed metric after a successful commit must
recover the committed numerical row. A failed next save leaves the previous
boundary valid. Preserve any failed test or collection as a distinct diagnostic.

## Evidence and remaining gates

Record current source/test/spec and shared snapshot identities, exact sanitized
commands, actual interpreter/package information, measured exit and runtime,
fixture/checkpoint hashes, uninterrupted and resumed numerical records and all
failure controls in new dated reports. Original reports are not rewritten.

Passing establishes an actual runner function and trusted local CPU recovery
contract only. It does not establish pretrained capability, CUDA/BF16 numerical
equivalence, save overhead at model scale, cross-machine determinism, hard whole
job resource containment, independent behavioral quality or a Spark pilot pass.
External deadline, total disk/token/work ceilings, platform containment and a
separately approved current-source profile and recovery stage remain required.
