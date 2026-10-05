# SFT recovery validation and persistent work limits

The actual SFT runner now reserves a complete declared semantic-validation panel
before inspecting saved history, tensor values or RNG state. The current CPU
reference passed 80 controls and exact completed-update replay in four fresh
processes, covering full training and rank 2 LoRA for seeds 1212 and 1213.
Failed validation remains spent when an older numerical checkpoint is restored.
This closes the runner-semantic accounting gate, not the complete production
recovery or physical containment gate.

The [premeasurement protocol](../specs/2026-10-05-sft-semantic-validation-work.md)
fixes the recipe, units, caps, failure injections and excluded work. The current
[verification record](2026-10-05-sft-semantic-validation-work/run-02/verification.json)
has SHA256 `69c49364ba3746e686a3cb988b48648acad4faddc579a76f130cdaac33a5776f`.

## Scientific recipe and version boundary

The reference uses the original locally constructed random one-layer Qwen3
model, vocabulary 32, width 16, intermediate width 32, two query and KV heads,
head dimension eight, context 64, tied embeddings and SDPA attention dropout
0.1. Training remains CPU FP32 with actual activation checkpointing, four
updates, microbatch one, accumulation two, learning rate 0.003, weight decay
zero and gradient clipping at one. The LoRA arm retains rank two on query and
value projections, zero LoRA dropout and the frozen original base weights.
The three ragged records, assistant and end-token masks, single causal shift,
EOS/padding alias and deterministic cyclic selector remain unchanged.

The strict `dongxi-sft-logical-work-v2` contract requires sixteen dimensions:
the original twelve training/evaluation/generation caps plus
`recovery_validation_operations`, `recovery_history_rows`,
`recovery_tensor_elements` and `recovery_rng_states`. Old budgeted v1 contracts
refuse; no migration or inferred allowance is implemented. Original unbudgeted
CPU reference APIs remain available. The new fixture preserves every original
twelve-dimensional cap and separately declares 64 validation operations, 256
history rows, 10000000 tensor elements and 128 RNG states. These are tiny CPU
fixture capacities, not approval or sizing evidence for a pretrained model.

The observed environment is Linux ARM64, Python 3.12.14, Torch 2.14.1+cpu,
Transformers 5.18.0, Tokenizers 0.23.2 and PEFT 0.20.0, with CUDA hidden,
offline model access and one numerical thread. Each retained arm contract
contains the observed package versions, lock, encoded records, tokenizer
interface, initial random policy and source identities.

## Admission and measured accounting

Before reservation, the runner requires the completed cursor, fixed container
lengths, strict v2 contract and the same physical journal prefix. It derives the
panel from layout metadata captured at loop construction, not untrusted saved
tensor sizes. One invocation reserves one operation, all completed history
rows, model-state tensor elements, populated Adam scalar steps and two moments
per optimized parameter, CPU/CUDA RNG tensor elements, and one Python, one CPU
Torch and each CUDA RNG state. Separately named tied state keys count separately
because they are separately checked. SFT adds no invented sampler draws.

The validator records entered and successfully checked components separately.
An exception leaves the complete conservative reservation in the journal, with
partial and uncertain work distinguished. It cannot apply policy, Adam or live
RNG state. Same-length bad history, invalid tensor/RNG values, oversized nested
metadata, sparse tensor layouts and oversized dictionary keys have retained
negative controls. Zero allowance refuses before semantic scans and application.

Actual save validates once, then refreshes the snapshot-carried work prefix to
include that charge. Restore charges its shared-loader callback once before
state application. The resumed lifecycle immediately saves the restored cursor
again and charges that distinct invocation; it is not silently deduplicated.

## Fresh process results

The collector's focused panel exited zero: 80 tests in 32.199 seconds reported
by unittest, 33.150 seconds observed by the parent subprocess. Sixteen controls
are in the new module; the other 64 preserve earlier SFT recovery, work,
contract and shared-ledger checks. Independent original-equation tests preserve
full and LoRA numerical trajectories.

Each arm durably saves update two, retains an authored later history-validation
failure, then launches a new CPU process using the same physical journal and a
new output directory. Every child exits zero and restores two before finishing
four. Policy including frozen LoRA base weights, Adam, order/cursor,
Python/Torch RNG and committed history match uninterrupted execution exactly,
excluding only resource receipts from the numerical comparison.

| Seed | Mode | Exact fresh replay | Reserved validation operations | Successful operations | Reserved tensor elements | Successful tensor elements |
| --- | --- | --- | --- | --- | --- | --- |
| 1212 | full | yes | 6 | 5 | 83649 | 68660 |
| 1212 | LoRA | yes | 6 | 5 | 54292 | 45200 |
| 1213 | full | yes | 6 | 5 | 83649 | 68660 |
| 1213 | LoRA | yes | 6 | 5 | 54292 | 45200 |

All four retain 12 reserved versus 10 successful history rows and RNG states,
and the failed ticket remains present. A separately predeclared seed 1212
full-mode campaign fixes its operation cap at six. Its fresh replay still
matches the original numerical endpoint; the next validation refuses with zero
semantic-scan calls and zero state-application calls, without changing spending.

The observer retains cached generation (`use_cache=True`) and conservative
uncached-prefix reservation, not a claim about exact cached FLOPs. Negative
generation remains visible: for seed 1212 full mode, development NLL changes
from 3.497850 to 3.355714, while both final four-token outputs remain capped,
fail exact match and repeat `assistant`. This is accounting and replay evidence,
not improved language capability.

## Evidence identity and earlier attempts

Current executable SHA256 is
`c0a5287fceeb2e20d80f0bc3125b0022a552659654c0016d3a1a6b91fae377f5`;
new test SHA256 is
`53b09254dbeac765e692bf38b62abbc3ca7dfffec46b7705a8c9b0913c17cdd8`;
protocol SHA256 is
`55fb38317a5fd6d659a9d503e2be7b24358ae6bcc8c19c5ccbb881986e3fa74a`.
Run-02 records twelve identical before/after source bindings and 121 artifact
hashes. A separate post-collection read-only check found zero mismatches across
all twelve current sources and all 121 retained artifacts. Raw journals,
snapshots, failures, all generated IDs, subprocess commands/exits and numerical
digests are retained beside the verification record.

The [first exclusive collection](2026-10-05-sft-semantic-validation-work/run-01/verification.json)
passed 79 controls and four fresh replays but is historical: reciprocal review
subsequently added length guards before optimizer/RNG/moment key traversal.
Run-02 adds the 5000-key no-traversal regression without changing the recipe.
Earlier developmental failures remain in the
[initial diagnostic](2026-10-05-sft-semantic-validation-initial-diagnostic.json),
[legacy compatibility diagnostic](2026-10-05-sft-semantic-validation-compatibility-diagnostic.json)
and [test-selector diagnostic](2026-10-05-sft-semantic-validation-test-selector-diagnostic.json).
Root-owned compatibility changes are separately declared in the
[fixture protocol](../specs/2026-10-05-validation-budget-fixture-compatibility.md);
they preserve the original twelve caps and numerical assertions. No earlier
report or raw evidence was overwritten.

## Remaining production gates

Shared restricted deserialization and generic tree/finite scans run before the
restore callback and are not charged by these new runner-semantic units.
Generic save checks, byte hashing, cloning, state application, serialization,
metadata/layout observation, journal-integrity work and general CPU time are
also excluded. A whole-restore envelope or shared hook therefore remains
pending, as do SFT snapshot artifact routing, broad log/export/scratch routing,
story-run work limits, live production cap mapping and physical containment.
There is no measured pretrained, CUDA/BF16 or model-scale overhead evidence.
No model/data acquisition, GPU/service, install, Git mutation, publication or
learner advancement occurred in this package.
