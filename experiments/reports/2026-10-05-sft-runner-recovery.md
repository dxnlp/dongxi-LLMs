# Actual SFT runner recovery: bounded CPU reference

The extracted **actual Chapter 9 SFT update path** reproduced its uninterrupted
continuation exactly after a completed-update snapshot in all four fixed
seed/mode arms. All four independently launched fresh-process continuations
also matched. The final focused suite passed 18 tests; the eight existing SFT
contract tests passed unchanged. This establishes small, same-environment CPU
recovery readiness, not pretrained Spark, CUDA or BF16 recovery acceptance.

## Evidence and frozen recipe

The [specification](../specs/2026-10-05-sft-runner-recovery.md) was saved before
the first measurements. Its final section records validation refinements after
preflight, without changing the numerical recipe. The
[current raw reference](2026-10-05-sft-runner-recovery-environment-hardening-reference.json) contains every
arm, all four uninterrupted update metrics, both resumed metrics, initial and
final state identities, committed receipts, full loop contracts, actual child
commands/exits and before/after source hashes. The
[current focused verification](2026-10-05-sft-runner-recovery-environment-hardening-verification.json) retains
the actual test command, stdout/stderr, exit and its source identities.
The earlier [17-test verification](2026-10-05-sft-runner-recovery-verification.json)
and [first four-arm reference](2026-10-05-sft-runner-recovery-reference.json)
remain unchanged historical evidence from before the observed-environment
contract hardening described below.

No weights or tokenizers were downloaded. Qwen3 was constructed randomly from a
local configuration: vocabulary 32, hidden width 16, intermediate width 32, one
layer, two attention heads, two KV heads, head dimension 8 and context 64. Training
uses CPU FP32, SDPA, gradient checkpointing and cache-disabled forwards. Attention
dropout 0.1 makes the Torch RNG relevant. The full model has 3,136 trainable
parameters. Rank-2 Q/V LoRA, alpha 2 and dropout 0, has 128 trainable parameters
out of 3,264 total parameters, including the frozen base.

Seeds are 1212 and 1213. Each arm runs four updates, with a recovery boundary
after update two. Microbatch is one, accumulation is two, AdamW learning rate is
0.003, weight decay is zero and gradient clipping is 1.0. Each snapshot has an
explicit 16 MiB maximum. These are tiny recovery controls, not tuned pretrained
model choices or evidence of practical LoRA quality.

The three authored message records have encoded lengths 13, 11 and 19 and
shifted assistant target counts 2, 3 and 5. The third is multi-turn. The local
WordLevel tokenizer and authored template use the same end-marker ID for genuine
answer ends and padding. Genuine end labels remain supervised; physical padding
has attention zero and label `-100`. Loss sums over assistant-owned body/end
tokens, shifts once and divides by the total surviving answer targets across
the complete accumulated update. The prompt still participates in the forward
computation despite not supplying direct loss targets.

The approved isolated environment was Python 3.12.14, Torch 2.14.1+cpu,
Transformers 5.18.0, Tokenizers 0.23.2 and PEFT 0.20.0 on Linux ARM64. Torch uses
one thread. CUDA is hidden; both offline HF flags are set. Actual package
versions and this CPU execution are distinct from a claim of recreating the
Spark CUDA environment merely because the same `uv.lock` bytes are hashed.

## All numerical arms

The table gives update answer NLL, not a held-out quality evaluation. Both
short LoRA trajectories remain nearly flat or fluctuate; they are retained
without selecting a favorable endpoint. Recovery success does not require
monotonically decreasing loss or claim that these four updates learned a useful
assistant.

| Seed / mode | Update 1 NLL | Update 2 NLL | Update 3 NLL | Update 4 NLL | Cumulative answer targets / physical positions | Boundary payload bytes | Exact resumed and fresh-process continuation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1212 / full | 3.505505 | 3.459347 | 3.392564 | 3.281359 | 27 / 118 | 70,465 | Pass |
| 1212 / LoRA | 3.507949 | 3.477081 | 3.488470 | 3.505558 | 27 / 118 | 41,734 | Pass |
| 1213 / full | 3.454464 | 3.429351 | 3.373835 | 3.266179 | 25 / 110 | 70,529 | Pass |
| 1213 / LoRA | 3.452240 | 3.468398 | 3.460676 | 3.452612 | 25 / 110 | 41,734 | Pass |

Each seed uses its fixed permutation cyclically, so four updates do not cover
each of the three unequal records equally. This explains the different target
and physical-position totals. Physical positions are the actual forwarded
microbatch sizes, including prompt positions and any padding, not the number of
loss targets. These four microbatch-one arms have no within-batch padding; the
separate ragged-batch test exercises padded geometry and verifies the derived
work-cycle counters against actual collations.

Every comparison checks the full tensor/primitive state: policy, Adam moments,
Python/Torch RNG, order/cursor, cumulative counters, model mode, full completed
metric history and loop layout/identity. The independent original-equation
control also matches the extracted update's metric and complete resulting state
exactly with identical dropout RNG. Fresh processes restore the snapshot and run
the same actual loop, rather than comparing only reloaded weights.

The numerical reference executes 16 unique fixed-recipe updates, eight
same-process replay updates, eight fresh-process replay updates and two separate
one-update original-equation parity controls. Duplicate recovery work is not new
training progress. The current reference completed in 9.972354 seconds; this is small CPU
test telemetry, not a Spark training or cache speed claim.

## Recovery contract and lifecycle

The public runner remains CUDA/BF16 gated. Its device-neutral `SFTLoop` is the
same path called by that public entry point, allowing bounded CPU construction
without mocking pretrained acquisition or claiming a production launch.

Scientific identity binds actual source/lock/data/base bytes, tokenizer mapping,
template/stop IDs, actual encoded records, objective, precision/backend, optimizer
recipe, geometry, seed and fixed update endpoint. It also binds the observed
Python version, platform, machine and package versions collected from the actual
environment, not just their possible declarations in a lock file. Operational output/input paths,
deadline, invocation timestamps and checkpoint cadence are excluded. The actual
model configuration excludes only `_name_or_path`; architecture and dropout
changes remain incompatible. Optimizer groups bind ordered actual trainable
parameter names and objects, not just matching parameter counts.

Resume requires independently retained contract JSON and independently expected
payload SHA256 and byte size. Pre-model inspection compares current observable
science and actual payload bytes before any deserialization. Sources, inputs and
lock are rehashed during pre-load preparation. The restricted shared loader then
uses the exact verified buffer, with `weights_only=True`, and invokes runner
validation before model or optimizer application. SFT does not have a separate
KL reference; none is invented in its snapshot contract.

The validator checks completed phase, fixed endpoint, exact typed counters and
cursor/order, tensor layout, Adam state coverage, scalar non-boolean step,
nonnegative second moments and applicable RNG. Full history must have exactly
one correctly sequenced row per committed update and agree with target/position
counters and the last metric. Boolean aliases for IDs are refused. CUDA RNG
validation checks list/type/layout and uses temporary generators to validate
bytes before any live state application. The CUDA tests are mocked metadata
controls only, not executed CUDA replay evidence.

Initial and completed-boundary snapshots are immutable and receipt-backed.
Periodic and terminal snapshots are committed before the metric observer, and
the terminal snapshot precedes final evaluation/export. A resumed invocation
writes its restored history separately rather than overwriting the parent log.
Baseline evaluation preserves training RNG. The fixed endpoint cannot be
extended by changing invocation paths or deadlines.

An interrupted backward or partially applied optimizer update is poisoned; it
cannot be saved or advanced until restored from a durable completed snapshot.
Unsaved attempted updates are distinct from committed updates. The failed-save
control leaves the earlier snapshot unchanged, and exclusive creation refuses
overwriting it. Injected baseline, metric-append, final-evaluation and export
failures leave recoverable boundaries at updates 0, 1, 4 and 4 respectively.
The interval-two control recovers both committed rows even when append fails
after committing update two. These are controlled failures, not unobserved
production incidents.

## Retained failures and final verification

The initial [tokenizer preflight attempt](2026-10-05-sft-runner-recovery-initial.json)
exited 1: 12 test methods produced 16 errors before any optimizer update.
Installed Transformers 5.18 returns `BatchEncoding` by default from
`apply_chat_template`; the existing renderer assumed a list. The repair requests
`return_dict=False` explicitly in both encoding and generation. It keeps the
strict prefix check, answer labels and one-shift objective unchanged. A later
12-test rerun passed before the validation suite was expanded.

The [hardening preflight](2026-10-05-sft-runner-recovery-hardening-preflight.json)
also exited 1: one malformed fixture tried to save boolean optimizer mapping
keys. The shared data-only serializer correctly rejected that fixture before a
malformed payload could be created. The control now asserts that serialization
rejection and independently checks the runner's strict key validation; no shared
serializer check was weakened. This setup error is distinct from the initial
tokenizer interoperability failure.

The first source-frozen panel exited 0 with 17 tests in 4.405 seconds and its
four-arm reference completed in 9.891684 seconds. Final contract review then
found that Torch/Transformers and lock bytes did not cover observed Python,
platform, PEFT and tokenizer dependencies. The approved source hardening binds
those observed identities and rejects changed Python/platform/machine or
Torch/Transformers/Tokenizers/PEFT versions before deserialization. The source
also rejects an explicit creation cap outside `1..2**63-1` before allocation.

The first environment-hardening rerun had a
[test-import setup error](2026-10-05-sft-runner-recovery-environment-preflight.json):
one indentation error in the child helper prevented importing any fixture. Its
actual exit and terminal traceback are retained with the source hashes at
failure; the transcript explicitly distinguishes manual terminal capture from
an executed self-collector. The indentation was corrected without changing the
recipe.

After the final source freeze, the
[environment-hardening focused attempt](2026-10-05-sft-runner-recovery-environment-hardening-verification.json)
exited 0 with 18 tests in 4.860 seconds. Its collector took 5.756020 seconds,
including child-process overhead. The eight original SFT contract tests passed
again, unchanged, in 0.530 seconds. The suite covers exact replay, original update
parity, mask/shift/end-token ownership, scientific and actual-byte refusals,
typed counters/history/RNG/layout/Adam checks, interruption poisoning, failure
ordering, output protection, fixed horizon, optimizer name binding, ragged work,
mocked CUDA rejection, operational model-config relocation and observed
environment refusals.

All seven current source/spec/test/lock hashes were unchanged before and after
both final environment-hardening collectors. The six hashes in the first
source-frozen collectors also remain unchanged within those historical runs.
All four uninterrupted metric sequences and initial/final numerical state
digests are exactly equal between the first and final references. Source science
contracts and snapshot byte receipts intentionally changed; no historical
identity was silently rewritten or declared a current-source resume input.
Temporary current committed fixture artifacts are retained at
`/tmp/dongxi-sft-recovery-reference.4t2p3mr5`; the first collection's artifacts
remain at `/tmp/dongxi-sft-recovery-reference.n1fi6x_d`. The raw references contain their
individual byte receipts. Snapshot byte hashes and framed numerical-state hashes
serve different purposes; repeated serialization is not assumed byte-identical.

| Frozen input | SHA256 |
| --- | --- |
| Runner | `91a53ba4a7a85c2bd9adebfdca4671c909a151cc3899559ba65b8afc7010d8b8` |
| Shared snapshot module | `9cad2851b665b5fe5d640c1327710352396218ea80fab3b0c82f01925e1749a3` |
| Run identity module | `a98d244efbfcbecf13ba97e28f512b2743642ac7d6e2ac8fad3a36eb5bcb4999` |
| Focused test/collector | `1ec61808ae741726c13954501260e132c9741a62703c5636013426fe376ad570` |
| Existing SFT contract tests | `a69fcff3d7fe0d2eb7e5dee1b84d4b34f8f70cb507dc4908f7a179ce61d401c9` |
| Final specification | `2ec0c47b9889f4807750bc8461b0a21584281a41db5ff83ccd0fe6d2baa0613c` |
| Lock file | `1371eb74c14be4ccf69b2f09ce078cbd6d09439792965064140a911906a448b3` |

| Evidence | SHA256 |
| --- | --- |
| Initial tokenizer failure JSON | `4413b4fd795b98faae952924e90acdf2ed4e85e902c5e8f03a02d5c02ce6e5af` |
| Hardening setup failure JSON | `8a12c8158f1dba13da03a9e6cf1bf10f3e93103f90c4cfdc3557235c2927a938` |
| First 17-test verification JSON | `81a2e36041fd4ffe22cf0b005f49e9ec9a54c688d586e0c84859c0acc48bcf78` |
| First all-arm numerical reference JSON | `d23fa63d3a38c80e7cf02f285805240e15c1f721d181301d2e02ecf9996ba000` |
| Environment import preflight JSON | `6adf63300b0ca1d690be973a86429102da84472476194a48bb8230f799cc633d` |
| Current 18-test verification JSON | `971f2df286095d16934ca0bda0fc4393de1118ab37fd0f2992c2822dd5c77620` |
| Current all-arm numerical reference JSON | `4692184d78da61765efd29f3bc1e79f65742e06c96e5170422e9c449d5f79c7d` |

## Reproduction and remaining gates

From the repository root, run these commands with new report names. The
collectors refuse to overwrite earlier evidence:

```bash
env PYTHONPATH=src CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python tests/test_sft_runner_recovery.py --verify-tests experiments/reports/NEW-sft-recovery-verification.json

env PYTHONPATH=src CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python tests/test_sft_runner_recovery.py --reference experiments/reports/NEW-sft-recovery-reference.json

env PYTHONPATH=src CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python -m unittest discover -s tests -p test_spark_sft_contract.py -v
```

The independent expected receipt and retained contract are trusted caller inputs;
restricted deserialization is not a malicious-file sandbox or proof of unlimited
allocation safety. No legacy unrestricted snapshot migration is implied.
Physical snapshot caps constrain this tiny fixture, not peak production RAM or
disk feasibility. No continuous memory or external supervisor guarantee was
measured in this package.

Pretrained full/adapter/merged serialization, current-source Spark recovery,
CUDA/BF16/Flash numerical behavior, production save/load envelope, supported
worker/supervisor behavior and all staged model-scale comparisons remain
separately gated. No GPU job, production inference, service, package installation,
model download, historical report rewrite or learner-progress change was made.
