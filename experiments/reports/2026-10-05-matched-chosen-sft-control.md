# Matched chosen response SFT and DPO source control

The callable chosen-response SFT path now runs from the same local parent,
preference pairs and replacement draws as the existing DPO update. Eighteen
focused CPU controls pass. The fixed two-seed comparison preserves a negative
result: unchanged parent, chosen-SFT and DPO each answer **0 of 4** independent
location questions correctly. This verifies a missing scientific source control,
not successful alignment, a pretrained Spark experiment or production readiness.

## Protocol and actual verification

The [specification](../specs/2026-10-05-matched-chosen-sft-control.md) was saved
before numerical work. [Raw verification](2026-10-05-matched-chosen-sft/run-01/verification.json)
retains the command, actual exit, sampled memory,16 before/after source/input
bindings and75 archived artifacts. [All measured observations](2026-10-05-matched-chosen-sft/run-01/campaign/results.json)
retain both seeds and every arm, update, draw, validation diagnostic and raw
publication response. The first collection passed without changing a recipe
after inspecting outcomes; no earlier failed collection was relabeled.

| Check | Actual result |
|---|---|
| Focused controls |18 tests, unittest3.180s, child5.167254s, exit0|
| Fixed campaign |child2.382959s, exit0|
| Smallest sampled MemAvailable |126,090,657,792bytes, above25GiB|
| Sources and original inputs |all16 retained bindings unchanged|
| Actual model |6,592 random local Qwen3 parameters, CPU FP32|
| Frozen recipe |seeds1818/1819,6 updates per trained arm, accumulation2, lr0.008, beta0.2|
| Total campaign updates |24, with zero updates in both unchanged controls|

Memory is sampled at0.05s intervals; this is not continuous monitoring or a
physical containment guarantee. Each child has an external60-second deadline.
The isolated measured environment is Linux ARM64, Python3.12.14,
Torch2.14.1+cpu, Transformers5.18.0 and Tokenizers0.23.2. The recorded lock's
byte identity is not a claim that this already-existing environment was newly
installed from it.

Verification SHA256: `3bf1189bea1ec047d5044f49fb60eb9e3d350343ad899445f9e34d3bebb8f3da`.
Observation SHA256: `d2bc2f19496e871cc864cfa7a1c6947e994364a6db0d08b0b0dea26490e0f00e`.

## What is actually matched

The control uses the original Chapter11 location fixtures:8 training pairs,
4 validation pairs and4 independent evaluation prompts. Their bytes remain
unchanged. The [new sidecar](../../fixtures/matched-chosen-sft/protocol.json)
assigns explicit scenario source groups; these are authored group declarations,
not external corpus provenance. Raw/encoded prompts and groups are disjoint
across splits. A fixed WordLevel inventory covers the complete predeclared
literal fixture vocabulary, including publication vocabulary; no publication
answer is optimized or used to choose a coefficient or checkpoint. This is an
intentionally closed-vocabulary toy, not unseen-language transfer.

Every arm starts from the same actually saved/reloaded local parent bytes for
its seed. The original parent and frozen reference stay unchanged; final full
HF exports reload to the exact recorded policy tensor digest. Chosen-SFT and
DPO have the same private generator seed, actual sampled indices, chosen IDs,
completion masks, real EOS supervision, AdamW rate/decay, clipping and horizon.
The comparison also binds raw/encoded data, chosen-branch and tokenizer interface
digests: equal-length but different answers cannot pass as matched exposure.

The implementation does not copy the large production runners. It reuses their
actual encoding, collation, chosen-token summed cross-entropy, DPO update,
validation and greedy uncached generation. Chosen-SFT uses the sum of selected
chosen-token losses divided by the total valid chosen targets in the update.
DPO retains its mean-over-pairs objective and summed completion likelihoods.
Changing that denominator would create a different control.

## Unequal supervision and work

These completed **training** counts are identical across the two fixed seeds;
evaluation and export are separate operations and are not included in this table.

| Per trained arm and seed |Chosen-SFT|DPO|
|---|---:|---:|
| Optimizer updates |6|6|
| Actual replacement draws |12|12|
| Chosen targets including real EOS |48|48|
| Rejected targets |0|36|
| Logical input tokens |288|564|
| Policy forward calls |12|24|
| Submitted policy input positions |288|540|
| Reference forward calls |0|24|
| Submitted reference input positions |0|540|

The native SFT function forwards a complete chosen input before shifting its
logits; native DPO forwards each input without its final token. Preserve this
24-versus23-position difference per chosen example rather than rewriting a
runner to make the table look matched. Input positions are neither FLOPs nor
backward recomputation, physical memory, runtime or token billing. The helper
also retains an original parent plus policy/reference copies for comparison;
no large-model memory-fit or native two-model memory parity is asserted.

## Independent outcomes and failure meaning

| Seed and arm |Mean heldout reference-relative margin|Natural stops|Independent exact answers|
|---|---:|---:|---:|
|1818 unchanged|0|0/4|0/4|
|1818 chosen-SFT|-0.007323|4/4|0/4|
|1818 DPO|0.023136|4/4|0/4|
|1819 unchanged|0|0/4|0/4|
|1819 chosen-SFT|-0.007624|4/4|0/4|
|1819 DPO|0.052973|1/4|0/4|

All24 raw publication rows are retained, including stop-inclusive IDs, original
prompt IDs, exact expected answers, cap status and generation errors. Better
pair margins or natural stopping do not establish correct location extraction.
For example, seed1818 chosen-SFT answers `the front door <END>` to the first
publication question, whose independent target is `the purple pouch`.
Seed1819 chosen-SFT emits `the <END>`, while a DPO response repeats `attic` to
the cap. These remain negative evidence, not a reason to fit longer, change the
tokenizer, choose another seed or present a favorable sample.

The controlled failures check source-group/raw-prefix overlap, duplicate IDs,
unknown/overlength data, invalid settings, changed actual encoding, wrong real
termination/mask types, EOS-as-padding, and changed saved parent template.
Independent CE/gradient/Adam equations agree within1e-6; the DPO wrapper agrees
exactly with the unchanged native update. A deliberately failed second forward
retains two entered/one successful call and real earlier gradients, then poisons
the loop. Completed update events precede evaluation failures; failed-update
events retain known work rather than claiming unknown backend partial work.
The real local-only CLI exports/reloads a tiny policy and refuses reuse of its
existing evidence output without altering old exports. These are original CPU
fault controls, not a pretrained recovery test.

## Callable source and remaining stage gate

Use [the helper](../../src/dongxi_llms/chosen_sft_control.py) for injected-model
controls and [the local command](../../scripts/run_matched_chosen_sft.py) for
existing full/merged HF parents. The CLI restores and verifies the saved parent
template/tokenizer/stops before weight loading. Its explicit `--device cpu|cuda`
defaults to CPU; CUDA source uses FP32 weights, BF16 autocast, SDPA and policy
activation checkpointing without automatic fallback. No CUDA command was run.

A separately approved intended Spark comparison can use this prepared source
command, replacing every path/ceiling with its frozen observed/approved value:

```bash
PYTHONPATH=src python scripts/run_matched_chosen_sft.py \
  --checkpoint APPROVED_FULL_SFT_PARENT --tokenizer APPROVED_LOCAL_TOKENIZER \
  --tokenizer-id Qwen/Qwen3-0.6B-Base \
  --tokenizer-revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --train APPROVED_PREFERENCE_TRAIN --validation APPROVED_PREFERENCE_VALIDATION \
  --evaluation APPROVED_INDEPENDENT_EVALUATION --groups APPROVED_SOURCE_GROUP_SIDECAR \
  --output NEW_EVIDENCE_DIRECTORY --environment-lock APPROVED_ENVIRONMENT_LOCK \
  --arm chosen-sft --device cuda --seed 1818 --updates 100 --accumulation 4 \
  --lr 5e-7 --beta 0.1 --max-length 512 --max-new-tokens 64 \
  --max-parameters APPROVED_PARAMETER_CEILING --runtime-seconds 1800 --reserve-gib 25
```

The placeholders intentionally make this preparation incomplete; it is not a
launch authorization. The parent, full input contract, resource ceilings and
external supervisor must be resolved before a separately approved invocation.

The source adapter lacks accounted chosen-SFT resume and production ledgers,
snapshot/output routing and an external Spark supervisor. Its sampled guard is
not a hard whole-job deadline or quota. Interrupted chosen-SFT training requires
restart from the original retained parent; the native DPO recovery runner remains
the separately tested production path. Current pretrained profile/smoke/recovery,
branch pilots, independent human review and actual genealogy remain pending.
All45 model-scale stage rows and945 actual fields remain unexecuted/null.

Reproduce this CPU component into a **new** exclusive directory with:

```bash
/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  scripts/verify_chosen_sft_control.py NEW_EXCLUSIVE_CPU_REPORT_DIRECTORY
```

This does not rerun all notebooks, advance the learner, execute hosted CI or
verify Mac, pretrained quality, CUDA/BF16 numerics or physical resources.
