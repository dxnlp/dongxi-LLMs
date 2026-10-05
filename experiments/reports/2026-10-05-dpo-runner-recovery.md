# Actual DPO runner: completed-update recovery

The bounded local CPU recovery gate passes. Twenty-one focused controls and an
exclusive evidence collection exercise the same update, commit, restore and
finalization functions used by `scripts/run_chapter11_spark_dpo.py`. No pretrained
checkpoint, CUDA launch, download, installation, service or Git operation was
performed. This is a source/CPU mechanism result, not a Spark pilot or language
quality result.

The [premeasurement protocol](../specs/2026-10-05-dpo-runner-recovery.md) freezes
seed 1818, six updates, accumulation 2, beta 0.2, AdamW learning rate 0.008,
weight decay 0.01 and clipping 1. The original random Qwen3 has vocabulary 16,
width 16, one layer, two query heads and one KV head. Its authored branch records
retain unequal response lengths, real EOS targets, prompt exclusion and
EOS-valued padding. Those records and their exact configuration are retained in
[the fixture](2026-10-05-dpo-runner-recovery/run-02/authored-fixture.json).

## What now recovers

A completed snapshot retains the policy, **original** frozen reference, AdamW
moments and steps, sampling generator, global Torch/applicable CUDA RNG, completed
cursor, cumulative token/forward work and authoritative numerical history. It
does not retain or bless a partly completed accumulation/backward/update. Resume
starts a new invocation/output directory from the last durable completed state.

The CLI adds `--checkpoint-every`, required `--snapshot-max-bytes`, and the four
joint resume inputs `--resume`, `--resume-sha256`, `--resume-bytes` and
`--resume-contract`. Expected bytes and contract come from a separately retained
receipt/contract, never from a payload/header being treated as its own authority.
The shared loader uses restricted data-only tensor deserialization. These are
trusted-local files; the envelope is not a malicious allocation sandbox.

Before model allocation the runner compares observable actual parent/data/source,
environment lock, tokenizer/template/stops, encoding, recipe and fixed endpoint.
Before state application it recomputes effective model/reference configuration,
attention backend, cache policy, functional/module dropout, trainable parameter
and optimizer ordering, tensor layouts and original reference identity. CUDA RNG
layout is validated before any saved policy/reference/optimizer tensor is applied.
Actual source/input/lock/parent bytes are rehashed on resume and before closure.
Role-bound dataset identities allow identical bytes to move; output paths,
snapshot cadence and invocation deadlines are not scientific recipe fields.

An initial completed boundary precedes baseline evaluation. Periodic snapshots
commit before their metric rows; the snapshot includes those rows, so an append
failure cannot erase a committed update. Final recovery commits before evaluation
or HF export. Evaluation preserves the training RNG. Output directories and
committed payload/commit files are not overwritten. No changed endpoint,
cross-source migration, mid-accumulation or pending-rollout claim is made.

## Measured comparisons and retained failures

[The verification record](2026-10-05-dpo-runner-recovery/run-02/verification.json)
contains the exact command, runtime, package/lock/source identities, raw focused
test output, every retained checkpoint digest and the numerical comparison.
Its SHA256 is `373405404bd8c21ac5d080ba8a62197fa4da39a25228bbb7b3d6d7944b572fbb`.

The uninterrupted run, update 3 restore in the same process, and a separately
launched Python process produce identical policy/reference tensors, optimizer
moments, sampler/global RNG, cursor, cumulative work and all six numerical rows.
The first post-restore row is also explicitly compared: update 4 selects `[1,1]`
and has loss 0.5337749719619751. Final policy and reference are different, while
the reference still equals its original parent; it has no trainable parameters
or gradients. Writer identity, parent receipt, output paths and measured elapsed
time are intentionally not numerical-equality targets.

| Completed update | Selected pair indices | Mean DPO loss |
| --- | --- | --- |
| 1 | 1,2 | 0.6931471824645996 |
| 2 | 0,1 | 0.650641143321991 |
| 3 | 0,0 | 0.5897907018661499 |
| 4 | 1,1 | 0.5337749719619751 |
| 5 | 1,0 | 0.4684799462556839 |
| 6 | 1,2 | 0.44729648530483246 |

Six updates expose 12 selected pairs, 32 chosen targets, 30 rejected targets and
110 logical branch tokens. Each model makes 24 training forwards covering 86 input
positions. The restored invocation performs only the remaining 6 pairs and 12
forwards per model, while retaining the complete cumulative totals. CPU training
elapsed 0.18887938500847667 s uninterrupted and 0.07233574299607426 s for the resumed
remaining updates, including the respective saves/diagnostics. These tiny timings
are not estimates of real checkpoint overhead or Spark throughput.

The collection deliberately raises a metric error **after** durable update 3:
the journaled metric file has rows 1–2, but the snapshot retains rows 1–3. The new
invocation explicitly writes recovered committed history and trains updates 4–6
once. Focused tests additionally retain final-save, initial-baseline, evaluation,
export and model-forward failure paths, existing-output refusal, original-reference
and external-digest/size/contract tampering, non-completed phase rejection, strict
integer/Adam/RNG/layout controls, actual tokenizer collisions and source groups.
Independent one-update calculations confirm the original summed response DPO
objective, single target shift and optimizer update without reusing the runner's
loss helper as their expected-value computation.

The first execution ran 19 tests in 3.764 s with 18 passes and one test-setup error:
HF 5.18 Qwen3Config no longer exposes the tested `rope_theta` attribute. That
[historical diagnostic](2026-10-05-dpo-runner-recovery/initial-test-diagnostic.json)
retains its original source hashes. The architecture mutation control now changes
context configuration without changing tensor shapes. No model dimensions, seed,
objective, learning rate or update budget were retuned. Final focused execution
inside the collector exits 0 with 21 passes; its measured process runtime is
5.946730447001755 s, including imports and the fresh-process subtest.

The first successful collection, `run-01`, is also retained unchanged. Its
collector called a typed tree digest `contract_sha256`, while snapshot headers
correctly used the separate canonical JSON contract hash. The
[identity clarification](2026-10-05-dpo-runner-recovery/identity-clarification.json)
records both values. The metadata-only collector correction labels
`contract_state_digest` and `fixture_state_digest` explicitly and adds the
canonical snapshot contract hash and actual fixture file-byte hash. `run-02`
repeats the unchanged runner/test/spec/recipe, obtains the same exact numerical
state digest and has no source drift. The canonical snapshot contract SHA256 is
`965d6e1c1e27601556993b57874ee165c694d55d596194877bc25d23b89dc292`.

Four independent greedy raw continuations are retained, including token IDs,
decoded special tokens, EOS versus cap, truncation and measured token counts.
Both initial continuations hit the four-token cap; both final ones terminate with
EOS, and one emits an unexpected assistant marker. These are random-model
diagnostics, not correct-answer, language-improvement or helpfulness evidence.
Internal HF generation forward work and unavailable partial tokens after errors
are explicitly not claimed as measured.

## Source identities and remaining production gates

The frozen runner SHA256 is
`f0a84e18015bc487e3944f592eb948fc605d9d8455c450cbe93d383d09b18bc7`;
the original focused test SHA256 is
`cf31bef76b2b8e7b47d405ccfa03337fe3a6e064afad661eedd02e577506eb35`.
The shared snapshot SHA256 is
`9cad2851b665b5fe5d640c1327710352396218ea80fab3b0c82f01925e1749a3`.
Before/after evidence hashes agree. Historical diagnostics are not silently
relabeled as current-source evidence.

The CPU fixture uses FP32, no CUDA autocast and no activation-checkpointing mode;
production enables gradient checkpointing and BF16 autocast. Pretrained model
compatibility, current-source CUDA/BF16 recovery, model-scale serialization cost,
cross-machine numerical equivalence, total-disk/token/work containment and the
separately approved supervised profile/pilot remain pending. Legacy Chapter 11
records without source-group provenance are explicitly labeled unverified;
unique IDs and collision-free prompts do not manufacture source independence.
The production hardware CLI itself was not launched by this CPU collection.

Reproduce into a **new** evidence directory with the existing isolated interpreter:

```bash
PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  experiments/reports/2026-10-05-dpo-runner-recovery/collect.py \
  --output /tmp/dongxi-dpo-recovery-new-output
```

The destination must not exist. The example interpreter is an actual retained
local environment, not an instruction to acquire dependencies or weights.
