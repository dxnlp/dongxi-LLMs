# A preparation record is not a launch

Historical first reference: this report describes the completed `run-01` source
revision. Its before/after hashes matched, but the DPO cap-file FIFO hardening and
one additional transitive source pin happened afterward. Keep this evidence;
it is not verification of the later source. The final current-source report will
be a separate preparation report and exclusive `run-02` collection.

The fixed campaign now has a nonexecuting, source-bound preparation compiler.
Its frozen CPU reference passed 44 controls: 19 new compiler tests, 14 existing
campaign/owned-leaf regressions and 11 mocked preflight tests. All 45 stage IDs
prepared without executable commands. Three authored fixture preparations passed
receipt validation; no model job started and none became production-ready or
launch-authorized. Every actual external campaign outcome remains null, and the
learner remains Day 9.

This is part of the remaining DXI-01/03 source preparation work, not a new
training comparison or completion of those external gates. The
[original specification](../specs/2026-10-05-production-stage-preparation.md),
[raw verification](2026-10-05-production-stage-preparation/run-01/verification.json)
and retained fixture files are the evidence. No frozen October 4 report, Day 9
training result, checkpoint or negative condition was rewritten.

## What was measured

Collection completed at 2026-10-04T23:33:42.533244 UTC, October 5 locally. The
isolated environment was Python 3.12.14 on Linux/aarch64, Torch 2.14.1+cpu,
Transformers 5.18.0, tokenizers 0.23.2 and PEFT 0.20.0. CUDA was hidden and Hugging
Face was offline; no installation or acquisition occurred.

| Retained command | Actual result | Scope |
|---|---|---|
| `test_production_stage.py` | 19 tests passed; exit 0 | Fixed compiler, original JSON data/receipts and refusal controls |
| `test_staged_campaign.py` | 14 tests passed; exit 0 | Nonexecuting campaign plus its original bounded self-owned CPU leaf children |
| `test_production_preflight.py` | 11 tests passed; exit 0 | Root-owned validators with authored observations, not real platform probes |

The collector itself exited 0 and took 5.226 seconds. Its 40 bound source, lock,
fixture and historical-input hashes were identical before and after collection.
An independent read-only hash check after collection matched every current file
and all three retained command logs. These are CPU control durations, not Spark
model throughput, profile latency or a notebook-learning assessment.

The positive fixtures were `story-profile`, `story-control-smoke` and
`assistant-dpo-smoke`. Each contains original tiny JSON data, a three-token
authored interface and parent-slot metadata, never model weights or a working
GPT-2/Qwen tokenizer. Their resource observations are injected claims, including
40 GiB available host memory, a 25 GiB reserve and a fictional private boundary;
they are not host measurements. Prerequisite preparation/result identities are
independently supplied fixture assertions, not completed upstream jobs.

Four explicit reference refusals also remain in the raw verification:

| Control | Actual outcome |
|---|---|
| Unknown stage ID | `ValueError`: unknown fixed campaign stage |
| Request supplies `approved: true` | `ValueError`: compile request schema mismatch |
| Approval reference absent | `ValueError`: independently retained receipt required |
| Stale observation time | `ValueError`: future-dated or stale observation |

The wider 19-test panel also exercises wrong hashes/lengths, duplicate/nonfinite
JSON, changed parent/interface/source/lock bytes, bool ceilings, missing or
forged backend summaries, substituted challenges, path escape, symlinked
ancestors/leaves, directories and FIFO refusal. Tests retain valid numerical
recipes rather than relaxing a failing boundary.

## What the source now does

[`production_stage.py`](../../src/dongxi_llms/production_stage.py) exposes two
different operations. `prepare_stage` accepts one of 45 literal allowlisted
IDs and records the current fixed branch, runner source, parent slots,
intervention, numerical budget and evaluation contract. Its default has no file
bindings and reports preparation only. `compile_stage` additionally checks a
separately supplied bounded inventory and independently retained prerequisite,
preflight and approval receipt bytes. Neither returns argv, a shell command or
an execution API.

Files are read through retained directory descriptors without following symlinks
at any component. Only bounded, nonempty regular files are accepted. Each
fixture inventory file must match an independently expected SHA256 and length;
receipt JSON has exact fields, bounded structure and duplicate/nonfinite-value
rejection. Receipt names cannot become filesystem paths. The retained receipt
root must be outside the artifact root, rather than treating the request's own
boolean as approval.

Preflight validation reconstructs the backend policy from the same preparation
and independently expected observation nonce/time. It revalidates both retained
observations and requires the resulting complete summary to match the receipt.
The root provider remains mocked and real production scope fails closed.
Preparation binds current compiler/runner/transitive numerical helper bytes,
the lock, interpreter binary, observed package/platform versions, actual fixture
input bytes, parent metadata and interface semantics. Changed science bytes
invalidate old fixture receipts.

Stages without a complete current adapter remain blocked even when authored
receipts are well formed. Chosen-only SFT, merge and comparison operations are
not silently relabeled as an existing DPO/SFT update. The original service-oriented
story launcher remains historical source, not an approved production launcher.

## Geometry and enforcement are different

For the fixed 100-update DPO pilot, accumulation 4 and maximum length 512, the
campaign's 204,800-position single-branch geometry is not total work. Chosen and
rejected branches each pass through policy and reference. Under the current
one-shift scoring path, the source-derived upper bound is 408,800 input positions
per network, 817,600 combined. It excludes evaluation, generation, retries and
backward recomputation; it is not a measured FLOP or dispatch count.

The RLVR declaration similarly keeps generated response slots separate from
growing-prefix forwards and old/current/reference rescoring. Without an explicit
prompt bound, prefix work stays unresolved rather than being inferred from the
response cap. Valid training targets are never inferred from padded geometry.

The DPO runner separately has cooperative actual-loop whole-job reservation
controls. This compiler does not yet project its coarse preparation ceilings
into that runner's 14-dimension cumulative ledger. That connection, other runner
coverage, actual physical aggregate quotas and bounded exports remain distinct
gates. A passing fixture neither reserves real resources nor proves that its
declared ceilings would suffice for a model.

## Initial failure preserved

The first focused source check ran 19 tests and exited 1 with 18 errors. Hashing
the existing 23,323,176-byte Python binary exceeded a 4 MiB source-file envelope.
The [retained diagnostic](2026-10-05-production-stage-initial-diagnostic.json)
records the actual failure and the initial source identities; the original tool
output was truncated, so it does not invent a complete raw log.

The repair separated interpreter identification into a 32 MiB hashing envelope.
Fixture files stayed 1 MiB, total fixture inventory 4 MiB and receipts 64 KiB.
This was an identity-read setup correction, not a changed stage recipe, a
relaxed model resource limit or a favorable post-test checkpoint choice.

## Reproduction and remaining gates

Use a new output directory; the collector refuses to overwrite an existing one:

```bash
PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  tests/test_production_stage.py --collect NEW_EVIDENCE_DIRECTORY
```

The current raw verification SHA256 is
`853ebb8854c8f304cab05e0c5958fee7832fe5537d8c31eab669d6a59663bf64`.
It binds compiler source
`96ea25d4051c44f93feec068b1bb85b800386d255bb422ba272607649bb2c153`,
compiler tests
`5f80c9601c36ebb78c7ab969864f49798ca4a21b865e46ef6427dd7d74041759`
and campaign source
`264b5cf562e297b3a3f05a6f44bd9588de0578d4a15d3fc9ec28319b8d329493`.
All 40 full identities and exact child invocations are in that verification.

These controls do not authenticate a real approving issuer, model checkpoint
lineage, actual tokenizer encodings or platform observations. Bounded file sizes
and regular-file refusal are not an OS-enforced I/O deadline or a hostile-writer
sandbox. Real private containment, physical aggregate quota, bounded live
observers, external watchdog, GPU-owner clearance and stage authority still
require separate platform integration and approval. Pretrained/CUDA/BF16 recovery,
Spark profiles/pilots, publication language evaluation, Mac and hosted execution
remain unmeasured. No actual stage launch, profile, inference, training, model
download, installation, service or Git operation occurred in this work.
