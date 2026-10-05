# Fixed preparation, no production launch

The current source-bound preparation reference passes 44 CPU controls: 19 new
compiler tests, 14 existing campaign/owned-leaf regressions and 11 mocked
preflight tests. All 45 fixed stage IDs prepare without executable commands.
Three authored local fixtures pass receipt validation and four explicit reference
refusals are retained. No model job started; no preparation is production-ready
or launch-authorized. The learner remains Day 9 and all 45 actual external
campaign outcomes remain null.

This is bounded DXI-01/03 preparation work, not authority for a profile, pilot or
model acquisition. The [frozen specification](../specs/2026-10-05-production-stage-preparation.md)
precedes these measurements. The current evidence is
[run-02 verification](2026-10-05-production-stage-preparation/run-02/verification.json),
including raw logs, fixture inputs, receipt bytes, exact commands and 41
before/after source, lock and input identities. Those hashes were unchanged
during collection and independently checked against current files afterward.

## Actual collection

The collector completed at 2026-10-04T23:38:10.348117 UTC, October 5 locally.
Its actual exit was 0 and elapsed time 7.712 seconds. This is CPU control time,
not model throughput or Spark profile latency. The environment was Python
3.12.14 on Linux/aarch64, Torch 2.14.1+cpu, Transformers 5.18.0, tokenizers
0.23.2 and PEFT 0.20.0. CUDA was hidden, Hugging Face offline and thread count
one; no installation occurred.

| Retained command | Actual result | Evidence scope |
|---|---|---|
| `test_production_stage.py` | 19 tests passed; exit 0 | Original JSON preparation and receipt controls |
| `test_staged_campaign.py` | 14 tests passed; exit 0 | Nonexecuting campaign and original bounded self-owned CPU leaf children |
| `test_production_preflight.py` | 11 tests passed; exit 0 | Mocked validator observations, not platform probes |

The three successful fixture IDs are `story-profile`, `story-control-smoke`
and `assistant-dpo-smoke`. Their files contain an original tiny data item, a
three-token authored interface and proposed parent-slot metadata, not model
weights or an actual GPT-2/Qwen tokenizer. Injected observations claim 40 GiB
host availability, a 25 GiB reserve and a fictional private backend. They do
not measure host memory or exercise that backend. The preflight receipt retains
`production_ready=false` and `launch_authorized=false` even on success.

The four explicit retained refusals are unknown stage, request self-approval,
absent approval reference and stale observations. The 19-test panel additionally
covers changed receipt hashes/lengths, duplicate/nonfinite JSON, bool ceilings,
changed parent/interface/source/lock bytes, missing adapters/backend gates,
forged summaries, substituted challenges, escaping paths, symlinked ancestors
and leaves, nonregular files and FIFOs. No failing stage was dropped or favorable
model checkpoint selected.

## Source contract and numerical boundary

[`prepare_stage`](../../src/dongxi_llms/production_stage.py) derives the branch,
runner source, parent slots, intervention, numerical declaration, evaluation
contract and budget from one literal allowlisted ID. Its default launches nothing
and leaves file bindings unresolved. `compile_stage` accepts exactly a stage ID
as the request; separately supplied inventory, prerequisite, preflight and
approval references must match independently expected bytes and hashes. Files
are bounded regular files opened through retained no-follow directory descriptors.
Receipts are strict bounded JSON, not argv, arbitrary request paths or a bare
approval boolean.

The compiler binds current runner and numerical helper bytes, the lock and
observed interpreter/package/platform identities. Parent/interface evidence is
only authored metadata. Prerequisite preparation/result hashes are independently
supplied fixture assertions, not measured upstream results. Complete preflight
observations are revalidated against the same preparation, expected challenge,
limits and explicit time; a substituted ready summary cannot bypass that check.
Actual production scope and absent stage adapters fail closed. No callable
launch result is returned.

Budget geometry is not total job work. The fixed DPO pilot's 100 updates,
accumulation 4 and maximum length 512 imply at most 408,800 shifted input positions
per network, 817,600 across policy and reference, because chosen and rejected
branches are both scored. The campaign's 204,800-position single-branch geometry
is not that total. These source-derived upper bounds exclude evaluation,
generation, retries and backward recomputation; they are not measured FLOPs.
RLVR response slots similarly do not bound growing-prefix/scoring work. Valid
targets are not inferred from padded positions.

The DPO runner separately has cooperative actual-loop cumulative reservation
proof. This compiler does not yet map its coarse declarations into the actual
14-dimension DPO ledger, complete other runner coverage, enforce a physical
aggregate quota or bound arbitrary exports. Those gates are not relabeled absent
CPU recovery, nor claimed complete by these preparation fixtures.

## Historical records remain separate

The [initial diagnostic](2026-10-05-production-stage-initial-diagnostic.json)
preserves the first actual exit 1: 19 tests ran with 18 errors because the
23,323,176-byte existing Python binary exceeded the source-read envelope. The
repair used a separate 32 MiB interpreter-identification bound; fixture files,
inventory and receipt bounds remained 1 MiB, 4 MiB and 64 KiB. This was an identity
setup correction, not a changed numerical recipe or model-resource allowance.
That tool output was truncated, so the diagnostic does not invent a complete log.

The [first reference narrative](2026-10-05-production-stage-first-reference.md)
and [run-01 raw verification](2026-10-05-production-stage-preparation/run-01/verification.json)
remain historical passing evidence for their 40-file source revision. Later
DPO cap-file FIFO hardening changed its runner source, and the campaign manifest
added the concrete `sampling_likelihood_lab.py` dependency used by reasoning
generation. Run-02 is a fresh exclusive recollection, not a rewrite of run-01.
The 45 stage declarations and scientific recipes are unchanged.

## Reproduction and limits

Use a new output directory; an existing destination is not overwritten:

```bash
PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  tests/test_production_stage.py --collect NEW_EVIDENCE_DIRECTORY
```

The current raw verification SHA256 is
`27c6efc5ecd44182b2c1d4eb2e52c65312d5e83a1843761bc745632dd0d6da14`.
Its full 41-file identity set includes compiler
`96ea25d4051c44f93feec068b1bb85b800386d255bb422ba272607649bb2c153`,
tests `27a95a7a0fffd09340d9785afb02fd275f5b773e2ced87ebf5a1554c3495f481`,
campaign `4391ed6bd73a07f16067ebd3eb7079ba3e5474414530e7efefbb312ff5d27953`
and current DPO runner
`227e9ab352779d8bc3a08ec7d0d62d34933b8f0bf9bddf3b35a8d4594ab4d0a8`.

Passing authored assertions does not authenticate a real approving issuer,
checkpoint lineage, tokenizer encoding or platform observation. Bounded plain
file contents do not establish OS-enforced I/O deadlines or a hostile-writer
sandbox. Real private containment, physical quota, bounded live observers,
external watchdog, GPU-owner clearance and stage authority remain separately
approved platform gates. Pretrained/CUDA/BF16 recovery, Spark profiles/pilots,
publication language evaluation, Mac and hosted execution remain unmeasured.
No actual stage, profile, inference, training, download, installation, service
or Git operation ran in this work.
