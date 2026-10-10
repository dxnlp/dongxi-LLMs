# RLVR runner: identity, recovery and bounded work

This is the operational companion to [Chapter 13](../../book/chapters/13-group-relative-policy-optimization.md)
and [Lab 13](../../book/labs/13-group-relative-policy-optimization.md).
The CPU notebooks need neither pretrained weights nor a GPU. The commands and
contracts below describe the separate native runner; reading them launches no job.
Use the existing [experiment contract](../../experiments/specs/day-23-qwen-rlvr.md)
for a declared run rather than inferring a new budget from a historical result.

## Canonical interfaces

| Surface | Responsibility |
|---|---|
| [`dongxi_llms.qwen_rlvr_lab`](../../src/dongxi_llms/qwen_rlvr_lab.py) | Actual native collection, strict verification, update and checkpoint lifecycle |
| [`run_native_rlvr_stages.py`](../../scripts/run_native_rlvr_stages.py) | Declared recovery/pilot stages and their separate acceptance gates |
| [`run_native_rlvr_evaluation.py`](../../scripts/run_native_rlvr_evaluation.py) | Original common-panel evaluation of identified exports |
| [`verify_rlvr_runner_recovery.py`](../../scripts/verify_rlvr_runner_recovery.py) | CPU source/recovery verification; not pretrained evidence |
| [`close_native_rlvr_export.py`](../../scripts/close_native_rlvr_export.py) | Explicit consistency check on a selected completed export; does not refit weights or rewrite failed supervision |

The reference policy is the original pinned parent. The behavior policy is the
snapshot that collected a group. The current policy receives gradients. These
roles do not become interchangeable because their architectures match.

## New-run inputs

The native module requires exact model/revision, output and environment identity
plus complete numerical and resource contracts. The current parser is the source
of truth for option names; the specification supplies the scientific limits.

| Input group | Required fields and meaning |
|---|---|
| Parent/interface | `--model-dir`, immutable `--revision`, `--prompt-mode`; an audited `--template` only for a declared custom interface |
| Numerical recipe | `--updates`, `--group-size`, `--max-new-tokens`, `--seed`, `--lr`, `--device` |
| Host/environment | Existing `--environment-lock`, declared `--max-seconds`, independently supervised host-memory reserve |
| Numerical snapshot | Explicit `--snapshot-max-bytes` within the save/load envelope |
| Logical model work | `--work-limits` containing all 23 RLVR dimensions and `--work-journal-max-bytes` |
| Shared snapshot I/O | Complete `--snapshot-io-limits` and separate `--snapshot-io-ledger`; payload bound agrees with the snapshot envelope |
| Output ownership | New exclusive `--output`; do not overwrite retained runs or treat a new directory as a budget refill |

Shared identity binds saved token semantics, encoding, template, terminal tokens
and source/environment/input evidence before model loading. A raw Base interface
differs from native Instruct chat. Disabling thinking does not convert Instruct
weights into Base weights. Legacy interfaces require explicit audited adoption
through `--allow-legacy-interface`; that flag is not upstream authentication.
See the [checkpoint-interface notebook](../../notebooks/day-01/04_checkpoint_interface.ipynb)
and [identity report](../../experiments/reports/2026-10-04-run-identity.md).

The runner's primary collection uses temperature one/full support. Altering
sampling support, extraction, reward, accumulation or output limits to evade a
failed comparison creates a different experiment. A cap in a JSON file neither
establishes physical containment nor grants a new model-scale run.

## Completed and pending recovery

A completed snapshot restores the policy, original reference, Adam state,
source/RNG state and committed numerical history. A post-collection snapshot
additionally restores its already observed group: IDs, valid masks, old selected
log probabilities, texts, stopping events, rewards and advantages. Apply that
pending group once before recollecting. Mid-generation continuation and partly
applied optimizer operations are outside this exact full-group guarantee.

Resume supplies all of the following together:

- `--resume`: the selected numerical payload.
- `--resume-contract`: independently retained scientific/interface expectation.
- `--resume-sha256` and `--resume-bytes`: independently expected payload identity.
- `--resume-io-receipt`: the independently retained shared-I/O expectation.
- `--work-journal` and `--snapshot-io-ledger`: the same physical journals retaining
  all later failed spending; use a new exclusive `--output`.

An unchecked marker adjacent to an untrusted payload cannot authenticate itself.
Both expected journal prefixes bind before shared inspection/model allocation.
Identity collection must not hash the resume payload before its read is admitted.
A terminal pending pool refuses before output, journal or model creation. Bounded
metadata and other source/data identity reads remain outside the shared I/O claim.

Check [the runner-owned recovery plan](../PRODUCTION_RECOVERY_PLAN.md) and
[Appendix D](../../book/appendices/d-reproduction-and-environments.md#policy-reference-and-pending-rollout-identity)
for the state/observation distinction. [Day 25 Exercise 8](../../notebooks/day-25/02_ragged_kv_cache_and_exact_recovery.ipynb)
isolates one real tiny-CPU boundary; it does not substitute for a full native
save/load schedule or another platform's replay.

## Work and storage boundaries

Whole collection, application, current/old/reference scoring, evaluation and
recovery validation are reserved before their operations. Pool checking can
execute a likelihood forward or temporary RNG replay; verification work is not
new training exposure. The fresh runner save schedule has one initial save and
two saves per update: a pending pool and a completed update. A manual one-save
notebook microscope therefore has a different I/O schedule.

Keep these quantities separate:

| Quantity | Interpretation |
|---|---|
| Collected valid actions | Actually emitted actions, including sampled stop, excluding post-stop padding |
| Dense batch slots/draws | Rectangular sampling operations, including stopped-row spending where present |
| Applied response targets | Targets entering optimizer applications, including retained-pool replay |
| Completed numerical cursor | Optimizer updates in the restored trajectory |
| Physical cumulative journal | All admitted attempts and later verification/saves remain spent |

None is a FLOP measurement. A logical ledger is not a hard aggregate storage
quota, and a per-file limit is not a total output-directory quota. A configured
supervisor must independently demonstrate deadline responsiveness, worker
ownership and cleanup; a successful leader exit cannot supply missing terminal
logging acknowledgment. [Appendix D](../../book/appendices/d-reproduction-and-environments.md#supervision-and-durable-evidence)
keeps these claims separate.

## Read the retained native outcomes

The [native reasoning/RLVR report](../../experiments/reports/2026-10-05-native-reasoning-rlvr-evidence.md)
is the canonical evidence summary. Six initial conditions complete all records;
two Base/custom-chat conditions fail decoding. Their combined 680/800 coverage
includes two errors and 120 missing responses. Missing rows are not zero grades.

The original G4 recovery attempt fails cleanup despite native exits zero.
The separately declared fresh recovery passes; G8 has its own accepted replay.
Both fixed 16 pilots complete native iterations/export. G8 passes supervision;
G4 fails final logging acknowledgment and subsequently has an explicit export
consistency closure. The closure does not retroactively accept its pilot.
Both policies are evaluated on the unchanged common panel with sampled and
greedy conditions kept separate. All training groups have zero task-relative
advantages; optimizer applications and numerical/KL movement do not establish
reward-driven improvement.

For numbered receipt-reading questions use [Chapter 13 solutions 22–25](../../book/solutions/13-group-relative-policy-optimization.md#evidence-reading-extensions)
and [Chapter 14's evidence extensions](../../book/solutions/14-when-optimization-goes-wrong.md#evidence-reading-extensions).
Preserve original response IDs, failure verdicts, masks, denominators and actual
cost boundaries. Do not rewrite a frozen unsupported equation as a supported
correct answer, deploy an undeclared oracle selector or count a new output
directory as unused capacity.
