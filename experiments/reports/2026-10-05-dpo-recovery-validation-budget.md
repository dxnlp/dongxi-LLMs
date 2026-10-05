# DPO semantic recovery validation accounting

The final bounded CPU reference passes 126 controls: 17 new validation-budget
tests and 109 existing DPO/shared snapshot regressions. Actual checkpoint-on DPO
replays from committed update three to six in a fresh process with its original
reference, Adam, histories and RNG exactly intact. Later failed validation
spending remains in the same physical journal. These are logical semantic
verification charges, not a quota for every CPU operation or permission to run
production. The learner stays at Day 9, package status stays 13 of 18 in bounded
CPU scope, and all 45 external campaign outcomes remain null.

The [specification](../specs/2026-10-05-dpo-recovery-validation-budget.md) was
saved before the new measurements. Current [run-02 verification](2026-10-05-dpo-recovery-validation-budget/run-02/verification.json)
records actual commands/exits, raw logs, observed environment and 23 source,
test, specification, compatibility and lock identities before and after
collection. All identities matched. Both test children and the fresh replay
child exited 0; the collector exited 0 after 37.456 seconds. No throughput or
Spark performance conclusion is drawn from this time.

## Explicit version boundary

The [actual DPO runner](../../scripts/run_chapter11_spark_dpo.py) now uses
`dongxi-dpo-logical-work-v2`: the original fourteen dimensions plus
`recovery_validation_operations`, `recovery_history_rows`,
`recovery_tensor_elements`, `recovery_rng_states` and `recovery_sampler_draws`.
Old fourteen-cap files and old budget schemas refuse, including a caller trying
to attach a v1 schema to nineteen caps. There is no inferred upgrade or refill.
The old unbudgeted CPU API remains valid for numerical controls.

Every actual semantic validator call is charged. Initial/periodic/final commits,
load callbacks and restore's duplicate checks do not share a free allowance.
Commit refreshes its carried journal prefix after its validation completes, so
that validation's charge is durable in the numerical snapshot. Restore binds
that historical prefix but retains all later journal records; it cannot rewind
capacity to the older weight cursor. Existing snapshot/artifact roots, receipts,
partial files and allowances were not changed by this implementation.

## Declared units and admission ordering

`recovery_validation_upper(contract, completed)` derives a bounded panel from
expected metadata before inspecting any saved tensor value. Let P be policy
state-dict elements, O optimizer parameter elements, n optimizer parameter count,
R CPU Torch RNG bytes and C total CUDA RNG bytes. One panel reserves:

- one validation operation;
- `completed` history rows and `completed * accumulation` replay draws;
- two plus CUDA-device-count RNG states;
- `3P + (3O + n if completed > 0 else 0) + 4R + C` tensor-element units.

The terms cover reference byte hashing plus policy/reference finite panels,
Adam finite/nonnegative checks and step scalars, and RNG probes/equality buffers.
They are a declared verifier envelope, not a profiler count of instructions or
allocations. Successful validation credits the complete panel. A failed panel
retains its entered upper envelope as uncertain rather than fabricating completed
per-scan work. Replayed sampler draws use a temporary generator; they do not
advance the training sampler or global Torch RNG.

Cheap exact-type/cursor/container bounds and expected layout arithmetic precede
hostile loops. Journal-prefix authentication must precede a reservation on an
unbound recovered journal. The whole panel is then permanently reserved before
reference tensor hashing, value scans, history replay or RNG probes. Actual
dense, real, nonquantized shape/dtype checks precede tensor scans/hashing.
Optimizer group metadata is compared against bounded observed structure, not
normalized through arbitrary recursive JSON. Counter/work/integer-metric values
have signed-64 bounds before arithmetic or finite conversion.

The five zero-cap controls show refusal before semantic digest, finite scans,
temporary RNG construction, history `randint` or state application. A failed
reservation fsync likewise prevents those operations and poisons the journal.
Malformed history, tensor, RNG, NaN policy, negative Adam, corrupt RNG bytes,
sparse layout, oversized nested optimizer metadata and huge integers retain
charged failed panels without state application. Absurd structural lengths
refuse before panel traversal. See the [focused raw log](2026-10-05-dpo-recovery-validation-budget/run-02/new-focused.log)
and [original focused source](../../tests/test_dpo_validation_budget.py).

## Actual loop and replay evidence

The unchanged fixture is a random one-layer Qwen decoder: vocabulary/hidden width
16, intermediate width 32, two query heads, one KV head, head dimension eight,
context 32 and zero dropout. Seed is 1818, horizon six, accumulation two, beta
0.2, AdamW learning rate 0.008, weight decay 0.01 and gradient clipping one.
The policy's actual gradient checkpointing is on; reference is frozen and cache
is off. Checkpoint cadence is three and snapshot envelope is 16 MiB.

Observed execution was Linux/aarch64, Python 3.12.14, Torch 2.14.1+cpu,
Transformers 5.18.0, tokenizers 0.23.2 and PEFT 0.20.0, with CUDA hidden,
Hugging Face offline and one CPU thread. No Hub weights or tokenizer were loaded.
Actual original-equation tests with checkpointing on pass in the
[109-test regression log](2026-10-05-dpo-recovery-validation-budget/run-02/existing-regressions.log).
Checkpoint-off/on comparison passes its predeclared atol=rtol=1e-6; all observed
maximum differences are zero in this CPU fixture, without extrapolating to
CUDA/BF16 or claiming general checkpoint-mode equivalence.

Unbudgeted checkpoint-on, budgeted checkpoint-on and fresh-process replay have
the same numerical digest:
`7ed7c945c62c1fd26ae2edee889eb41362468bbeee9ae26e0061883999aff255`.
It includes policy, original reference, optimizer, sampler/global Torch RNG,
counters and full committed history; invocation, work and artifact receipts are
outside that comparison. The next update-four metric/action row also matches
exactly. The original fourteen work caps, masks, numerical objective, sampler,
cursor, reference and checkpoint mode were not altered to obtain this result.

The interrupted reference commits update three, then deliberately fails metric
publication. A later load callback succeeds and a separately injected malformed
sampler-RNG dtype fails semantic validation, retaining ticket 13. The fresh child
opens the same journal, validates in its load callback, validates again in
restore, recommits the restored boundary, continues to six and loads its final
snapshot for inspection. The failed ticket and its capacity remain consumed.

| Reserved semantic units | Fresh six-update run plus final inspection | Interrupted plus fresh replay |
| --- | ---: | ---: |
| Validation panels | 4 | 9 |
| History rows | 15 | 30 |
| Tensor-element envelope | 141,418 | 329,008 |
| RNG states | 8 | 18 |
| Temporary sampler replay draws | 30 | 60 |

The uninterrupted snapshot carries its three commit validations; the final
inspection is a fourth panel, not silently added to that earlier snapshot.
One completed-three callback or restore panel costs one operation, three rows,
37,518 tensor units, two RNG states and six replay draws. Both duplicate calls
are visible in the fresh child's summaries. Training still reserves six updates
and twelve actual training draws; recovery replay draws are a separate dimension.
The [final work summary](2026-10-05-dpo-recovery-validation-budget/run-02/fresh-replay/after-final-load.json),
[failed validation](2026-10-05-dpo-recovery-validation-budget/run-02/later-failed-validation.json)
and [fresh numerical record](2026-10-05-dpo-recovery-validation-budget/run-02/fresh-replay/numerical.json)
retain these observations and the precise receipts.

## Historical failures and hardening

The [first compatibility diagnostic](2026-10-05-dpo-validation-budget-compatibility-first.json)
retains 109 tests with one failure and one error: an old whole-journal equality
assumed restore was free, and a joint artifact fixture's dynamically assigned
100,000 tensor units were insufficient for newly declared validation work.
Root's [predeclared compatibility addendum](../specs/2026-10-05-validation-budget-fixture-compatibility.md)
records the observation before restore, adds an explicit two-panel delta and
declares an allowance only for the new tensor dimension. Original fourteen caps
and numerical/artifact assertions remain unchanged.

The [constructor-spy diagnostic](2026-10-05-dpo-validation-budget-constructor-first.json)
retains 15 tests with one setup error: a substituted encoding digest reached JSON
normalization before the new schema gate. The cheap supplied-budget gate now
precedes observed encoding construction. No objective, tolerance or cap changed.

[Run-01](2026-10-05-dpo-recovery-validation-budget/run-01/verification.json)
passes 124 controls and exact replay at its then-current source identity.
Independent review subsequently identified an oversized optimizer-metadata walk
and unbounded scalar integers. Their bounded guards and two new controls produce
current run-02's 126-test result. Run-01 and all earlier reports keep their
original bytes; no historical reference is relabeled current.

## Limits and reproduction

Semantic panels do **not** cover the shared restricted loader's deserialization,
generic tree/finite scans before the callback, source/header/payload byte hashing,
serialization, snapshot cloning, journal integrity, application copies, initial
contract/model construction or every CPU operation. Nor do they establish
physical memory/PID/disk containment, wall-time enforcement or a complete output
quota. A separate versioned pre-deserialization/generic-check admission hook
and actual platform evidence are still needed for those boundaries. The existing
bounded snapshot reader and original guard remain intact; this report does not
claim they supply those missing charges or production safety.

Mapper/live encoding, supplier authentication, independent approvals, real
checkpoint lineage, platform/watchdog/GPU clearance and broad output routing
are separate gates. This source package launches no pretrained/GPU/Mac/hosted
stage, tests no capability and advances no learner assessment.

The final collector's actual invocation was:

```bash
PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python tests/test_dpo_validation_budget.py --collect experiments/reports/2026-10-05-dpo-recovery-validation-budget/run-02
```

Collection rejects existing destinations; use a new exclusive directory for
another reference. Current verification SHA-256 is
`2b6372edc52eec9c00b592a068eb6f569175fd75140f5e76d5f7c223ac00c56a`.
Runner and focused test/collector hashes are respectively
`ef6e366dc008366313fedcdb873dd41988a44167c909bc6f28d480263b134fd9`
and `c611f51f8edd1b0603f02ab94946342502f255b716b69bab2202b8a9b0fa62d5`.
All 23 exact before/after identities, complete raw logs and archived source bytes
are retained under run-02. Independent read-only review confirmed the final
metadata/scalar fixes and found no additional substantive scoped issue.
