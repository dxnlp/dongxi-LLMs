# Independent actual native DPO replay 03 review

Date: 2026-10-05. Result: **CLEAR at the declared native two-update recovery boundary**. No blocking discrepancy was found in the reviewed acceptance, launches, saved supervision, runtime witnesses, numerical comparison, metrics or physical accounting. This is not a result for the separate 100-update pilot, broad preference learning, or a causal speedup benchmark.

The reviewer read saved JSON/stdout/metrics/journals and hashed only small source/input/receipt/stdout files. No models or checkpoints were loaded, tensor/weight bodies rehashed, tests or GPU jobs run, or sources/producers/specifications changed. The existing DPO100 queue was not touched.

## Evidence identities

| Actual run-03 record | SHA-256 |
| --- | --- |
| `acceptance.json` | `a1f64ea171859fef58aafe134c648cdaf1ede2f3af0a9461b6465f1bcf546416` |
| `comparison.json` | `15aa790bcfbd1dc59151f80701b4837fc28c69be57398ac8d789ec4f7024b885` |
| `preparation.json` | `f0d1e85e7e325c5758e7a98d5d011e07282b35c9089898c961241e1a236c2ee3` |

All paths above are under `experiments/reports/native-dpo-replay-20261005-run-03/`. All 14 prepared source-file byte bindings matched on independent read-only recheck, including the previously reviewed DPO entry, CPU8 wrapper, stage adapter and ArtifactBudget implementation. The three input fixtures, environment lock and both work/I/O cap documents also matched their recorded byte bindings. Parent/weight inventories remain the original producers' recorded evidence; no new parent/tensor body digest was computed here.

## Every literal acceptance check

| Literal check | Independent retained-evidence check |
| --- | --- |
| `all_actual_children_completed` | Three native roles plus one separate CPU comparison have completed status, actual exit 0 and empty cleanup-error arrays. All four saved launch argv vectors exactly match terminal `child_command` vectors. |
| `actual_fixed_cpu8_threads` | Each actual stdout contains one CPU8 getter witness: OMP `8`, Torch intra-op `8`, inter-op `1`, with the correct native/comparison target. All four retained witnesses exactly match their embedded acceptance records and actual bound stdout. |
| `actual_bounded_complete_hash_workers` | Each native stdout contains exactly one actual-setting witness with schema `dongxi-snapshot-hash-runtime-v1`, workers `4`. CPU comparison stdout contains all three witnesses in `clean`, `source`, `resumed` order. All four retained witness records equal their acceptance copies; independently rehashed stdout bytes match each recorded digest. |
| `equal_numerical_components` | All three recorded ten-component digest vectors are identical: schema, policy, reference, optimizer, sampler RNG, Torch RNG, CUDA RNG, completed cursor, counters and history. This verifies the retained independent inspect/load comparison; the reviewer did not recompute state-body digests. |
| `unchanged_reference` | Every comparison endpoint matches the original reference digest `86ddf28b00dcb8933461687256d3cb640dbaf1465b03a73d5784d93b7dafcf63`; every actual child result reports no reference gradients. |
| `completed2` | All three endpoints have completed cursor 2; fresh resume reports restored cursor 1. |
| `same_scientific_contract` | Every endpoint uses `bee5044b18afb036d48423a4ccfa40faa7f2697209ba4b1e81936c1b9af3c579`; acceptance's parent inventory equals preparation and the contract's parent inventory. |
| `clean_journals` | Compared work/I/O ledgers have no open or failed tickets. All reserved and completed counters remain within their original limits. Source/resume retain identical work and I/O ledger IDs/file identities; clean is a separate trajectory. |
| `exact_numerical_metric_tail` | Independently projecting actual metrics onto update/loss/gradient-norm/margin/indices/work gives exact clean/source equality and the fresh resumed metric equals the second source row. |

The comparison's declared exclusions remain `resume_parent`, `work_ledger` and `snapshot_artifact_ledger`. Serialized payload hashes need not match when these operational histories differ; the numerical component equality does not claim byte-identical complete payloads or refunded spending.

## Actual timing and resource boundary

| Supervised role | `child_seconds` | Supervisor `seconds` | Minimum sampled MemAvailable bytes | Actual exit |
| --- | ---: | ---: | ---: | ---: |
| Independent clean2 | 319.773992 | 319.819049 | 95,734,423,552 | 0 |
| Source2 | 296.337269 | 296.382683 | 96,150,581,248 | 0 |
| Fresh completed1→2 | 392.908058 | 392.957157 | 86,374,494,208 | 0 |
| Independent CPU comparison | 127.596575 | 127.643941 | 101,931,466,752 | 0 |

Every saved supervisor retains the original 600-second deadline and 26,843,545,600-byte (25-GiB) reserve. All four intervals finish within that limit, and sampled minima remain above the reserve. The separate totals are **1,136.6158937330474 child-seconds** and **1,136.8028306140332 supervisor-seconds**. They are overlapping timing boundaries, not additive costs, whole-parent duration, FLOPs or energy. The CPU comparison is not a fourth native training child. Samples do not establish continuous memory enforcement or general GPU-idle proof.

Returned supervision differs from the earlier persisted terminal record only in expected post-return final-record/helper-cleanup/queue-feeder metadata. Native exits, timing, commands, resource minima and limits agree; returned cleanup records report helpers/queue feeders no longer alive without errors.

## Unchanged method, caps and physical work

Run-03 preparation's science, encoded geometry, full limits and selected full400 parent exactly match run-02 preparation. Actual commands/contracts retain two updates, seed 1818, accumulation 4, beta 0.1, AdamW learning rate 5e-7/weight decay 0.01/clip 1, maximum length 512, cap 64, cadence 1, FP32 policy/reference and BF16 CUDA autocast. The opt-in worker setting and new source/run identity are execution provenance, not a changed scientific recipe or larger allowance.

The DPO19 work caps, I/O9 limits/envelopes, 16-GiB per-payload ceiling, 86,973,087,744-byte snapshot-artifact ceiling, 32 entries, and 4-MiB journals are unchanged. Source and resume preserve the same actual artifact ledger identity/root; clean has a separate ledger. Resumed actual usage is 16 entries/49,141,624,137 pathname bytes, below the original bounds. This snapshot-pathname budget is not a whole-output or physical disk quota. Repeated ArtifactBudget inventories remain complete-file integrity work outside declared I/O9 counters, as documented previously; worker count does not create a new charged/scientific dimension.

Actual metrics distinguish numerical and physical exposure:

| Training-only boundary | Chosen targets | Rejected targets | Policy positions | Reference positions |
| --- | ---: | ---: | ---: | ---: |
| Recovered two-update trajectory | 40 | 32 | 508 | 508 |
| Physical source plus fresh resume | 60 | 48 | 764 | 764 |

The actual shared work journal independently records three physical updates/12 replacement draws/108 total valid training labels and 764 positions in each policy/reference path. Validation, generation, recovery-state checks and snapshot I/O are separate additional work. The final shared I/O journal contains three inspect/load pairs and five saves; comparison reading is not free or a cap reset.

All reviewed before/after location panels remain 0/4 exact with 4/4 declared natural stops. The training loss moves from 0.6931471824645996 to 0.6864173412322998, but two-update optimization, stopping and useful preference behavior remain different claims.

## Historical failures and present scope

Failed resumed children from run-01 and run-02 still report `actual_exit_code: null`, deadline stops and `Owned leader reap timed out`. Their saved supervisor intervals remain 600.6833310350194 and 600.682756085007 seconds respectively. Run-03 does not overwrite those outcomes, recover their originally unobserved native exits, refund their spending, or convert their CPU/source controls into actual successes.

The earlier deadline-analysis report correctly supports a code-based repeated cached-read explanation without claiming a measured causal speedup. Its run-02 readiness language is historical now that run-03 has separate actual acceptance; the parallel course-integration update should provide a current-result bridge while retaining the old failures and diagnosis. This review establishes the current bounded recovery result only. It does not certify the running/future DPO100 pilot, common preference comparison, remaining reasoning/RLVR jobs, all-course verification, package completion or learner mastery.
