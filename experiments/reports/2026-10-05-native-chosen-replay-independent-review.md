# Independent review of native chosen replay 01

Date: 2026-10-05. Result: **CLEAR for the declared two-update recovery and course-integration claims**, with timing provenance explicitly qualified below. This review reads already saved JSON, metrics, journal records and source/prose only. It does not deserialize models/checkpoints, rehash tensor bodies, rerun comparison/training/tests, or alter a producer.

## Actual recovery evidence

The reviewed source is `native-chosen-replay-20261005-run-01/acceptance.json`, its separate `comparison.json` and four saved `supervision-*/result.json` records. All four supervised roles are completed with actual native exit `0` and empty cleanup-error arrays. The CPU comparison is a separate supervised CPU process, not a fourth native training trajectory.

The eight retained numerical component-digest vectors (`schema`, `model`, `optimizer`, `sampler_rng`, `torch_rng`, `cuda_rng`, `completed`, `history`) match across clean/source/resumed. All endpoints have completed cursor `2` and the same scientific contract digest, `982c95619db428219f8118b41f586bb41002829c9a77ab463cf03f45ce79928d`. Independently projecting the actual metrics onto update/loss/gradient-norm/indices/work reproduces clean/source equality and exact fresh-resume equality to the second source row.

Source/resumed preserve the same work-ledger and I/O-ledger IDs and file identities; clean has a separate ledger. Retained work/I/O receipts have no open or failed tickets. Snapshot readers add their counted I/O at comparison time; that operational spending need not equal the old numerical snapshot prefix. The fixed reference has identical initial/final digests and no gradients in all three child results. These checks inspect retained numerical-comparison evidence, not newly computed checkpoint body digests.

## Independently reproduced accounting

Actual metrics record each source/clean update as 20 chosen targets, with 132 and 134 policy positions respectively. The actual resumed metric is the same second 20-target/134-position window.

| Training-only boundary | Targets | Policy positions |
| --- | ---: | ---: |
| Recovered numerical two-update trajectory | 40 | 266 |
| Physical source plus fresh resume | 60 | 400 |
| All three native training children | 100 | 666 |

The shared physical work journal independently reproduces 60/400, with zero reference forwards in training. Its two separate preference-validation records add 16 actual reference-forward calls/512 reference positions. Including those observation lanes in the shared completed ledger does not turn them into training exposure. All-native metrics contain 20 replacement draws and zero rejected training targets.

The source's original and final four-location panels each have 0 exact answers, 4 declared natural stops and no cap truncations. Therefore the chapter's distinction between recovery, stopping and correctly following the location-only instruction is warranted; natural stops do not establish useful answering.

## Recipe, costs and prose boundary

Preparation/contracts retain the predeclared selected full400 parent, original 8/4/4 populations, seed 1818, replacement sampling, accumulation 4, learning rate 5e-7, length 512, greedy uncached cap 64, FP32 policy/reference and BF16 CUDA autocast. The two-update replay has cadence 1 and the unchanged 600-second external deadline/25-GiB sampled host reserve. Chosen still uses WorkLedger/SnapshotIOBudget without ArtifactBudget. No DPO hashing-worker or actual Torch thread-getter witness was found or inferred from requested OMP8.

The four actual `child_seconds` sum to **509.5152141539147**; the four supervisor `seconds` sum to **509.75196472596144**. These are different timing fields. The report's separate **546.650581** outer-launcher interval was initially missing a durable provenance link. The owner subsequently retained `native-chosen-replay-20261005-run-01/outer-launcher-observation.json`: an explicitly after-completion transcription of owned PTY session 38514, with `parent_monotonic_seconds` 546.6505807769718, null original-stdout file hash and `retained_before_launch: false`. The updated report correctly labels this owner console observation rather than a hashed terminal archive or native supervisor receipt. It is not independently remeasured by this reviewer, additive to the constituent child times, or another model invocation. No original acceptance criterion requires this auxiliary owner timing to have been retained before launch; actual native launch/supervision records remain separate.

`2026-10-05-native-chosen-replay.md`, Chapter 11's native-replay paragraph, worked solution 16 and the lab paragraph correctly limit the result to native recovery, preserve negative quality, separate numerical from physical exposure, and do not claim the separate 100-update pilot has run or starts from the replay endpoint. Source/snapshot tensor bodies and parent/export byte inventories were not independently rehashed by this review; existing recorded identities remain evidence from their original producers.

The accepted bounded recovery is not a broader capability, speedup, total-compute, human-review, whole-output-quota or learner-mastery result. No running DPO source or job was touched.
