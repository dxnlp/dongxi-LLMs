# Actual chosen-only recovery from the selected full400 parent

The fixed native replay passes its [acceptance](native-chosen-replay-20261005-run-01/acceptance.json)
and [independent numerical comparison](native-chosen-replay-20261005-run-01/comparison.json).
This is a real pretrained Spark run, not the earlier random-model CPU control.
It validates recovery; it does not establish location-answering quality or
execute the separate100-update pilot.

## Recipe and observed outcome

All paths use the predeclared full400 export, original Chapter11 8/4/4 populations,
seed1818, replacement sampling, accumulation4, learning rate5e-7, length512,
generation cap64 and checkpoint cadence1. Policy weights are FP32 and native
forwards use BF16 autocast. The global valid-chosen-target mean is the training
objective; rejected responses and reference scores are separate validation work.
The [preparation](native-chosen-replay-20261005-run-01/preparation.json) retains
exact parent/interface/input/source bytes and the closed commands. Each child
keeps the original600-second external deadline and25-GiB sampled host reserve.
OMP8 is requested; these receipts do not contain DPO's separate Torch thread
getter witness. No observed Torch thread count is invented.

| Actual role | Supervisor `child_seconds` | Minimum sampled MemAvailable bytes | Actual exit |
| --- | ---: | ---: | ---: |
| Clean2 |148.172204|99,212,230,656|0|
| Source2 |148.538984|99,555,758,080|0|
| Fresh completed1→2 |148.474742|100,431,323,136|0|
| Independent CPU comparison |64.329285|109,192,208,384|0|

The four supervised intervals total509.515214 seconds. The separate owner
[console observation](native-chosen-replay-20261005-run-01/outer-launcher-observation.json)
reports546.650581 seconds for the outer launcher, including parent work. This
is an after-completion transcription of the actual owned PTY stdout, not a
hashed terminal archive or native supervisor receipt. These are different boundaries,
not additive costs or a compute/energy estimate. All cleanup arrays are empty.
Resource minima are sampled observations, not continuous guarantees.

The final contract hash is
`982c95619db428219f8118b41f586bb41002829c9a77ab463cf03f45ce79928d`
for all three numerical endpoints. Policy, Adam, private sampler, CPU/CUDA RNG,
counters/history and completed cursor match exactly, as does the resumed metric
tail. Operational work-ledger prefixes are intentionally not numerical state
equality: source/resume share their original physical work and I/O journals.
The separate diagnostic reference remains unchanged and has no gradients.

## Exposure, cost and quality are separate

Each complete two-update numerical trajectory sees40 chosen targets across8
replacement draws and266 full-sequence policy positions. The fresh resume
physically executes the second20-target/134-position window again. Thus
source-plus-resume presents60 training targets over400 positions, not40/266;
including independent clean training gives100 targets over666 positions.
These counts exclude validation, generation, semantic state checks and snapshot
I/O. Shared reservations include those additional lanes and must not be reported
as successful training exposure or summed into FLOPs.

Both source before/after location panels remain0/4 exact with4/4 natural declared
stops. An EOS/message-end stop does not establish that a requested location was
answered correctly. These recovery diagnostics do not select the later pilot or
replace its common retained-capability evaluation.

Chosen uses its original WorkLedger/SnapshotIOBudget path without an
ArtifactBudget instance. Per-payload16-GiB envelopes and disk-space planning are
not whole-output or physical quotas. Shared reader work is charged to I/O9;
independent component/reference hashing remains outside chosen19/I/O9 under
the external supervisor. DPO's separately reviewed bounded inventory parallelism
does not apply to this runner. Every original check and deadline remains.

This accepted state permits the fixed fresh100-update chosen control. It is not
its initialization: that pilot must start again from the selected full400 parent.
