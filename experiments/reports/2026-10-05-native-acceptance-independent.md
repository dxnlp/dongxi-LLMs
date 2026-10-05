# Independent actual native acceptance review

The fixed native full/LoRA recovery checks pass. The original BF16 merge does
not pass; the separately declared FP32 merge/reload check does. Recommend
closing **DXI-01's four unchanged original identity/interface criteria only**,
with the numerical-precision qualification below. This is not completion of
the overall improvement goal or its model-quality campaign.

The [byte/journal review JSON](2026-10-05-native-acceptance-independent.json)
was collected independently in20.431seconds. The
[criterion/identity supplement](2026-10-05-native-acceptance-criteria-independent.json)
rechecks four actual completed invocation identities, seven actual Base files,
current source/input bytes, and their canonical hashes. The two report-only
collectors import no Torch, deserialize no tensors/pickle, load no model and
run no inference/GPU job. Numeric merge observations are verified against the
original retained measurement records, not rerun here.

## Actual recovery, not a readiness proxy

Both recipes retain the original20-update horizon, pinned
`Qwen/Qwen3-0.6B-Base` revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, seed1212,
microbatch1/accumulation4, length256 and learning rate2e-5. LoRA is rank8 Q/V.
Original checkpoints0/10/20 are committed; fresh resume10→20 also saves10/20.

| Independently verified | Full | LoRA |
|---|---:|---:|
| Original training targets over20updates |455|455|
| Fresh replay training targets over10updates |229|229|
| Final model cumulative training targets |455|455|
| Exact numerical update11–20 records |10/10|10/10|
| Exact streamed raw tensor ZIP entries |1,243|761|
| Exact final raw generated records |8/8|8/8|
| Final development NLL,360targets |1.10018933084276|3.691563532087538|

Tensor equality is equality of every retained numeric storage-entry byte hash,
not a tensor-loading comparison or a claim about arbitrary pickle metadata,
Python RNG, other machines or different backend versions. All actual retained
training losses and gradient norms are finite. All five payload byte sizes and
SHA256 values per pair match their commits and work receipts. Stable recovery
contracts and both historical journal prefixes match the final independently
replayed journals. Current sources/input files reproduce invocation bindings.

The final cumulative SFT16 work is30updates/120examples:
455+229=684actual training targets plus four360-target development panels,
for2,124likelihood targets. This is below the predeclared2,350-target allowance;
the final numerical model still has455training targets, not684or2,124.
Each pair has six semantic recovery validations,70history rows and18RNG-state
validations. All16scientific and9I/O dimensions stay within their original
declared caps, with no open/failed ticket. Each I/O ledger accounts for
five saves, one inspection and one load, including retained-prefix consumption
that a resume must not erase. Exclusive raw journal copies are retained in
[the evidence directory](2026-10-05-native-acceptance-independent/).

## Actual external outcomes and retained failure

Each separately bounded invocation has a900-second deadline and25GiB sampled
host-memory reserve. The returned final ACK is observed as true; owned helper
and feeder cleanup reports no survivor/error. Stored supervision records that
were written before returning can still say unknown-at-write; those are not
silently rewritten to imply they observed a later ACK.

| Actual invocation | Exit | Child seconds | Sampled minimum MemAvailable,GiB |
|---|---:|---:|---:|
| Full original |0|73.495|103.358|
| Full first resume: conflict-stopped |−15|4.550|115.868|
| Full retry, new output |0|81.304|100.887|
| LoRA original |0|61.430|111.450|
| LoRA fresh resume |0|58.601|109.836|
| BF16 merge/reload: failed |1|14.336|114.597|
| FP32 merge/reload |0|41.637|109.274|

The full first resume remains an actual failure caused by a conflicting
parallel CLI process. Its receipt/cost/output are preserved, not promoted to
success or overwritten. Its abrupt-stop child identity remains partial/running;
the external returned failure record is authoritative for termination. The
later successful retry uses separately retained launch/output/deadline evidence
and the unchanged scientific recipe/cumulative journals.

MemAvailable is sampled, not a claim of a continuous physical low-water mark
or a hard whole-job quota. Per-save5GiB and cumulative I/O allowances are
declared upper bounds; the actual full payloads are1,503,441,805bytes initially
and about3.888GB thereafter. Actual LoRA payloads are about1.508–1.517GB.

## Merge result: keep the precision distinction visible

The original BF16/SDPA merge comparison used predeclared atol0.125/rtol0.015625
and failed at the first prefix: maximum absolute discrepancy0.67578125.
[Its failure receipt](2026-10-05-native-lora-merge-reload/failure.json) is
retained. No BF16 merged export success follows.

The separately predeclared FP32/SDPA diagnostic represents the same actual
BF16 Base values in FP32, with the same actual20-update rank8 Q/V adapter. Its
atol0.002/rtol0.001 were fixed before execution. Eight full-prefix comparisons
pass; reported maximum absolute logit discrepancy is
0.00005364418029785156, with last-position argmax agreement8/8. The original
measurement reports exact saved/reloaded logit equality across its eight
prefixes,24forwards/269prompt tokens/807full-prefix positions.

The independent review hashes every actual merged file against the original
inventory and checks its tokenizer semantics/template/stops against genealogy.
Total retained FP32 export size is2,395,662,856bytes, below its separately
declared4GiB output allowance. Its input identity separately binds the actual
Base/adapter bytes, command, precision/backend/tolerances and source/lock/input
identity. This verifies an explicit FP32 handoff; it does **not** establish
BF16 policy equivalence or nominate a selected downstream training parent.

## Original DXI-01 acceptance map

The original four acceptance strings and criterion-specific conclusions are
retained unchanged in the criterion supplement. The original criteria do not
require a BF16 merged policy or a400-update campaign; neither omission is used
to hide the failed BF16 comparison.

| Original criterion | Evidence and qualification |
|---|---|
|1.Record Git/dirty/source/environment/command/device/input/parent/interface/objective identity | Four actual native identities independently reproduce canonical hashes; actual Base/source/input bytes, recorded GB10/driver/Python/packages/lock, masked-loss/precision configuration and full/LoRA export genealogy are bound. Lock identity is not proof that every installed package was resolved from it. |
|2.Reject token permutation/changed stops/template/incompatible parent; support genuine reload and explicit legacy adoption | Historical actual CPU controls in the [identity report](2026-10-04-run-identity.json), unchanged inspected tests/source, and current full CPU closure. Actual native interfaces survive full/LoRA recovery and explicit FP32 export; no shape-only test substitutes for token identity. |
|3.Separate declared revision from actual bytes; preserve partial failures | Independently hashed actual pinned local Base/tokenizer/adapter bytes, original identity records and retained conflict/BF16 failures. Abrupt-stop partial child identity is not misreported as a finalized successful journal. |
|4.Test full checkpoint and explicit LoRA merge/reload compatibility | Actual full and rank8 Q/V LoRA fresh recovery has exact metrics/storage hashes; explicit separately declared FP32 merge has actual byte/interface/reload measurements. BF16 merging remains failed, clearly scoped. |

The current [CPU closure run03](2026-10-05-current-cpu-closure/run-03/cpu-verification.json)
passes: actual unittest footer1,134tests/192.459seconds/OK; math54files/
1,294expressions/zero issues; route check exit0; all76fresh notebook references
exit0. The receipt reports no executable-source drift. Earlier run02's
16watchdog isolation errors remain historical evidence rather than being
deleted. This CPU closure is Linux/aarch64, not Mac or hosted-CI evidence.

Recommendation: mark only original-scope DXI-01 complete after root verifies
unchanged acceptance/dependency strings. Preserve all old pending-check text
and explicitly route unfinished SFT→DPO/RLVR campaign/parent selection to
DXI-03, and later broader containment work to readiness planning. Do not quietly
delete historical pending assertions or redefine DXI-01 to mean those later
projects are achieved.

## Do not infer a successful assistant from a successful replay

Final full and LoRA panels both have0/8reported exact answers and0/8message-end
stops. Full truncates8/8; LoRA truncates6/8. LoRA's remaining two records are
not successful message-end termination. Lower teacher-forced NLL and exact
recovery establish narrower observations than generated-answer quality.

No400-update pilot, selected parent, pretrained DPO/RLVR campaign, broad safety,
Mac/hosted CI, hard whole-job quota, publication rendering or learner mastery
is established here. The overall goal remains active.

Two separate supplemental collection mistakes—dtype spelling and fingerprint
field selection—were repaired only in the independent report collector. Their
[first](2026-10-05-native-acceptance-criteria-collection-failure-01.json) and
[second](2026-10-05-native-acceptance-criteria-collection-failure-02.json)
actual failures/source fingerprints remain retained. Neither changed any
native source, original tolerance, result, identity or model invocation.
