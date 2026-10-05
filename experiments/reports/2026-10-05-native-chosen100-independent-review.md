# Independent review: actual native chosen-only100

Review date: 2026-10-05. This is a separate read-only review of the completed
chosen-only pilot, not a rerun or a review of the still-running common consumer.

## Finding and boundary

**CLEAR at the fixed native chosen-only100 boundary.** The five literal
acceptance checks, accepted full400 parent/chosen replay01 prerequisites,
source/input and final metadata bindings,100 metrics, exact matched chosen
exposure against accepted DPO100, reference assertions, work/I/O journals and
full-policy export genealogy are consistent.

No broad quality or equal-compute claim is supported. The pilot's native
FP32-weights/BF16-autocast four-answer result is0/4 before and4/4 after, with
natural declared stops. The common BF16-loaded/eager evaluation and retention
comparison was still running at this review boundary. Its intermediate files
were not consumed, and this pilot panel must not replace its final evidence.

## Evidence and verification method

Reviewed actual receipts are in
`experiments/reports/native-chosen-pilot-20261005-run-01/`, actual output in
`outputs/native-chosen-pilot-20261005-run-01-pilot/`, and work/I/O journals in
`outputs/native-chosen-pilot-20261005-run-01-pilot-journals/`.

Independent standard-library JSON/SHA-256/arithmetic checks reverified all19
frozen source bindings and10 input bindings against current resolved paths,
byte sizes and hashes, plus the full400 parent acceptance and3 final snapshot
metadata bindings. This covers the source/tests/specs, three fixtures, group
sidecar, interpreter/lock, actual chosen replay01 acceptance/preparation and
pilot caps. No model or checkpoint body was opened or hashed; no Torch import,
inference, GPU job, full CPU/test suite, Git/network operation, source edit,
producer rewrite, status change or acceptance/dependency edit was performed.
Snapshot body sizes were checked with filesystem metadata only.

Selected reviewed serialized-file SHA-256 values:

| Pilot receipt | SHA-256 |
| --- | --- |
| `acceptance.json` | `372db9daf108431e011694c3adf186eb7f04004d0b25896b865731f6b165f12d` |
| `preparation.json` | `db9776091bc7cbbba01c20aa9af1a80929f83930f1506dd0254f6075e28d22dd` |
| `launch-pilot.json` | `8808a5f72f745a44e42bbc1a8ddddff6f74b9e2d8988e5ed34fbb7e2ab18452a` |
| `returned-supervision-pilot.json` | `b0792d244f784fc4e61fe06adee5577fb45ba8877d67d78b13bc6fce3c4305f7` |
| `supervision-pilot/result.json` | `a3c64e36f71690fe7aec54d2e2736d9246727ffc8f89ad8da6e68cfdaf73f18f` |

Preparation's embedded canonical-object digest is separately
`826d700e06c0d8fd40668efa5825197859ba0a150055b9c6823d6ab5b4241d75`;
recomputation with its digest field excluded passed. It is not the serialized
preparation file digest.

## Parent, recovery gate and unchanged recipe

The parent is exactly the predeclared accepted full400 export,
`outputs/native-sft-full-pilot400-20261005-run-01/policy`. Its accepted400
update/result/file map and Base-model genealogy match both chosen100 and
DPO100's retained parent binding. Parent acceptance SHA-256 is
`72b6438c0f57c74130f368796cf8210733e19a7b8ca26f80d1b7cddbce1d3c48`.
No adapter or quality-selected replacement was used. Base tokenizer/model
revision remains `Qwen/Qwen3-0.6B-Base` at
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.

The chosen prerequisite is actual accepted
`experiments/reports/native-chosen-replay-20261005-run-01/`, with acceptance
SHA-256 `97d5a4018ce669cd7ec2b01ae614468dbf956f82975ea8e58f1f1cead65503b7`
and preparation SHA-256
`11acfc73e86706cabcce67c01a05da6cdb626554bddf858b962332bf9b59980e`.
Its clean/source/resumed/CPU-comparison roles all exited0 and all literal
checks are true. Chosen100 has exactly the same source bindings, parent binding
and encoded geometry as that actual accepted replay01. Its earlier independent
review is [the actual chosen replay review](2026-10-05-native-chosen-replay-independent-review.md).
This prerequisite is not an authored tiny CPU control or a planned replay.

The actual launch equals the closed prepared pilot command plus its retained
operator declaration and exactly matches both supervisor child-command
records. It begins fresh from full400 (`restored_update: 0`). Recipe remains
100updates/accumulation4/seed1818/learning-rate`5e-7`/weight-decay0.01/clip1,
maximum sequence512/generation64, greedy uncached decoding, FP32 policy and
reference weights, BF16 CUDA autocast, dropout disabled. Its objective is
summed native chosen-token NLL divided by the accumulation window's valid
chosen target count, not DPO's mean pairwise objective. Beta0.1 is retained for
the separate preference validation, not an invented chosen-training multiplier.

Original native deadline1800seconds, sampled reserve25GiB,19-dimensional work,
9-dimensional I/O,4MiB journals, six saves at0/20/40/60/80/100, and the16GiB
per-payload envelope remain unchanged. The wrapper keeps110GiB output-space
planning. ChosenSFTLoop has **no ArtifactBudget integration**; planning and
payload/I/O limits are not cooperative artifact or physical disk quotas.
No actual Torch-thread-count or artifact hash-worker witness was fabricated
from OMP text or from DPO's separate runtime witnesses.

## Five literal acceptance checks and actual timing

`acceptance.json` has exactly these five true checks:

| Literal check | Independent retained-evidence verification |
| --- | --- |
| `all_actual_children_completed` | One actual native pilot, exit0/status `completed`, no failure, signal, stop reason, cleanup or journal error; returned record matches acceptance. |
| `completed100` | Completed100 header/result/expectation and bound canonical contract agree, with six completed metadata markers. |
| `exactly100_metrics` |100 finite records in consecutive update order1–100, one invocation and four valid fixture indices per row. |
| `unchanged_reference` | Actual retained start/end typed-reference digests equal, and no reference gradients; source computes these before training and after export. |
| `clean_journals` | Independent raw work/I/O hashes, consecutive previous links, complete tickets, within-cap reservation/actual vectors and aggregate summaries agree. |

Actual supervisor child time is **426.071010885993seconds**, supervisor
interval426.11806763499044seconds and minimum sampled Linux `MemAvailable`
**100,037,660,672bytes**, inside the native1800second deadline and above25GiB.
These are labeled point samples, not continuous memory enforcement, GPU peak
memory, GPU-idle proof or hostile-tree/physical-quota containment.

The retained supervisor file precedes final logger cleanup. Returned common
fields agree except the expected expanded helper-cleanup list; its final
writer ACK is true, queue feeders are stopped and helper cleanup is complete
without errors. Acceptance embeds that returned record. A logger helper's
cleanup signal is not the native model's exit code.

The separate owner after-completion transcription
`outer-launcher-observation.json` records queue64405 outer exit0 and
**456.0562492070021seconds**. It includes preparation/supervision/acceptance
and is not a prelaunch-bound terminal archive, native receipt or additional
model invocation. Native result body time422.1424599030288seconds is a
narrower boundary. None of these intervals should be added together.

## Exact matched chosen exposure, not equal compute

All100 chosen100 metric index lists exactly equal the accepted DPO100 lists,
and every update has the same chosen-target count and four pair draws. The
actual recovery contracts also bind the **same complete paired encoded train
IDs and boolean masks** with digest
`052f3c980ff80b9d14cd16f80ea354c18c3ced93e6d654e4e9b92b9c4ba34e15`.
Validation encodings, parent/interface, tokenizer and evaluation-prefix
geometry match. The source uses the native DPO pair encoder, takes its chosen
branch without altering IDs/masks, maps masked positions to−100 for native
SFT labels, and applies one causal shift. The real turn ending151645 and
trailing newline198 remain supervised; no fake stop or suffix trimming was
introduced (`src/dongxi_llms/chosen_sft_control.py:281–334`,
`scripts/run_native_chosen_stages.py:118–156`). This is a source/retained
encoding binding check, not a new tokenizer/model execution.

Independent summation of all100 actual metric rows gives:

| Training-only quantity | Chosen-only100 | DPO100 |
| --- | ---: | ---: |
| Sampled pairs / draws |400 /400|400 /400|
| Chosen targets |2047|2047|
| Rejected targets |0|1600|
| Policy forwards |400|800|
| Reference forwards |0|800|
| Policy forwarded positions |13343|25439|
| Reference forwarded positions |0|25439|
| Logical sequence tokens |13343|26239|

The chosen loop uses a native full-input forward while DPO forwards the
causally shifted chosen and rejected inputs. Both supervise the matched chosen
target positions, but their forward geometry, objectives, reference work,
snapshot sizes and elapsed times differ. Neither equal update count nor equal
chosen exposure makes them equal-compute experiments. Their absolute loss
numbers are also different objectives, not a shared quality scale. Chosen loss
falls from2.5518688201904296 to0.0006960701153037094; this is fitting of the
small controlled fixture, not a broad capability conclusion.

## Work and I/O journals, including separate reference validation

The raw work journal is a header plus218 hash-linked events,109 completed
tickets:100 chosen updates,6 semantic validations,1 preference-validation
panel and2 generation panels. Final chain is
`327fa84c1631238e60faf82c42a8d393809323f36b0b2700f2056f1c89f02fa0`.
Independent hashes/links/sequences/ticket pairing, file identity, all
actual≤reserved≤cap checks and vector sums match the retained result. There
are no open/failed tickets or known-partial/uncertain counters.

Training contributes2047 targets,400 draws and400 policy forwards/13343
positions, with **zero training reference forwards**. Separate work is:

- Preference validation: four pairs,36 targets,8 policy+8 fixed-reference
  forwards/256positions each;16 evaluation calls/512positions.
- Before/after generation:8 requests,35 actions/forwards,1091uncached
  positions,263 logical tokens; no reference forward. Before19 actions and
  after16 are actual outputs, not the512 reserved generation-token allowance.
- Semantic snapshot validation:6 operations,300 history-row visits,
  18 RNG-state visits,1200 private sampler reconstruction draws and
  13,450,666,094 tensor-element visits. These are not new training examples.

The whole ledger consequently contains2083 targets,443 policy and8 reference
forwards,14690/256 forwarded positions,13870logical tokens,51evaluation
calls/1603positions and35generation actions. Saying the whole invocation has
zero reference forwards would be false: the eight separate validation
reference calls are explicitly retained.

The raw I/O journal is a header plus12 events, six fully completed save tickets,
with final chain
`381461c47d249d3bd1545c47e50e15450d65f750ce5e2ac77b5644cdd2678030`.
Independent chain/arithmetic/identity checks pass with no failed or open work.
Actual shared payload accounting is6saves/0inspect/0load,
41,883,858,042hash and serialization bytes,41,881,240,984clone bytes,
10,470,355,822tensor-element visits,39818tree nodes and679966primitive bytes.
All reservations fit the original caps. These counters exclude independently
typed reference/component hashing, metadata, journals, model export and
general job I/O; no claim is made that all physical I/O is counted or bounded.

## Reference, completed export and native answers

Actual retained start and final reference digests are both
`86ddf28b00dcb8933461687256d3cb640dbaf1465b03a73d5784d93b7dafcf63`,
also equal to the DPO100 fixed-reference contract digest; no reference gradients
are present. `scripts/run_native_chosen_stages.py:457` and`:492–498` compute
and retain the actual start/end values; these typed hashes are outside
chosen19/I/O9 under external supervision. The reviewer did not reopen the
reference tensor body or perform a new tensor comparison.

The final completed payload is7,775,429,411bytes, producer SHA-256
`45c30047d234402ae1f9b6df81b467aa4b71983c64240d25c8d4e009e3e729db`,
and scientific contract
`848eec80c25cb75f542c690c1685e7ca10fe20bc0dc778df26187078dc5ae452`.
Its header/expectation/work receipt and actual result agree. Six completed
markers exist at0/20/40/60/80/100. No body digest was independently recomputed.

Export `outputs/native-chosen-pilot-20261005-run-01-pilot/policy` has kind
`full-HF-chosen-policy`, the exact full400 parent/file map, same interface,
completed100 recovery, canonical scientific contract and preserved non-legacy
parent genealogy. All small exported file hashes match acceptance. The retained
producer weight digest is
`8b16a7a8a3777c2d6e9e7ad24700c26daac46252907ac9164091140219f11b64`;
it was not independently rehashed here.

Both panel files exactly match the retained result. The native before panel's
prompt IDs, generated IDs, expected strings and0/4 exact results also match
DPO100's native before panel. The chosen after answers are exactly
`the purple pouch`, `the tall vase`, `the green folder`, `the orange tin`:
**4/4 exact**, and all four naturally stop at151645 without error/truncation.
All four before outputs also naturally stop. This is a small held-out scenario
panel under the native FP32/BF16-autocast execution recipe, not proof of general
preference learning, instruction following or retention.

The protocol sidecar provides explicit disjoint scenario groups and the actual
chosen split check reports no encoded-prompt collisions. This improves fixture
group bookkeeping; it does not turn eight training/four validation/four
evaluation scenarios into a broad independent-source benchmark. At review time
`acceptance.comparison` remains null. The common BF16-loaded/eager generation
and retention consumer requires its own completed receipts and independent
review; partial consumer output was not counted here.
