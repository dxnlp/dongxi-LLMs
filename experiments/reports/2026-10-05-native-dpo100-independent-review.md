# Independent review: actual native DPO100

Review date: 2026-10-05. Reviewer scope: read-only source, small input and
receipt binding checks, retained metadata, metrics, and journal arithmetic.

## Finding

**CLEAR at the declared fixed DPO100 boundary.** No actionable inconsistency was
found in the six literal acceptance checks, actual launch and runtime witnesses,
full400 parent/replay03 prerequisite, completed-update evidence, work/I/O
journals, or exported-policy genealogy.

This is not acceptance of the whole campaign or a broad quality claim. The
common BF16-loaded/eager generation and retention comparison was pending when
this review was performed. The pilot's own four-answer panel is a distinct
FP32-weights/BF16-CUDA-autocast observation and must not be substituted for that
common comparison. No chosen100 or subsequent queue outcome is claimed here.

## Reviewed evidence and method

The actual producer is
`experiments/reports/native-dpo-pilot-20261005-run-01/`, its native output is
`outputs/native-dpo-pilot-20261005-run-01-pilot/`, its work/I/O journals are in
`outputs/native-dpo-pilot-20261005-run-01-pilot-journals/`, and its cooperative
snapshot inventory is in
`outputs/native-dpo-pilot-20261005-run-01-pilot-snapshots/`.

Independent checks used Python standard-library JSON, SHA-256 and arithmetic
only. All 14 frozen source bindings and all 21 input bindings were rechecked
against their retained byte sizes, resolved paths and SHA-256 values. The
additional full400 parent acceptance and four final snapshot metadata bindings
were checked separately. This includes small fixtures, lock/interpreter,
replay03 receipts/witnesses/stdout, source and tests, not weight payloads.

The reviewer did **not** import/load a model, run training or inference, run a
test suite, open or body-hash `.pt`/`model.safetensors`, change producer files,
alter package criteria/status, or perform Git/network operations. Checkpoint
payload sizes were inspected with filesystem metadata only. Weight and payload
digests below are retained producer assertions, not new independent body hashes.

Selected actual reviewed file digests:

| File in the pilot report directory | SHA-256 |
| --- | --- |
| `acceptance.json` | `1b881cb0d973a9a37f95d0ce0355c336477b0db233d419ab1abaf57d9df4505e` |
| `preparation.json` | `e35952f7730b45a37f8905a1d996334467daa7c3a45fdb0b067438658477bf8a` |
| `launch-pilot.json` | `85ae93f66161ab30b62c4eb4b4cd5f6c03aaad3eacc76607d3ee603224451b05` |
| `returned-supervision-pilot.json` | `9606116fae87b21ce7b2b562d16c491b3bd2f7f077b6ed55df4c7e607ab28509` |
| `supervision-pilot/result.json` | `e2d315e5301d5e87db582710fd504d9ebf38707c2c00728998dc7f432a5a3bfe` |
| `cpu-thread-witness-pilot.json` | `69d6924162238bb3cd55681ff4c2ba8c94d9018de68a8471962abc71526a8b25` |
| `snapshot-hash-witness-pilot.json` | `30e24d8eb835cc02e1bc4bfec51b232a69739e89b3c1ee9736db4b597b560be7` |

The preparation file's embedded **canonical object** digest is
`08ba684b2b89853737cee41ac7470f3cf2ffc4268736dca7a113aeb80b6db4c1`.
It differs from the serialized-file digest above by construction. Recomputing
the object digest with its digest field excluded passed.

## Prerequisite, source and scientific contract

The selected parent is exactly the predeclared accepted full400 export:
`outputs/native-sft-full-pilot400-20261005-run-01/policy`. Its acceptance file
SHA-256 is
`72b6438c0f57c74130f368796cf8210733e19a7b8ca26f80d1b7cddbce1d3c48`;
the actual file has `status: passed`, all checks true, 400 completed updates and
the same exported-policy path/file map retained in this pilot preparation. Its
genealogy is a full HF model from `Qwen/Qwen3-0.6B-Base`, pinned revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, not an adapter or a
quality-selected replacement parent.

The pilot's bound recovery prerequisite is actual accepted
`native-dpo-replay-20261005-run-03`, not failed run01/run02 and not a tiny CPU
control. Its acceptance and preparation serialized-file hashes are respectively
`a1f64ea171859fef58aafe134c648cdaf1ede2f3af0a9461b6465f1bcf546416`
and `f0d1e85e7e325c5758e7a98d5d011e07282b35c9089898c961241e1a236c2ee3`.
All its literal checks are true; the pilot and accepted replay03 preparations
have exactly equal source bindings, parent binding and encoded geometry.
The four replay roles' actual CPU/hash witnesses and stdout are included in the
pilot's input bindings. The prior independent actual review remains
[the replay03 review](2026-10-05-native-dpo-replay03-independent-review.md).
The unsuccessful historical run01/run02 are not relabeled or used as acceptance.

The frozen wrapper source `scripts/run_native_dpo_stages.py` verifies the
actual full400 parent at lines105–121, adds the actual replay03 inputs at
lines124–135, requires its literal replay checks and launch/witness bindings at
lines297–327, and revalidates preparation/source/input/parent bindings before
and after native work. Its pilot allowances at lines206–226 retain:

- 100 updates, accumulation4, seed1818, AdamW learning rate `5e-7`, beta0.1;
- maximum sequence512, generation64, completed snapshots every20 updates;
- FP32 policy/reference weights and BF16 CUDA autocast, dropout disabled;
- native-child deadline1800seconds and sampled OS reserve25GiB;
- original19-dimensional work and9-dimensional snapshot-I/O accounting;
- six snapshot saves, no snapshot inspect/load in this fresh100-update run;
- cooperative97GiB/32-entry snapshot inventory and4MiB journal envelopes.

The launch's argv exactly equals `preparation.commands.pilot` and both actual
supervisor command records. It begins a fresh100-update invocation from the
full400 export (`restored_update: 0`), not from the two-update replay export.
No scientific, quota, stopping, deadline or checkpoint-selection rule was
changed for this pilot.

## Literal acceptance and actual execution

`acceptance.json` has exactly these six true checks:

| Literal check | Independent retained-evidence check |
| --- | --- |
| `all_actual_children_completed` | Exactly one pilot invocation; actual native child exit0 and supervisor status `completed`; no failure, signal, stop reason, cleanup or journal error. |
| `actual_fixed_cpu8_threads` | Exactly one actual `DONGXI_DPO_CPU8` stdout witness: Torch intra-op8/inter-op1, OMP string8, exact native target. |
| `actual_bounded_complete_hash_workers` | Exactly one actual `DONGXI_SNAPSHOT_HASH` witness, schema `dongxi-snapshot-hash-runtime-v1`, actual ledger property4. |
| `completed100` | Final completed-phase snapshot header and result cursor100 agree, with bound canonical contract and final metadata. |
| `exactly100_metrics` | Exactly100 records, update sequence1 through100, one invocation, finite loss/margin/gradient norm, four valid fixture indices per row. |
| `clean_journals` | Independent raw work/I/O hash-chain and reservation/completion aggregation; no open, failed, known-partial or uncertain work. |

The actual stdout is454bytes and has SHA-256
`9a399a9f6c3b15de51ed1047aac4199d0ff1987d881213051f536a70b72c6665`.
Both retained witness JSON objects and acceptance's embedded witnesses bind to
these exact bytes and match the parsed actual stdout. Torch counts are therefore
observed child settings, not inferred from OMP environment text. The native
entry prints the hash witness only after constructing/restoring and validating
the actual ledger (`scripts/run_chapter11_spark_dpo.py:1233–1243`). Four hash
workers retain complete-file hashing; their presence is not a relaxed hash,
fewer-byte check, proof of speed causality, or enlargement of any budget.

Actual supervisor child time is **986.4903892719885seconds**, inside the
unchanged1800second native deadline. Supervisor interval is
986.539017128991seconds. Minimum point-sampled Linux `MemAvailable` is
**96,169,480,192bytes**, above the25GiB reserve. These are labeled point
samples, not continuous memory enforcement, peak-GPU-memory measurement,
cgroup containment or proof that the whole host was GPU-idle.

The retained supervisor result describes the point before final logger cleanup;
the returned record additionally has final writer acknowledgment, queue feeder
shutdown and the completed logger cleanup. All common fields except the expected
expanded `helper_cleanup` list agree. The returned record has
`final_record_retained: true`, neither queue feeder remains alive, and both
recorded helpers were cleaned without errors. Acceptance's invocation embeds
that exact returned record. The logger-helper exit−15 is not a native model
failure. This distinction follows the source's explicit post-write sequence
(`src/dongxi_llms/native_profile_supervisor.py:558–601`).

The separate owner after-completion console transcription
`outer-launcher-observation.json` records outer exit0 and
**998.0056047239923seconds**, from queue session64405. It is an auxiliary owner
observation, not a native producer receipt or prelaunch-bound terminal archive.
Its interval includes preparation, supervision and acceptance; it is neither an
additional invocation nor a value to sum with child time. Native result body time
955.9669898040011seconds is a third, narrower producer-body boundary.

## Actual training work versus separate evaluation/recovery work

Summing all100 metric rows independently exactly reproduces both
`attempted_training_work_this_invocation` and
`cumulative_committed_training_work`:

| Training quantity | Actual total |
| --- | ---: |
| Sampled pairs / sampler draws | 400 /400 |
| Chosen / rejected target presentations | 2047 /1600 |
| Policy / fixed-reference forwards | 800 /800 |
| Policy / fixed-reference forwarded positions | 25439 /25439 |
| Logical sequence tokens | 26239 |

The training target count is3647, not the4036 reserved allowance or the3683
training-plus-validation total. Each pair uses two policy and two reference
forwards; the400 draws are100updates×4, not the1200 separately charged private
sampler reconstructions in semantic validation. All recorded losses, margins
and gradient norms are finite. First loss0.6931471824645996 and final
loss0.0005021771539759357 show fitting of this fixed preference fixture, not
proof of a general preference or language improvement.

The raw work journal has a header plus218 hash-linked events,109 fully completed
tickets:100 update tickets,6 semantic-validation tickets,1 preference-validation
panel and2 generation panels. Its final chain is
`b856c9caba50ffeacb3ce020b1d30015fae6f389bf0a1160727c7c4efb00af5a`.
Independent event hashes, previous links, consecutive sequences, ticket
pairing, actual≤reservation inequalities, aggregate vectors and file identity
match the retained result; all reservations remain within the declared caps.

Separate non-training costs are:

- Preference validation: four pairs,36 response targets,8 policy+8 reference
  forwards,256 positions for each,16 evaluation forwards/512positions.
- Native before/after generation:8 requests,32 greedy actions/forwards,
  996 uncached positions, no reference forwards. The before panel uses19
  actions and the after panel13. Generation contributes260 logical tokens.
- Semantic snapshot validation:6 operations at cursors0/20/40/60/80/100,
  300 retained-history row visits,18 RNG-state visits,1200 private sampler
  reconstruction draws and22,470,254,702 tensor-element visits. Those visits
  are not additional optimizer updates or newly sampled training examples.

Consequently the whole completed work ledger has840 policy/808 reference
forwards,26691/25695 forwarded positions,3683 targets,48 evaluation calls,
1508 evaluation positions and26763 logical tokens. Reserved unused generation
allowance is not silently refunded or counted as actual outputs.

## Snapshot I/O, cooperative inventory and fixed reference

The raw I/O journal has a header plus12 events:6 save reservations and6
completions. Its final chain is
`320eef304b9c2d9e34f335802a2d8e6e069f68aed96d7d32d3fc174331b36b41`.
Independent record/link/ticket/arithmetic checks match the retained result,
with no incomplete or failed work and all whole-operation reservations within
the original caps. Actual shared snapshot-payload accounting is:

- 6 saves,0 inspect operations,0 load operations;
- 59,923,664,265 hash bytes and the same serialization bytes;
- 59,920,418,200 clone bytes,14,980,150,126 tensor-element visits;
- 42,656 tree nodes and760,252 primitive bytes.

These counters cover the hooked shared payload path, not every byte read by
artifact validation, metadata, journals, model export or the whole job.

The artifact journal's67rows (sequence0 through66) independently match all
hash links and its final receipt/head
`f18557db651e78fafdebe39d61f3f8243e39831e53e653210b297b6e29cb79fb`.
Receipt identity/length agree with the retained result. Usage is19actual and
19reserved entries,59,923,728,739actual pathname bytes and
59,927,899,253reserved pathname bytes, below the fixed32-entry/97GiB ceilings.
This is cooperative direct snapshot/staging/marker inventory, not a physical
disk quota or a cap on HF export/general output.

Six completed-phase metadata markers exist at0/20/40/60/80/100. The final
payload is10,782,064,684bytes and carries contract
`be915a562334ad6ba91cd04a5ab3cde1082c73b59409fe7e2021d892241addd6`;
its expectation/header/work receipt/artifact receipt metadata bindings agree.
The retained producer payload hash is
`990d7d9f0c19ae1443fe0b8eb68e3d5520b3380c86e9bcf5143a0c1c90be3a73`.
No payload body was reread by this reviewer.

The contract fixes the detached reference to the original parent with state
digest `86ddf28b00dcb8933461687256d3cb640dbaf1465b03a73d5784d93b7dafcf63`.
The source checks saved reference state against this digest during semantic
validation (`scripts/run_chapter11_spark_dpo.py:589`) before each successful
commit (`:719–740`); the six completed validation operations and
`reference_has_gradients: false` retain that producer-level unchanged-reference
evidence. This review did not perform an additional tensor-body comparison of
the100-update reference. The earlier actual replay03 component comparison is
separate evidence, not a reread of this pilot's final tensor state.

## Export genealogy and native four-answer result

The export is `outputs/native-dpo-pilot-20261005-run-01-pilot/policy`, kind
`full-HF-DPO-policy`. Its genealogy binds the same full400 parent file map,
three exact fixture hashes, the completed100 snapshot, exact Base tokenizer
revision and checkpoint interface. Its legacy-adoption record preserves
`legacy_adoption: false` and the actual parent genealogy. All small exported
file digests match acceptance. Retained producer weight SHA-256 is
`b9b1c15f6025943fa2d33a6ce3d7c6b308e5bbf8a5bd480fea8e3d5eb074052c`;
it was not independently rehashed here.

The native greedy, single-sequence, uncached generation helper is
`scripts/run_chapter11_spark_dpo.py:826`. Before and after panel files exactly
match the result's arrays. Every one of the four before and four after answers
has no error, no truncation and a declared stop151645. Exact matches move from
**0/4 to1/4**, with the after answers:

| Expected | Generated after | Exact |
| --- | --- | --- |
| `the purple pouch` | `purple pouch` | false |
| `the tall vase` | `the tall vase` | true |
| `the green folder` | `green` followed by newline | false |
| `the orange tin` | `orange tin` | false |

The article and full-answer requirements remain unchanged; token overlap is
not silently promoted to exact correctness. Natural stopping is a decoder
contract observation, not correctness. Native source/result retain
`encoded_prompt_collisions: none observed` while explicitly recording
`missing-legacy-groups; source-independent split evidence still pending`.
No source-independent generalization, broad preference benefit or retention
conclusion follows from this small fixture.

At this review boundary `acceptance.comparison` is null. A future common
BF16-loaded/eager, source-bound generation/retention consumer must be reviewed
on its own actual receipts. This review preserves that outstanding boundary and
does not change any original package acceptance/dependency array.
