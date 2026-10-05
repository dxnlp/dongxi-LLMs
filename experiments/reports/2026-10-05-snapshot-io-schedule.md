# Checkpoint IO schedule verification

The new pure calculator passed 25 focused tests and 960 independently enumerated
small schedules on 2026-10-05 UTC. It gives checkpoint IO its own nine dimensions;
the existing nineteen DPO model and recovery-semantic dimensions were not edited
or silently extended. These results verify arithmetic and input refusal, not
shared-reader enforcement, a production launcher or physical containment.

## Actual check and retained outputs

The [premeasurement protocol](../specs/2026-10-05-snapshot-io-schedule.md) was saved
before this command:

```bash
env PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python tests/test_snapshot_io_schedule.py --collect experiments/reports/2026-10-05-snapshot-io-schedule/run-01
```

The actual child command was the same interpreter with
`-m unittest tests.test_snapshot_io_schedule -v`. It exited 0, reporting 25 passing
tests in 0.208 seconds; the collected child duration was 0.249382954 seconds. The
first collection passed; no failed attempt was replaced. The raw
[verification record](2026-10-05-snapshot-io-schedule/run-01/verification.json) has
SHA256 `29ce962f0f1775887ddcddd20b651b615c48050bba141f7c3258da4fdfc6ef84`.
The [full test log](2026-10-05-snapshot-io-schedule/run-01/tests.json),
[primary requirements](2026-10-05-snapshot-io-schedule/run-01/requirements.json)
and [cadence-one requirements](2026-10-05-snapshot-io-schedule/run-01/cadence-one-requirements.json)
remain separate immutable measurement outputs.

The controls cover exact envelopes and nine-key vectors, all positive insufficient
dimensions, bool aliases, signed-64-bit overflow, missing and extra fields, old schema,
rehashed omitted operations, exact shorter read sizes, commit deduplication,
resuming at the final cursor, explicit diagnostic counts and an import in a fresh
process without Torch or Transformers. The exhaustive reference enumerates
literal inspect, load and save events rather than calling the production cost
calculator to compute its expected vectors.

## Declared allowance is not measured IO

The primary supplied envelope allows 16777216 payload bytes, 4096 visited nodes,
21646 tensor elements, 56248 tensor bytes and 262144 primitive bytes. Its node and
primitive values are declared examples, not observed reader counts. For two
updates, cadence two, one final diagnostic load and capacity for two complete
attempts, the requirement vector is:

| Logical dimension | Required allowance |
| --- | ---: |
| Inspect operations | 1 |
| Load operations | 3 |
| Save operations | 4 |
| Payload hash bytes | 134217728 |
| Visited tree nodes | 32768 |
| Inspected tensor elements | 151522 |
| Primitive bytes | 2097152 |
| Clone bytes | 224992 |
| Serialization bytes | 67108864 |

The fresh schedule saves cursors 0 and 2, then performs its declared diagnostic
load. A resumed schedule inspects and loads its independently retained parent,
recommits that durable cursor, saves later cursors and performs its diagnostic
load. DPO restore's duplicate semantic check does not become an invented second
generic load. Requirements use the componentwise worst resumed boundary, not a
claim that all capacity was consumed by one measured trajectory. Cadence one
adds cursor 1: its allowance is one inspect, three loads and six saves.

Individual inspect/load reservations use an independently declared exact positive
payload size; conservative schedule calculations use the maximum. Saves reserve
maximum bytes before serialization. `snapshot_hash_bytes` describes payload hash
bytes, not every contract hash or filesystem read. Node and primitive envelopes
must span the entire operation, including repeated contract/header visits, rather
than reset between phases. The pure module does not implement those guards.

## Source identity and remaining boundary

The three new source/protocol hashes were identical before and after collection:

- Module `bbd1f1cb93a196fba92697b1b842d0ada6326644f819bb47e83d16dfe219fb1b`.
- Tests `304ea5712e852ef39339b7e4fab38f6ff182713b48a7538fc1f6de4ff71b9ce0`.
- Protocol `46783ce45ff00d5853e0b919ce4083fb579c7198255a7cfb483e0fcb01b70f6b`.

The collector independently rehashed the previous DPO mapper, its tests,
validation-stage protocol and final run-03 raw record; all four remained unchanged.
Their old scientific recipe, nineteen-dimensional allowances and numerical
observations remain historical evidence, not rerun results from this task.

Reader admission still needs an independently retained same-journal prefix
before payload hashing or deserialization. Save admission must precede generic
contract/state scans and cloning. Actual runner identity hashing must not perform
an uncharged earlier checkpoint read. A checkpoint carrying a prefix does not
solve this circular dependency by itself. Supplied envelope values are not
authenticated architecture or supplier observations. This task neither acquires
models nor runs numerical training, GPUs, Mac, hosted CI, services or physical
quota controls; it does not change Day 9 learner progress or grant launch authority.
