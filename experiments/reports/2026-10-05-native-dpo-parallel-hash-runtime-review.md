# Independent DPO bounded-hashing runtime review

Date: 2026-10-05. Result: **PASS at the scoped source and tiny CPU-control boundary**. This is not a native training, CUDA replay, pilot, or performance result.

## Reviewed candidate

| File | SHA-256 |
| --- | --- |
| `scripts/run_chapter11_spark_dpo.py` | `dafbd73e1f8ee2c13ca866c58f523990deadfdb441a830f54b16428423a2424b` |
| `scripts/run_native_dpo_stages.py` | `a150cdc182919a7f3af74cfb1f6d7e52ab328c65f329d3c8291091185782e7c0` |
| `scripts/run_native_dpo_cpu8_child.py` | `64ea08157c87f54a846bf1259c5756d351708b8ee689c0f1b44977f779090e60` |
| `tests/test_native_dpo_stages.py` | `0d73808f2c5d0cd0c15f7bfa5493c7120733ea74a1e680d45b8b9291682c6595` |
| `src/dongxi_llms/artifact_budget.py` | `af1c1fcc4349393e50f5603e85017f851128c92c34dbcb82f75aa4694efdacc0` |

The hashes were independently checked after review. No producer source or tests were edited by this reviewer.

## Runtime and evidence boundary

- The native CLI remains serial by default (`1`), accepts only integer choices `1` through `4`, and passes the chosen setting to the existing artifact-ledger create/restore paths. It emits `DONGXI_SNAPSHOT_HASH` with schema `dongxi-snapshot-hash-runtime-v1` and the actual ledger's readonly `hash_workers` value after ledger validation.
- Every fixed DPO native child command requests `4`. The CPU comparison restores all three existing ledgers with `4` and emits actual-setting witnesses in the exact roles/order `clean`, `source`, `resumed`.
- Parent-side checks bind the actual child stdout digest, reject absent, incorrect, extra, reordered, or mutated runtime witnesses, and retain the witness records. A successful child exit alone does not satisfy this gate.
- The replay gate selects `run-03`, its current source/input bindings, current supervision records, actual CPU8 witnesses, and hashing witnesses. Failed `run-01` or `run-02` cannot stand in for this new replay.
- Scientific recipe and numerical comparisons, the original nineteen-dimensional work/I/O accounting, artifact quotas, physical journal continuity, complete-file integrity checks, and the existing 600/1800-second supervision deadlines remain unchanged. Worker count is a runtime setting, not a new scientific claim or budget refill.
- `scripts/run_native_dpo_cpu8_child.py` is unchanged. The new source identity still requires new actual source-bound replay evidence before the pilot may proceed.

## Independent controls executed

```bash
env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dongxi/dgx-spark-dongxi/.venv/bin/python \
  -m unittest discover -s tests -p test_native_dpo_stages.py -v
```

Working directory: `/home/dongxi/dongxi_ai/Dongxi_LLMs`. Actual result: **29 tests passed; exit 0; unittest elapsed 1.162 seconds** (shell wall time approximately 1.872 seconds).

The controls include the actual tiny CPU artifact-ledger restore/inspect/load path with worker `4`, ordered comparison witnesses, missing/wrong/mutated witness rejection, source/input drift refusal, current-run gating, exact work/I/O caps, unchanged CPU8 entry behavior, and preservation of failure receipts. Authored tiny fixtures and injected supervision in these tests are controls, not native model evidence.

## Explicit limits and corrected scope

No GPU/native jobs or full suite were launched by this reviewer. This review establishes neither CUDA compatibility nor a hashing speedup, completed native replay, or completed pilot. Actual `run-03` execution and its gates remain necessary; the older failures remain failures.

Chosen-SFT and RLVR currently use their own existing work/snapshot-I/O mechanisms rather than this DPO artifact-ledger integration. They were left unchanged: no new ArtifactBudget integration, quota, or fabricated hashing witness was introduced. The opt-in reviewed here applies only to the existing DPO ledger paths.
