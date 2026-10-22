# All course notebooks on the DGX Spark GPU kernel — 2026-10-22

**Why:** the learner decided that the DGX Spark GPU is the priority environment for course
notebooks and asked for the manifest's CPU-only rule to be replaced and every notebook
checked. Until now `docs/course_manifest.json` accepted only `execution: "cpu-offline"`, and
`scripts/verify_course_notebooks.py` hid the GPU (`CUDA_VISIBLE_DEVICES=''`) and asserted
that CUDA was absent.

## Changes

- `src/dongxi_llms/course_manifest.py`: three declared execution types giving each
  notebook's minimum runtime: `cpu-offline` (any kernel, no network or model caches),
  `local-model-cache` (any kernel, needs cached Hugging Face files) and `spark-gpu` (needs
  CUDA on the Spark). `select_notebooks(..., execution=...)` filters by type.
- `scripts/verify_course_notebooks.py`: CUDA visible by default (`--cuda show`), with
  `--cuda hide` to reproduce older CPU-only runs. The preamble asserts CUDA only for
  `spark-gpu` notebooks. New `--execution` filter, and each record stores the execution type.
- Manifest: `day-11/05_qwen_chat_template_and_labels` is `local-model-cache`; all others
  remain `cpu-offline`. No notebook was `spark-gpu` at the time of this check; the first, `day-12/03_instruction_tuning_end_to_end`, was added later the same day and passes on the Spark GPU kernel (see the `2026-10-22-spark-gpu-notebook-fixes` run).
- Tests: new cases in `tests/test_course_manifest.py` and `tests/test_course_verification.py`.

## Run

Command (from repository root, base commit `fdf3f91` plus the uncommitted changes above):

```sh
PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/verify_course_notebooks.py \
  --kernel dgx-spark-native --cuda show --timeout 600 --output <new empty directory>
```

Kernel identity: `dgx-spark-native`, CUDA available, torch `2.13.0+cu130`, transformers
`5.16.1`. 84 registered notebooks (83 `cpu-offline`, 1 `local-model-cache`) in 3.4 minutes.
Executed outputs were kept in a session scratch directory, not committed.

**Result: 81 passed, 3 failed** (552 executed code cells and 223 figures among the passes).

| Notebook | Failure | Cause | Passes with `--cuda hide`? |
|---|---|---|---|
| `day-18/02_dpo_retention_and_preference_controls` | `assert not torch.cuda.is_available(), "Use the CPU teaching kernel"` | Hard-coded CPU-only assertion; a visible GPU is the only problem | Yes |
| `day-16/03_text_reward_and_process_labels` | exact `torch.equal` between a live character reward and the frozen export | Bit-exact replay of a state saved under the course lock's torch 2.14.1; the Spark env has 2.13.0 | No |
| `day-26/05_student_prefix_and_teacher_context` | fresh 30-update `final_state_sha256` differs from the stored campaign hash | Same: bit-exact training replay across torch versions | No |

The two version-sensitive failures are consistent with Chapter 10's existing note that "fresh
retraining under a newer Torch version is a different reproducibility question". They were not
rerun in the locked teaching environment here (`uv sync --extra course` was not performed), so
"passes under torch 2.14.1" comes from the earlier recorded CPU verifications, not this run.

## Resolution: three declared revisions

The learner approved the fixes. Each one was measured before editing.

| Notebook | Measured on the Spark (torch 2.13) | Revision |
|---|---|---|
| `day-18/02` | Passes with the GPU hidden; only the assertion fails | Replace the assertion with `torch.set_default_device("cpu")` and print the device and GPU visibility |
| `day-16/03` | Fresh vs. frozen character scores differ by at most 6.7e-11 (relative ~1e-11) in float64; reloading the saved state is still exact; the word-model delta is 0.0 | `torch.testing.assert_close(atol=1e-9, rtol=1e-9)` for fresh vs. frozen; report bit-exactness and max difference; new exact same-state reload assertion |
| `day-26/05` | Over all 30 updates, sampled states, wrong-prefix counts, unique state keys and audits are identical; losses differ by ≤2.9e-12 and gradient norms by ≤3.9e-10; only wall-clock `teacher_query_seconds` and hashes that include weights differ | Exact equality for `state_count`, `wrong_prefix_states`, `unique_state_keys`; `np.testing.assert_allclose(rtol=1e-8, atol=1e-9)` for losses and gradient norms; report the final weight-hash comparison |

No markdown, cell order or cell IDs changed. The changes are limited to one code cell per notebook.

**Preservation baseline.** `scripts/check_notebook_contracts.py` gained an explicit
*declared revision* for tracked notebooks. The baseline entry still verifies the Git
original's file hash at base commit `7f5175b`, and adds `declared_revision` = {frozen revised
file, its SHA-256, reason, date}. The Git original remains the authority for the ledger's
Markdown inventory and prompt indices; the revision replaces only the code (and notebook
header) reference. Run: `experiments/reports/2026-10-22-spark-gpu-notebook-fixes/run-01/`,
which carries the previous ledger forward and re-binds the two affected reference pairs. New
tests cover an undeclared change failing, a declared one passing, a tampered revision file and
a missing reason.

## Final run after the revisions

Same command, kernel `dgx-spark-native`, CUDA visible, torch `2.13.0+cu130`:
**84 passed, 0 failed** in 3.7 minutes (581 executed code cells, 243 figures).
