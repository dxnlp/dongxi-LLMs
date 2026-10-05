# RLVR snapshot I O implementation notes

These notes retain developmental implementation checks under the predeclared
[admission protocol](../specs/2026-10-05-rlvr-snapshot-io-admission.md). They are
not final acceptance evidence. The original runner remains independently archived
in `2026-10-05-rlvr-snapshot-io-preintegration/qwen_rlvr_lab.py.txt`; its SHA256
begins `819cd8bb`. No pretrained acquisition, GPU run, installation, service or
Git mutation was performed.

## Preserved scientific boundary

The runner keeps its twenty-three logical and semantic work dimensions and its
completed/pending numerical state capture. In particular, it captures the saved
runner journal prefix **before** save-time semantic validation. The shared writer
uses exactly that saved prefix in the external receipt. Reopening the same
physical journal retains later semantic and I/O charges, so the earlier prefix
does not refill capacity. A proposal to refresh this prefix was considered and
rejected before implementation because the shared receipt already matches it.

The new optional CPU interfaces use `io_budget` on save, restore and lifecycle;
the science records `snapshot_io_contract`. The production CLI requires complete
I/O caps and a separate journal, plus an independent receipt on resume. Metadata
readers are bounded and no-follow. Checkpoint payloads are removed from both
generic input-hash passes. Shared inspection occurs only after both physical
journal prefixes bind, and its measured header is distinguished from the supplied
expectation. Caller capture, metadata/journal processing, state application,
artifact inventory and physical containment are still excluded.

## Identity role alias guard declared before testing

Before the next focused collection, add a cheap metadata-only disjointness gate
between a selected resume payload and the template, environment lock, named
identity source files, and the parent's already recognized artifact patterns.
Compare absolute paths and observed device/inode pairs without reading or hashing
payload bytes. This prevents a selected payload or a hardlink alias from hiding
in a generic identity role. It does not add broader parent traversal, charge
ordinary source/parent hashing, authenticate paths against a hostile actor, or
change training data, numerical settings or any of the twenty-three caps.
The test owner will exercise actual CLI identity spies for direct and hardlink
aliases; admission and model-provider spies must remain unused on refusal.

## Developmental test failures retained

Both commands below used the existing isolated interpreter with `PYTHONPATH=src:tests`,
hidden CUDA, Hugging Face offline flags and one CPU thread. They were ordinary
developmental unittest runs, not the test owner's final source-frozen collector.

The first command was:

```text
/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python -m unittest -q test_qwen_rlvr_recovery test_rlvr_work_budget test_run_identity test_grpo_lab
```

It exited **1**, with the actual footer `Ran 37 tests in 13.469s` and
`FAILED (errors=3)`. Two test selectors were misspelled: imports of
`test_qwen_rlvr_recovery` and `test_grpo_lab` failed. The third error was the
historical changed-cap CLI scaffold lacking `snapshot_io_limits`, producing
`AttributeError` before the intended changed-cap assertion. This was fixture
interface maintenance, not a failed training equation. The adequate local random
HF identity CLI still executed its unchanged one-update numerical path.

The second command corrected only the recovery selector:

```text
/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python -m unittest -q test_rlvr_runner_recovery test_rlvr_work_budget test_run_identity test_grpo_lab
```

It exited **1**, with `Ran 55 tests in 19.256s` and `FAILED (errors=2)`.
The nonexistent `test_grpo_lab` selector and the same absent scaffold field
remained. Neither result is relabeled as a passing acceptance panel. The final
collector must use the actual module names and root's separately declared CLI
fixture compatibility changes, then pin before/after source identities and
retain fresh-process completed/pending replay evidence.

After the declared scaffold update, the corrected development command selected
`test_rlvr_runner_recovery test_rlvr_work_budget test_run_identity test_grpo_capstone_labs`.
It exited **0**, with the actual footer `Ran 69 tests in 19.123s` and `OK`.
This passes the existing numerical and identity checks at the candidate source;
it does not replace the dedicated source-frozen I/O acceptance collector.
The runner at this command had SHA256
`477f45d71448bcdef51f0ce4605361f382ce1a74ef2e3ca56402c4c723f8260b`.
The separately declared identity-alias guard was added afterward; this historical
command is not evidence for that guard.
