# Chapters 13–15 material verification

Prepared on 2026-10-04 in the course's Spark workspace. This verifies course
mechanisms and executable companions, not learner mastery or Qwen capability.

## Inventory

- Three coherent chapters, about 2000 words each, with independent derivations,
  concrete failures, measured CPU evidence and primary-source links.
- Thirty worked conceptual answers and three lab routes.
- Eleven notebooks across every Day22–28:33 code cells,12 explanatory PNG outputs.
  Every exercise has an adjacent reference explanation/solution and controlled
  change. Saved previews link to exported figures; executable source regenerates
  them from current arrays. Reference previews are labeled fixed results.
- Four reusable modules: `grpo_lab`, `optimization_diagnostics_lab`,
  `distillation_lab`, and the optional offline `qwen_rlvr_lab` adapter.
- Original arithmetic/verifier fixtures, three bounded mechanism experiments,
  one selection simulation, specifications/config and raw JSON/Markdown reports.
- Seven day indexes and topic artifacts. Progress is prepared material, with live
  learner explanations and model-scale execution recorded separately.

## Checks actually executed

```bash
PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m unittest discover -s tests -p test_grpo_capstone_labs.py -v
/home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/verify_course_notebooks.py --days 22 23 24 25 26 27 28 --kernel dgx-spark-native --export-figures
/home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/run_grpo_capstone_experiments.py
PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m dongxi_llms.qwen_rlvr_lab --help
python3 scripts/check_book_math.py
git diff --check
```

All 15 gradient/invariant cases passed: population std/constant groups, clipping
signs, pad mask/empty response, exact-KL derivative, adversarial verifier,
actual decoder rollout/alignment/gradient boundaries, HF-style model adapter,
EOS/context guards, chat/SFT template hash/stops, frozen greedy panel, reward
exploit/repair, reduction/version contracts, distillation temperature gradient,
genealogy/cycles/panel identity and release evidence. The final file-fixture test
forces a failure after one completed update, then checks retained baseline rows,
initial panel, JSONL/update records, failed status, exception and active stage.
The test also rejects reusing that run's output directory. No HF model is loaded
by this test.

Final adapter SHA256: `35a0c645f6770c509eee291af5567b4c6500b02102e0bd0ceeac1a9a4a13470d`.
Final test-file SHA256: `d07d5bfef9d2b155e46c04811546431dfb111130bec579466b0fad364a4d3230`.

The notebook subset passed all11 sessions/33 code cells, exporting12 figures.
Executed copies/manifest were written under a temporary verification directory;
source notebooks were preserved. A later source edit added only saved-preview
Markdown cells. The whole-course final report provides the current source-hash
snapshot. Decoder workflow, genealogy and selection figures were inspected at
full size and readable. Additional scientific figures should be inspected when
their controls/data are materially changed.

Book math check at this checkpoint:50 Markdown files,1053 expressions,zero
issues. This is conservative source validation, not a claim of live GitHub
browser rendering. The root course audit may include more files added afterward.

## Actual mechanism evidence

[Raw bounded evidence](2026-10-04-grpo-diagnostics-distillation.json) and
[interpretation](2026-10-04-grpo-diagnostics-distillation.md):

- GRPO executes a real one-layer TinyDecoder rollout/verify/backward/update at
  G4/G8. The held-out result is negative:greedy accuracy remains0/4; no gain is
  invented. G8 has a larger response-token budget.
- The deliberately flawed proxy rises while strict correctness falls; a
  strict-reward restart succeeds in the three-action task.
- Distribution distillation fits a fixed three-logit teacher using80 SGD steps.
- Candidate selection is a labeled categorical Monte Carlo simulation; candidate
  token costs are illustrative. System throughput budgets are projections.

## Model-scale readiness and remaining empirical work

The optional local HF runner has bounded rollout/update and initial/final frozen
greedy evaluation, recorded prompt/template/stopping identity, memory/time guards,
full model/tokenizer export and parent/source/evaluation hashes. Tiny HF-style
tests validate its reusable path. No Qwen weights were loaded and no CUDA job
was executed. The supplied upstream SHA is metadata; actual local input file
hashes identify bytes. A four-prompt panel is a descriptive pilot and does not
establish broad reasoning quality.

The validated output directory now exists before any load. Atomic status/report
snapshots and per-update JSONL preserve partial evidence if a load, resource guard
or nonfinite failure interrupts the run. Process termination outside Python's
exception handler can leave `running` status; process exit must still be checked.

The final state includes RNG/optimizer/update metadata, but an exact-resume CLI
and replay equivalence have not been implemented or claimed. A vLLM/Open Instruct
adapter comparison remains a separately tested extension. Any actual model run
needs its own process exit, environment lock, peak memory, outputs and independent
capability evidence under the frozen contract.

No checkpoint/dataset download, notebook server, animation production, external
publishing, Git commit or push was performed by this scoped material build.
