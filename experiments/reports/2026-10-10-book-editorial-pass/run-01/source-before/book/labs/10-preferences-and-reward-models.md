# Chapter 10 — Preference and Reward Laboratories

Use a Python environment with PyTorch and Matplotlib, plus Jupyter if opening
the notebooks. Mac Studio is the ordinary CPU learning lane; these checks also
run on Spark CPU. No model/data download, credential, or inference server is
needed. The feature microscope uses `reward_model_lab.py`; the collection
microscope uses `preference_audit.py`, original authored texts and explicitly
simulated judge observations. Neither is human feedback collection.

| Day | Session | Prediction and controlled change |
|---|---|---|
| 15 | [Bradley–Terry margin](../../notebooks/day-15/01_bradley_terry_margin.ipynb) | Predict signs; change score offsets, vote fraction, and margin |
| 15 | [Disagreement and calibration](../../notebooks/day-15/02_disagreement_and_calibration.ipynb) | Predict a noisy optimum; stretch margins without changing ranks |
| 15 | [Preference collection and judges](../../notebooks/day-15/03_preference_collection_and_judges.ipynb) | Swap blind displays, retain failed verdicts, perturb verbosity/injection and audit source weighting |
| 16 | [Reward shortcut audit](../../notebooks/day-16/01_reward_model_bias_audit.ipynb) | Compare confounded versus balanced nuisance features on an independent population |
| 16 | [Text reward and process labels](../../notebooks/day-16/03_text_reward_and_process_labels.ipynb) | Train token-sequence heads, move padding/EOS, audit encoding independence, inspect negative transfer/calibration and separate terminal/step supervision |

The first notebook connects scalar gradients to the shared reward model. The
second distinguishes ranking from probability quality. The third measures a
distribution intervention and retains the adversarial failure. Every exercise
has an adjacent runnable reference, labeled plots, and an interpretation. Saved
figures are reference outputs; rerunning a cell regenerates the current values.

Reproduce from the course root:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/run_preference_policy_cpu.py
PYTHONPATH=src python -m dongxi_llms.preference_audit --fixture fixtures/preference-audit/pairs.json
PYTHONPATH=src OMP_NUM_THREADS=1 python -m dongxi_llms.text_reward_lab --fixture fixtures/text-reward/records.json
python scripts/verify_course_notebooks.py --days 15 16 --kernel dgx-spark-native --export-figures
```

The runner prints a deterministic JSON evidence object; it does not overwrite
the historical report. The [predeclared specification](../../experiments/specs/2026-10-04-preference-policy-cpu.md)
and [report](../../experiments/reports/2026-10-04-preference-policy-cpu.md) describe
the actual fixture and measurement. Reading or running these material checks
does not assess learner mastery. A neural reward-model Spark extension should
start with a small independently labeled pair set, frozen source groups,
endpoint-mask tests, and an explicit rubric; no such model-scale run is claimed.

For the collection audit, the [specification](../../experiments/specs/2026-10-04-preference-audit.md)
and [measured report](../../experiments/reports/2026-10-04-preference-audit.md)
retain 484 observations, including every malformed verdict and simulated
transport failure. `--output NEW.json` exports a new full raw ledger and refuses
to overwrite old evidence. Six exact source groups, their family/sibling split
contracts and equal-source pair weights are explicit; no semantic near-duplicate
detector is claimed. Optional human or live-AI collection requires separate
authority, content/privacy terms and independently reviewed held-out labels.

The [text-model specification](../../experiments/specs/2026-10-04-text-reward.md)
and [three-seed report](../../experiments/reports/2026-10-04-text-reward.md)
retain ordinary and nuisance pair predictions, calibration bins, terminal/step
predictions and unknown-token failures. The predeclared seed 1601
[frozen preference export](../../fixtures/text-reward/frozen-preference-seed1601.json)
is a complete JSON-only historical tiny checkpoint, not
a pretrained model. `load_frozen` validates identity/shapes and offers detached
CPU `score(prompt, completion)` and `score_many` without reward gradients.
Use new `--output`/`--frozen-export` paths to retain a new run; old evidence
cannot be overwritten by the runner.

The separately [specified character arm](../../experiments/specs/2026-10-04-text-reward-vocabulary-intervention.md)
preserves the original word artifacts and runs pre-fit pair/swap/source/STEP-
prefix gates. Its [measured report](../../experiments/reports/2026-10-04-text-reward-character.md)
retains all three seeds' failed heldout predictions despite fitted training
objectives. Distinct inputs close the encoding-collision requirement, not a
robust reward-learning claim. For a new run, choose fresh output/export paths:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m dongxi_llms.char_reward_lab \
  --fixture fixtures/text-reward/records.json \
  --protocol experiments/specs/2026-10-04-text-reward-vocabulary-intervention.md \
  --output NEW_CHARACTER_RESULTS.json --frozen-export NEW_CHARACTER_REWARD.json
```

`load_char_reward` validates the fixed ASCII encoding and numeric state and
can enforce an externally expected file/interface hash. The seed 1611 export
is an imperfect detached scalar proxy, not a factual-quality promise. The
extended notebook has twelve complete code cells and eight explanatory
previews. It distinguishes exact saved-state reload from cross-version
retraining; the first historical-word equality failure remains recorded.
Fresh execution preserves source/learner cells. Use an isolated locked kernel,
not a notebook server; the new character previews are additive indices 06–08.
