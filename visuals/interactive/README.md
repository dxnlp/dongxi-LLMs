# Live candidate-selection teaching pilot

The [selection fragment](selection-budget.html) lets the learner change the
candidate prefix and gold-blind rule while watching available correctness,
delivered answer and generated/rescored tokens change. It uses the actual
seed10052, final-policy `train-11` pool, not simulated candidate probabilities.
At eight samples, three correct0s lose to five wrong1s. At three, the majority
returns0; a correct pool does not guarantee a monotonic selector.

Canonical prose: Chapter7 and Chapter15.2; complete underlying
[measurement](../../experiments/reports/2026-10-04-inference-selection.md).
The fragment makes no new model calls, network requests or paid API calls.
Gold is confined to displayed evaluation; selectors receive candidate views.
The recorded rescoring is charged to every method and is not optimized serving.

The [local browser verification](../../experiments/reports/2026-10-04-selection-visual/verification.json)
checks24 candidate/rule combinations at each of320/736 pixels, real input/change
events, state restoration, invalid-state fallback and visible horizontal bounds.
Screenshots were inspected at both widths. This validates local CPU-browser
behavior, not a pass on the actual inline host, Mac or learner understanding.
No browser/environment was installed for the check.

To reproduce with an already installed CPU browser and the current host's
visualization stylesheet, supply explicit paths and an unused output directory:

~~~bash
python scripts/verify_selection_visual.py \
  --fragment visuals/interactive/selection-budget.html \
  --stylesheet /absolute/path/to/visualize.css \
  --browser /absolute/path/to/headless_shell \
  --output /absolute/path/to/new-proof-directory
~~~

The direct conversation must embed this fragment through its supported visual
content reference. Git carries the original source, not browser sessions or
host appearance. No external website/export/publication is implied. The
larger Day8 optimizer playground remains a separate unimplemented idea.

The [October 5 delivery check](../../experiments/reports/2026-10-05-selection-visual-delivery/run-01/verification.json)
repeats all 48 local-browser control combinations at 320/736 pixels against the
unchanged fragment and original measurement bytes. Both screenshots were
inspected. The audit continuation supplies the supported inline reference in
its response; delivery is distinct from observing the actual app host render,
user interaction or learner understanding. Those are not measured by this file.
