# Original text reward fixtures

These original templated texts and labels are course-authored, not human
feedback, AI judgments or an imported benchmark. Twelve training color-source
groups, four calibration groups and four test groups are assigned before pairs
are formed. The vocabulary is fitted only on training text; unknown held-out
words map to an explicit unknown token under the saved contract. The experiment
reports those counts rather than silently truncating or pretending the words
were known. A fixed unlabeled formatting alphabet supplies punctuation/headings
without preference labels. The measured encoding audit finds that unseen
“box” and “book” both become `<unk>`, so all four baseline test pairs duplicate
calibration encodings despite disjoint raw source groups. The original result
is retained as a representation-boundary failure, not independent calibration
evidence.

The terminal/process fixture has nineteen traces across five source groups.
It intentionally crosses final-answer correctness with arithmetic-step validity:
a correct final answer can follow a false equation, and a wrong final answer can
follow a valid equation. Three training traces have two steps. Step-boundary
labels concern the displayed equality, not future return or human confidence.

Matched-length/plain-format test pairs are the baseline. The module constructs
matched-format headings, longer incorrect answers and equal-substance nuisance
variants from the same held-out groups. Exact variants and all predictions are
retained; they do not become additional independent source questions.

This fixture is a mechanism microscope. It cannot establish general arithmetic
reasoning, robust reward-model transfer or calibrated human preferences. The
earlier explicit-feature shortcut experiment remains a separate intact lesson.
