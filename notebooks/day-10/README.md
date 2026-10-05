# Day 10 — Evaluation Is a Contract

These are complete CPU lessons for Chapter 7. Material readiness is separate from learner practice and real-model evidence.

- [What does this score actually mean?](01_metrics_and_contracts.ipynb).
- [Is the measured improvement resolved?](02_paired_uncertainty.ipynb).
- [Can an average conceal the important failure?](03_slices_and_contamination.ipynb).
- [Can the grader understand the answer without changing the task?](04_mathematical_grading_and_response_replay.ipynb).
- [Can a selector find a correct answer that the model generated?](05_actual_candidates_and_answer_selection.ipynb).

Each notebook contains predictions, adjacent explanations and runnable solutions, and plots from the actual lesson tensors or declared illustrative fixtures. Use the course kernel with PyTorch and Matplotlib. No downloads or GPU work are required.

Reader route: [chapter](../../book/chapters/07-evaluation-is-a-contract.md), [solutions](../../book/solutions/07-evaluation-is-a-contract.md), [lab](../../book/labs/07-evaluation-is-a-contract.md).

The fourth session adds bounded exact mathematical grading and complete
saved-response replay. Its two response panels are authored fixtures, not real
model checkpoints. It preserves unsupported/ambiguous answers, unknown costs,
errors, truncation and deliberately planted capability/interface regressions.
The separate generation adapter and actual pretrained evaluation remain pending.

The fifth session adds actual original tiny-decoder sampling, fixed train-only
fits and gold-blind answer selection. Its seven reference code cells and five
scientific figures distinguish any-correct availability from first/vote/likelihood
decisions, charge every invalid attempt and whole-attempt token overshoot, and
inspect source/template/family failures plus error dependence. It has no English
parser or pretrained capability claim; the full measured panel remains immutable.
