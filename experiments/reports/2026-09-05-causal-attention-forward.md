# Causal attention forward verification

- Date: 2026-09-05
- Specification: `experiments/specs/2026-09-05-causal-attention-forward.md`
- Executed source commit: `b62c20e`
- Environment: platform interpreter
  `/home/dongxi/dgx-spark-dongxi/.venv/bin/python`, PyTorch `2.13.0+cu130`,
  CPU float64. Fixed teaching inputs; random-test seed 4. No training or GPU use.

## Observations

- Three focused unittest cases passed (exit code 0). They check SDPA agreement
  for three sequence/head-width configurations, causal invariants, two broken
  masking methods, prefix truncation, and renormalization with underflow.
- nbformat validation and fresh-kernel execution passed: all 13 code cells,
  kernel `dgx-spark-native`, 60-second timeout per cell, notebook directory as
  working directory. Execution used nbclient; source notebook retains empty
  outputs and unfilled learner predictions. Validation does not mark the learner
  checkpoints complete.
- Notebook manual output versus SDPA: maximum absolute error `0.0` on its
  teaching inputs. Assertions use absolute and relative tolerance `1e-12`.
- Changing only the fourth input row preserved the first three outputs for
  correct attention. Maximum differences by position, rounded to five decimals:
  correct `[0, 0, 0, 18.43930]`; naive post-softmax masking
  `[0.57173, 0, 0.99875, 18.43930]`; zero-score replacement
  `[3.77875, 3.77875, 2.07209, 18.43930]`.
- Scores `[0,0,10]` with only the first two positions allowed: correct weights
  `[0.5,0.5,0]`; naive post-mask row sum `9.079161565902182e-05`;
  zero-score replacement weights `[1/3,1/3,1/3]`.
- Post-mask renormalization agreed with correct masking for future score 10.
  At future score 10000 the allowed weights underflowed to zero and the
  deliberate repair returned NaNs; masking before softmax stayed finite.

## Interpretation and boundary

All specified readiness criteria passed. The counterexamples demonstrate why
zero future weights alone, or row sums alone, do not establish causal behavior.
The unchanged second output in the naive post-mask example also illustrates
that leakage need not affect every position for every perturbation.

This verifies tiny forward computations, not learned quality, gradient paths,
KV-cache equivalence, or serving speed. Day 4 remains in progress. Chapter 4
prose and worked solutions will integrate this evidence during Day 4 synthesis
after the remaining gradient and cache sessions.

## Reproduction

Run from the repository root:

```bash
PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m unittest discover -s tests -p test_causal_attention_lab.py -v
```

Execute `notebooks/day-04/01_causal_attention_forward.ipynb` in a fresh
`dgx-spark-native` kernel. Leave `run_my_implementation=False` to validate the
reference path. The NaNs in the optional underflow demonstration are intentional
and asserted; they are not a failed notebook execution.
