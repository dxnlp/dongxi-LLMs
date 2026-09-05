# Day 4 gradient, scaling, and cache verification

- Mode: smoke; CPU float64; synthetic inputs, no training or model downloads.
- Freeze source and this specification in Git before execution. All expectations
  below are predictions, not results.
- Gradient fixture: existing teaching_inputs; one fixed three-class linear head;
  cross-entropy target 1 at position 2. Position 3 is forbidden.
- Require manual chain-rule gradients for Q/K/V/A/scores/X and W_Q/W_K/W_V to
  agree with autograd within 1e-12 max absolute error. Require central differences
  for every scalar projection weight (epsilon 1e-6) within 1e-7.
- Require zero forbidden-score and future-input gradients, nonzero earlier-input
  gradients, and branch-specific absence of parameter gradients when A or V is
  detached. No optimizer run is claimed.
- Scaling: seed 40, IID normal coordinates, 256 independent queries and 16 keys,
  widths 8/64/512; paired scaled/unscaled draws. Predict scaled standard deviation
  within .15 of 1, higher scaled entropy at each width, and higher average
  softmax-Jacobian trace at width 512. No universal trained-model claim.
- Cache: seed 41, two single-head residual attention layers, width 4, six feature
  rows, output width 5, no positions, norm, MLP, or dropout. Replay fixed rows;
  this is not free generation. Prefill lengths 1/2/4/6; require every incremental
  output versus full prefix within 1e-12. Logical final K/V size: 768 bytes.
- Changing the first input while holding the second identical must leave its
  first-layer K identical but change its second-layer K by >1e-6; stale-cache
  output must differ from correct recomputation by >1e-6.
- Count projected rows analytically; do not infer measured speedup or peak RAM.
- Notebook acceptance: validate schema and execute reference cells in fresh
  dgx-spark-native kernels. Preserve current learner edits in sessions 1 and Day 3.
- Acceptance command: PYTHONPATH=src platform-python -m unittest discover -s tests.
- Failure: any required assertion or new notebook cell error blocks readiness.
