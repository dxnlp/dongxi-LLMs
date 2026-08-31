# Experiment Report: Transparent Embedding and Masking Gradient Paths

## Identity

- Experiment ID: `2026-08-31-embedding-gradient-paths`
- Specification: `experiments/specs/2026-08-31-embedding-gradient-paths.yaml`
- Date: 2026-08-31
- Course code commit: `4d747f536ffe19bfbd0d50aa94e9e3f60594dbe0`
- Machine manifest: `manifests/2026-08-29-dgx-spark-native.json`
- Environment lock: `/home/dongxi/dgx-spark-dongxi/uv.lock`, SHA-256
  `ca1e48e9cc3a73f4a37f425181e94248a4abfb2bd0141125337bd78749cb6efa`
- Python: 3.12.14
- PyTorch: `2.13.0+cu130`
- Device and dtype: CPU, float32
- Exact verification command:
  `PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m unittest tests.test_embedding_gradient_lab`
- Exact experiment command:
  `PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m dongxi_llms.embedding_gradient_lab`
- Exit code: 0 for both commands

## Protocol

- Intended protocol: run three fixed, transparent PyTorch demonstrations using
  seed 7, vocabulary size 6, embedding width 4, and a `1e-9` threshold for
  classifying a row gradient as nonzero.
- The repeated-lookup demonstration uses an exact sum loss so each occurrence
  contributes a vector of ones.
- The classifier comparison uses identical initial numeric values for tied and
  untied output weights; only parameter sharing differs.
- The masking demonstration uses one-head causal self-attention with identity
  query, key, and value projections. Positions 0 and 1 are prompt positions;
  only position 2 contributes to the optimized loss.
- Deviations from specification: none.

## Measurements

| Measurement | Value |
|---|---|
| Unit tests | 3 passed in 0.131 s |
| Lookup input shape | `[1, 3]` |
| Lookup output shape | `[1, 3, 4]` |
| Repeated row 2 gradient | `[2, 2, 2, 2]` |
| Single-use row 5 gradient | `[1, 1, 1, 1]` |
| Untied nonzero input-embedding rows | `[1, 3]` |
| Tied nonzero embedding rows | `[0, 1, 2, 3, 4, 5]` |
| Untied and tied forward loss | `0.699771` in both cases |
| Response-only loss mask | `[0, 0, 1]` |
| Response attention over positions 0, 1, 2 | `[0.129963, 0.227205, 0.642832]` |
| Prompt row 1 gradient norm | `0.303322` |
| Prompt row 2 gradient norm | `0.668485` |
| Optimized response loss | `4.622450` |

Memory use and throughput were not measured because this CPU-scale numerical
experiment was designed to expose graph structure, not characterize performance.

## Representative outputs

For the tied-versus-untied comparison, the input rows were `[1, 3]` and the
target row was 4. The input embedding gradient norms were:

```text
untied: [0, 0.345023, 0, 0.345023, 0, 0]
tied:   [0.045018, 0.294493, 0.053737, 0.364069, 0.644414, 0.003910]
```

For response-only supervision, all three per-position losses were mechanically
computed as `[2.776799, 0.601840, 4.622450]`, but the mask `[0, 0, 1]` made only
the final value part of the optimized scalar loss. The prompt rows nevertheless
had nonzero gradients.

## Observations

- All precommitted success criteria passed.
- The two occurrences of ID 2 accumulated exactly twice the gradient of the
  single occurrence of ID 5 under the controlled sum loss.
- Untied and tied classifiers produced the same logits and scalar loss from the
  matched initial numeric weights, but produced different gradients for the
  input embedding parameter.
- The untied embedding table had nonzero gradients only at selected input rows.
- Reusing the embedding parameter as the output classifier produced nonzero
  gradients at all six vocabulary rows in this dense-softmax example.
- A zero prompt loss mask did not prevent prompt embedding gradients when the
  supervised response state attended to those prompt positions.

## Interpretations

- Repeated lookup is parameter reuse: position-level contributions add into one
  shared embedding row.
- Weight tying changes the computation graph and gradient routing even when the
  forward numeric values are initially identical. Parameter identity therefore
  matters in addition to the values stored in a tensor.
- A loss mask controls where loss terms originate; it does not detach visible
  prompt computations that influence a supervised response.
- The measurements support the analytical gradient-path model recorded in the
  Day 2 learning artifacts.

## Claims not established

- This experiment does not establish semantic quality, convergence, or model
  quality improvement.
- It does not establish exact gradient magnitudes for Qwen3.
- It does not establish how every trainer API combines attention and loss masks.
- It does not establish that tied weights are universally better than untied
  weights.
- It does not measure GPU performance or memory safety.

## Failures or surprises

- No success criterion failed.
- The equal forward losses but unequal embedding gradients provide a useful
  illustration that numerically equal tensors are not necessarily the same
  parameter in the computation graph.

## Decision and next experiment

- Decision: pass. The Day 2 embedding lookup, tying, and response-only masking
  mechanisms now have executable evidence.
- Exact next action: implement and report the pinned Qwen3 multilingual
  tokenizer exploration, including exact round trips and efficiency ratios.

