# Experiment Report: Repeated One-Hot Targets Learn a Distribution

## Identity

- Experiment ID: `2026-09-03-next-token-distribution`
- Specification:
  `experiments/specs/2026-09-03-next-token-distribution.yaml`
- Date: 2026-09-03
- Pre-execution specification commit:
  `c9c8bbfde796e5e359d350579d5e8029594d42ba`
- Executed code commit: `c27bb398261eb958993edfda774345bc071457d7`
- Machine manifest: `manifests/2026-08-29-dgx-spark-native.json`
- Environment lock: `/home/dongxi/dgx-spark-dongxi/uv.lock`, SHA-256
  `c179a5db74b36f6e417b231cef334123e24802086391f9296cbaf776ea76c261`
- Python: 3.12.14
- PyTorch: `2.13.0+cu130`
- Device and dtype: CPU, float32
- Exact verification command:
  `env PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m unittest tests.test_next_token_distribution_lab`
- Exact experiment command:
  `env PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m dongxi_llms.next_token_distribution_lab --steps 500 --learning-rate 0.5 --output experiments/outputs/2026-09-03-next-token-distribution.json`
- Exit code: 0 for both commands

## Protocol

- Use one abstract context and two trainable float32 logits initialized to zero.
- Present all 100 targets on every step: 70 class-0 targets and 30 class-1
  targets.
- Optimize mean cross-entropy with full-batch SGD, learning rate 0.5, for 500
  steps.
- At fixed checkpoints, compare PyTorch's autograd gradient with the analytical
  gradient `p-r`, where `r=[0.7,0.3]`.
- Deviations from final specification: none. An import-path omission was found
  by a non-experimental environment check and corrected in commit `c9c8bbf`
  before implementation or execution.

## Measurements

| Measurement | Value |
|---|---:|
| Unit tests | 3 passed in 0.692 s |
| Initial probabilities | `[0.5, 0.5]` |
| Final probabilities | `[0.69999999, 0.30000004]` |
| Initial mean cross-entropy | `0.69314730` |
| Final mean cross-entropy | `0.61086428` |
| Empirical target entropy | `0.61086434` |
| Maximum recorded gradient disagreement | `5.96e-08` |
| Final gradient norm | `7.45e-09` |
| Final logits | `[0.42364874, -0.42364898]` |
| Final logit gap | `0.84729773` |
| Precommitted criteria | 7 of 7 passed |

Runtime and memory were not independently measured because the specification
explicitly excludes hardware-performance and memory-safety claims.

## Representative outputs

```text
step 0:   p=[0.5000000, 0.5000000], loss=0.6931473,
          grad=[-0.2000000, 0.2000000]
step 10:  p=[0.6854001, 0.3145998], loss=0.6113629,
          grad=[-0.0145999, 0.0145998]
step 50:  p=[0.6999989, 0.3000011], loss=0.6108643,
          grad=[-1.12e-06, 1.12e-06]
step 500: p=[0.7000000, 0.3000000], loss=0.6108643,
          grad=[0.0, 7.45e-09]
```

The complete JSON output is a local ignored artifact at
`experiments/outputs/2026-09-03-next-token-distribution.json`.

## Observations

- The process and all three unit tests exited successfully.
- Every recorded numeric value was finite.
- Probabilities moved from the uniform initialization to the precommitted 70/30
  target within tolerance.
- Mean loss decreased and matched empirical target entropy within `1e-4`.
- Autograd and `p-r` agreed at every recorded checkpoint; the largest absolute
  disagreement was below the precommitted `1e-6` threshold.
- The gradient norm approached zero as the predicted and empirical
  distributions converged.
- The final logit difference was approximately `0.8472977`.

## Interpretations

- Repeated one-hot targets can collectively teach a non-one-hot distribution.
- The observed gradient equilibrium supports the analytical condition `p=r`.
- Only relative logits are identified: the final gap agrees with
  `log(0.7/0.3)`, while adding a shared constant would preserve the prediction.
- The limiting cross-entropy equals empirical entropy because the controlled
  model has eliminated its KL mismatch to the empirical distribution without
  eliminating that distribution's intrinsic uncertainty.

## Claims not established

- The experiment does not establish generalization to unseen contexts or real
  language.
- It does not test a transformer, attention, embeddings, or a tokenizer.
- It does not establish language understanding, factuality, or generation
  quality.
- It does not establish calibration under dataset or decoding shift.
- It does not measure GPU performance, throughput, or memory safety.

## Failures or surprises

- No precommitted criterion failed.
- No protocol deviation occurred after the final specification was committed.

## Decision and next experiment

- Decision: pass. The result supports the bounded hypothesis that full-batch
  cross-entropy on this controlled two-logit system recovers its empirical
  target frequencies and produces the analytical `p-r` gradient.
- Exact next action: incorporate the verified mechanism into the Day 3 chapter
  and solutions, then keep any larger natural-language generalization claim for
  a separately specified experiment.
