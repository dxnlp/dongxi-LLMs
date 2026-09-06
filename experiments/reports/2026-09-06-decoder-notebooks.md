# Chapter 5 — Notebook Reference Verification

## Result

All eleven new notebooks passed fresh-kernel execution: **93 code cells**, with
adjacent runnable reference solutions and no execution errors. All **38 repository
unit tests** passed, including 12 new decoder tests. Both verification commands
exited with code 0. The fixed one-batch learning experiment met its predeclared
loss, accuracy, and finite-gradient criteria.

- [Pre-run specification](../specs/2026-09-06-decoder-notebooks.md).
- [Machine-readable manifest and full loss history](2026-09-06-decoder-notebooks.json).
- [Notebook pathway](../../notebooks/day-05/README.md).
- [Reader-facing companion lab](../../book/labs/05-building-a-modern-decoder.md).

## Execution identity

- Final reference pass: 2026-09-06, starting 11:40:10 UTC.
- Source base: `6d62d6462992cd4c9a485ed485511b4bf7b9689d` plus the newly authored
  working-tree source. Exact notebook, implementation, and test SHA-256 hashes
  are recorded in the JSON manifest, not inferred from the base commit alone.
- Python 3.12.14; PyTorch 2.13.0+cu130; nbformat 5.11.1; nbclient 0.11.0.
- Linux aarch64; `dgx-spark-native` kernel; all tensors used CPU, one Torch thread.
- Initial host MemAvailable: 124170960 kB. This is an initial observation, not a
  measured minimum or peak process memory claim. CUDA was not used.
- Eleven notebook kernels were started independently and shut down after their
  reference paths. No JupyterLab server was started.
- Final verification runner elapsed time: 25.57 seconds, including kernel startup
  and a separate repeat of the tiny training fixture. This is not model latency.

## Coverage

| Session | Code cells | Verified mechanism |
|---|---:|---|
| 01 Embeddings/positions | 9 | Lookup identity, repeated-row gradients, position intervention |
| 02 Multi-head attention | 7 | Looped/vectorized forward and input gradients, head ablation, causal prefix |
| 03 Residual stream | 9 | Zero-branch identity, gradient sum, cancellation, pre-norm block identity |
| 04 LayerNorm/placement | 9 | Library forward/backward, wrong-axis leakage, constants, pre/post-norm controls |
| 05 Positionwise MLP | 7 | Explicit forward, affine collapse, positionwise influence and gradients |
| 06 Decoder assembly | 9 | Shapes, initialization, tied parameters, gradients, full causality/cache replay, broken labels |
| 07 One-batch learning | 7 | Fixed-budget fitting, frozen control, predictions and evidence limits |
| 08 RMSNorm/SwiGLU | 9 | Norm reference derivatives, gating, parameter budget, isolated replacements |
| 09 RoPE | 9 | Pair norms, numerical gradcheck, relative-position identity, cache offsets |
| 10 GQA/QK norm/costs | 9 | Independent reference gradients, MHA/MQA endpoints, QK sensitivity, compact caches/counts |
| 11 Fixed recurrence | 9 | Shared/copy forward identity, gradient accumulation, storage versus applications, distinct KV |

Float64 reference comparisons use atol 1e-10, rtol 1e-8 unless a cell explicitly
states otherwise. Numerical gradcheck uses its declared library defaults.
The suite separately verifies chunked cache replay, future input gradients,
analytical counts across tied/untied and MHA/GQA/MQA configurations, and invalid
configuration rejection.

## Fixed one-batch learning

| Quantity | Observation |
|---|---:|
| Seed | 505 |
| Optimizer updates | 160 |
| AdamW learning rate / weight decay | 0.02 / 0 |
| Initial cross-entropy | 2.774792432785034 |
| Final cross-entropy, after final update | 0.0007856183219701052 |
| Final training-token accuracy | 1.0 |
| Nonfinite loss/gradient checks | Passed throughout |

The dataset is exactly the two seven-token integer sequences in
`teaching_batch()`, shifted to 2x6 inputs and labels. Model parameters, dataset,
seed, and step budget were not selected using a held-out set. The initial frozen
control remained unchanged. The reversed-context probe has no declared gold
labels; it is not scored as a generalization evaluation. The loss history is
stored in the manifest; history entries precede their updates, whereas final
loss follows update 160.

## Accounting checks, not serving claims

The modern fixture uses B=2, T=6, L=2, D=16, Hq=4, d=4, float64, tied output
embeddings, and the lab's per-head QK RMSNorm. Analytical values matched stored
parameters and actual compact cache tensor payloads:

| KV heads | Stored parameters | Logical cache bytes | Estimated dense forward matmul FLOPs |
|---:|---:|---:|---:|
| 4 | 5472 | 6144 | 138240 |
| 2 | 4960 | 3072 | 125952 |
| 1 | 4704 | 1536 | 119808 |

FLOPs count multiply-add as two and include all-position vocabulary projection;
they exclude softmax, norms, activations, backward, allocator overhead, and other
operations. The transparent grouped implementation expands K/V temporarily for
arithmetic. Logical cache reductions are not measured latency speedups.

Three accounting-only larger design exercises have 50,480,512; 100,587,008; and
152,508,544 parameters under the exact configs in session 10. These models were
not instantiated, trained, or benchmarked.

## Failures, changes, and limits

No notebook execution or unit-test criterion failed. After the first successful
pass, source metadata was clarified and accounting-only configurations were
adjusted to approximately 50M/100M/150M parameters; the complete notebook suite
was rerun successfully. The training contract was unchanged. Local Jupyter
emitted a TCP-encryption warning; these were temporary local kernels, not an
externally exposed notebook service, and no authentication material is recorded.

Source notebooks retain clean, empty outputs for prediction-before-reveal study.
The verifier writes executed copies to a newly created temporary directory;
reproduction does not overwrite the sources or learner predictions. Existing
Day 3 and Day 4 learner notebook edits were not executed, modified, or staged.

These checks establish mechanism correctness within tested fixtures and bounded
memorization, not learner mastery, real-language generalization, modern-variant
quality superiority, long-context extrapolation, serving performance, or Mac
environment compatibility. The optional recurrence lesson tests fixed sharing,
not adaptive routing or a trained recurrent-model comparison. No animation was
rendered and no public article was published.

## Reproduce

From the repository root, use the existing platform environment:

```bash
/home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/verify_decoder_notebooks.py
PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m unittest discover -s tests -v
```

For interactive study, select `Python (DGX Spark Native)` and run setup once,
then attempt each exercise before its adjacent reference solution. Another
machine can use its own Torch-enabled Python kernel; verify it there before
claiming cross-machine numerical agreement.
