# Independent preference-figure source review

Date:2026-10-05. Result: no actionable scoped defect found. This review supports
the consumer implementation and authored CPU controls, not an actual native
comparison, rendered publication figure or completed DPO recovery gate.

The inspected [consumer](../../scripts/plot_native_preference_comparison.py)
requires the fixed run01 preparation and passed common-comparison acceptance,
selected Full400/chosen100/DPO100 producer metadata, ordered4-pair populations,
actual exit0/launch/supervision joins and complete raw generation identities.
It does not accept an optimistic check dictionary in place of the original
required checks. Missing, failed, incomplete, mutated and swapped-arm fixtures
are rejected rather than displayed as successful observations.

The displayed means are absolute chosen and rejected **complete-sequence**
log probabilities on the same4 pairs. The separate margin is
`(chosen − rejected) − (Full400 chosen − Full400 rejected)`, in unscaled nats;
the retained beta0.1-scaled margin is checked but not confused with that plot.
The consumer joins the unscaled margin back to the common frozen reference.
It does not normalize these sequence scores by length or turn them into a
general model-quality ranking.

Retention is independently regraded from complete saved response records.
Location4, assistant120 and annotated reasoning20 remain separate populations;
the visible assistant and reasoning plots show counts of correctness, natural
termination and truncation, not a pooled capability score. The axis language
does not treat repeated task-template items as independent populations. No new
bootstrap is performed, and greedy64/custom-template reasoning retention is
not relabeled the original native-Instruct32/128 evaluation.

The precision labels match the inspected
[producer source](../../scripts/run_native_preference_evaluation.py): likelihood
uses FP32 policy/reference weights with BF16 CUDA autocast, original
single-shift masks and SDPA without a forced backend; generation uses
BF16-loaded CUDA policies, eager attention and the common saved-template
greedy64 contract. There is no MATH-kernel, FLOP, broad behavioral improvement
or new checkpoint-identity measurement claim. Weight inventories are explicitly
retained producer declarations, not fresh body hashes by the plotter.

The output route creates an exclusive report directory. Failed or changed-input
prefixes retain failure evidence without replacing producer records. Source,
fixed input and metadata/raw-response bindings are checked again after rendering.
The reviewed import path uses the bounded offline grader and identity helpers,
not model/tokenizer loading.

## Actual bounded verification

The independently executed command was:

```bash
env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src:tests /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python -B -m unittest discover -s tests -p 'test_native_preference_figure.py' -v
```

Actual exit:0. Actual unittest footer: `Ran 29 tests in 1.288s`, `OK`.
No skipped tests were reported. These are explicitly authored schema fixtures,
including tiny synthetic-only PNG/exclusive-output checks. Their inert producer
receipts, PIDs, scores and weight hashes are not native model results.

The independently checked source identities match the supplied review pins:

| File | SHA256 |
| --- | --- |
| `scripts/plot_native_preference_comparison.py` | `16133b68d19d48f6e5904ee3f311fceaefaaf83e7903ccc00bbc2fe17a90fd4c` |
| `tests/test_native_preference_figure.py` | `28adf20266151f99346cf985c1564add24970ae265d8017ee2cec3de3b8efdfd` |
| `scripts/run_native_preference_evaluation.py` | `1c1bb663db33c4ee6b840d5732e4236cb3a03d41aded0c7d17c61643f7939023` |

Only this new report was written. No producer/source/test changes, actual native
consumer invocation, model/weight/corpus-body hashing, model loading, GPU work,
downloads, installation, services, Git writes or wider test suite were performed.
Actual figure acceptance remains contingent on the eventual real three-arm
comparison and its unchanged-input receipt; this source review does not invent it.
