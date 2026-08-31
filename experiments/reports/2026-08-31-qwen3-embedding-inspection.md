# Experiment Report: Pinned Qwen3 Embedding Interface Inspection

## Identity

- Experiment ID: `2026-08-31-qwen3-embedding-inspection`
- Specification: `experiments/specs/2026-08-31-qwen3-embedding-inspection.yaml`
- Execution date: 2026-08-31
- Course code commit: `684f9368fc94a5deff4d6319cdbb73c8841a057c`
- Model: `Qwen/Qwen3-0.6B`
- Revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Machine manifest: `manifests/2026-08-29-dgx-spark-native.json`
- Environment lock: `/home/dongxi/dgx-spark-dongxi/uv.lock`, SHA-256
  `ca1e48e9cc3a73f4a37f425181e94248a4abfb2bd0141125337bd78749cb6efa`
- Python: 3.12.14
- PyTorch: `2.13.0+cu130`
- Transformers: 5.16.1
- Device: CPU
- Exact command:
  `PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m dongxi_llms.qwen_embedding_inspection`
- Exit code: 0

## Protocol

- This reproducibility protocol followed an exploratory read-only inspection and
  is not presented as a precommitted hypothesis test.
- Load the exact tokenizer and causal language model from the local cache using
  checkpoint-default dtype.
- Record tokenizer boundaries, configured dimensions, runtime input/output
  parameter identity, and a one-token forward-logit shape.
- Deviations from specification: none.

## Measurements

| Measurement | Value |
|---|---:|
| Tokenizer base vocabulary | 151,643 |
| Tokenizer total entries | 151,669 |
| Maximum assigned tokenizer ID | 151,668 |
| Model vocabulary rows | 151,936 |
| Hidden width | 1,024 |
| Input embedding shape | `[151936, 1024]` |
| Output embedding shape | `[151936, 1024]` |
| Runtime parameter object identical | yes |
| Runtime storage pointer identical | yes |
| One-token logits shape | `[1, 1, 151936]` |
| All one-token logits finite | yes |
| Model rows beyond tokenizer entries | 267 |
| Model vocabulary divisible by 128 | yes |

Tokenizer boundary lookup:

```text
ID 151668 → </think>
ID 151669 → no tokenizer piece
ID 151935 → no tokenizer piece
```

## Observations

- The pinned tokenizer and model loaded successfully and the forward pass exited
  with finite logits.
- Both input and output accessors returned a matrix of shape `[151936, 1024]`.
- The two accessors returned the same Python parameter object backed by the same
  storage, verifying runtime weight tying.
- The tokenizer exposes IDs through 151,668, but the model contains and scores
  267 additional rows through ID 151,935.
- The configured model vocabulary size is divisible by 128.

## Interpretations

- The embedding-table row count follows the model's configured vocabulary
  dimension, which need not equal the number of entries exposed by the paired
  tokenizer snapshot.
- The 267 extra rows are not addressable through ordinary encoding with this
  tokenizer, yet the model's output projection emits logits for them.
- Because the weights are tied, one parameter serves both input lookup and output
  classification. Analytically, extra rows can therefore receive dense
  output-classifier gradients even though the tokenizer cannot select them as
  ordinary input IDs; this run did not measure such training gradients.
- Divisibility by 128 is an observed structural fact. Calling the extra rows
  hardware padding is plausible but not established by the inspected artifacts.

## Claims not established

- The inspection does not establish why Qwen selected 151,936 model rows.
- It does not measure hardware performance benefits or costs.
- It does not measure probability mass assigned to tokenizer-unassigned IDs.
- It does not establish semantic quality, capability, or training improvement.

## Failures or surprises

- No load, identity, shape, or finite-logit check failed.
- The serialized checkpoint contains names for both matrices, but runtime loading
  ties them into the same parameter. Serialized names alone were therefore not
  treated as evidence of untied runtime weights.

## Decision and next experiment

- Decision: pass. The remaining Qwen3-specific Day 2 embedding questions now
  have reproducible structural evidence.
- Exact next action: integrate the verified tokenization, embedding, contextual
  representation, masking, and gradient arguments into Chapter 2 and its
  exercises.
