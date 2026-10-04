# Chapter 11 — DPO Laboratories and Spark Extension

| Day | Session | Mechanism and evidence |
|---|---|---|
| 17 | [Finite KL optimum](../../notebooks/day-17/01_kl_regularized_optimum.ipynb) | Exact exponential tilt, stationarity, beta, reward gauge |
| 17 | [Sequence masks](../../notebooks/day-17/02_sequence_likelihood_and_masks.ipynb) | One shift; `[B,T,V]` logits; prompt/response/EOS/pad heatmap; sum versus average |
| 18 | [Controlled comparison](../../notebooks/day-18/01_dpo_controlled_comparison.ipynb) | Categorical DPO/label-flip/SFT plus actual tiny decoder DPO/SFT and an independent held-out token metric |

These CPU sessions need PyTorch/Matplotlib and no downloaded checkpoints. Use
`dpo_lab.py` for reusable calculations and the [worked answers](../solutions/11-direct-preference-optimization.md)
for conceptual explanations. The negative tiny-sequence outcome is useful
evidence about margins versus absolute likelihood, not a benchmark verdict.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/run_preference_policy_cpu.py
python scripts/verify_course_notebooks.py --days 17 18 --kernel dgx-spark-native --export-figures
PYTHONPATH=src python scripts/run_chapter11_spark_dpo.py --help
```

## Runnable Spark route

The optional `run_chapter11_spark_dpo.py` accepts a **local full or merged HF-format SFT
checkpoint** (an unmerged LoRA adapter is rejected), an explicitly identified tokenizer and exact 40-character revision,
train/validation preference JSONL, and independent generation JSONL. The base
checkpoint should come from Chapter 9's base-to-assistant experiment; using an
already post-trained assistant is a different experiment and must be identified.
The Chapter9 base route pins `Qwen/Qwen3-0.6B-Base` at
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; its tokenizer metadata includes the
chat template. The runner restores the **audited template saved with the SFT
checkpoint** from `chat_template.jinja` (or the legacy tokenizer JSON field),
because Chapter9 explicitly supplies its own template rather than silently
adopting the Hub template. A missing template or a mismatch against checkpoint
genealogy is rejected.
Validate vocabulary/revision compatibility with the actual SFT checkpoint rather
than assuming matching names are sufficient. Generation stops on either the
tokenizer EOS or its recognized chat-turn-ending token, avoiding a base EOS
convention that would continue into another turn.

Preference row contract:

```json
{"id":"train-001","prompt":[{"role":"user","content":"Answer with the location only. Lily put her ball in the red box. Where is it?"}],"chosen":"the red box","rejected":"the garden"}
```

Independent row contract:

```json
{"id":"test-001","prompt":[{"role":"user","content":"Answer with the location only. Tom put the key in a green bag. Where is it?"}],"expected":"the green bag"}
```

Use separate source groups and disjoint exact prompts for each split. These
examples illustrate the schema; they are not a sufficient training or evaluation
dataset. The runner rejects duplicate IDs across splits, exact prompt overlap,
empty completions, overlength examples, and template-prefix mismatches. It scores
the response and termination tokens only, shifts exactly once, and keeps the
reference frozen. An unsupported tokenizer boundary is a visible error, not an
automatic silent repair.

The complete command, after preparing the checkpoint and frozen authored files,
is:

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 python scripts/run_chapter11_spark_dpo.py \
  --checkpoint outputs/chapter09-sft/policy \
  --tokenizer Qwen/Qwen3-0.6B-Base \
  --tokenizer-revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --train fixtures/chapter11/train.jsonl \
  --validation fixtures/chapter11/validation.jsonl \
  --evaluation fixtures/chapter11/evaluation.jsonl \
  --output outputs/chapter11-dpo-01 \
  --updates 100 --accumulation 4 --beta 0.1 --lr 5e-7 --max-length 512
```

The command is a proposed run and is not executed when the course material is
created. The authored fixture files exist; the SFT checkpoint path is a product
of the Chapter9 run, not evidence fabricated during this preparation. The small
fixture targets location extraction, not broad preference alignment; expand
it only under a new predeclared data and evaluation contract.
The runner defaults to cached tokenizer files and refuses an existing output
directory. Only `--allow-download` authorizes fetching that pinned tokenizer;
model weights must already exist locally. It uses full FP32 policy/reference
weights with BF16 autocast on CUDA, SDPA, no stochastic dropout, policy gradient
checkpointing, four one-pair accumulation microbatches, clipping at one, a
30-minute whole-run sampled wall cap, and a sampled 25 GiB host reserve. Guards
cover loading boundaries, accumulation, validation, and independent generation;
a single blocking model call can exceed a sampling interval. This memory
policy is a guard, not a prediction of peak memory or a guarantee against
between-sample spikes. Stop any known inference service before a new GPU run.

Outputs include input/checkpoint/template hashes, environment versions, per-update
loss/margins/gradients, sampled memory, held-out pair scores, independent exact
generation outputs before/after, GPU driver/Git/source identity, HF policy files
with parent-checkpoint genealogy, and optimizer/RNG state. The
runner currently saves recovery state but does not expose a resume CLI; do not
claim resumable training from a file's mere existence. Independent exact match
is deliberately strict; a fuller frozen rubric can be applied to saved outputs.
It is not tuned to the reward-margin metric.

Preserve an untouched SFT checkpoint and run matched controls under a separate
approved budget. Model-scale correctness, template compatibility, performance,
and sample quality remain unverified until an actual Spark smoke and learning
run are recorded. The bounded CPU report verifies the shared objective and mask
mechanisms; it cannot replace these hardware checks.
