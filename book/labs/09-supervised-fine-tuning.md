# Lab 9 — From an answer gradient to a defended recipe

## CPU lesson route

Study [objective and gradient paths](../../notebooks/day-12/01_sft_objective_and_gradient_paths.ipynb), then [accumulation and restart](../../notebooks/day-12/02_accumulation_and_checkpoint_identity.ipynb). Continue with [tiny assistant training](../../notebooks/day-13/01_tiny_assistant_training.ipynb) and [full versus LoRA](../../notebooks/day-14/01_full_sft_lora_and_recipe_defense.ipynb).

The reusable [sft_lab.py](../../src/dongxi_llms/sft_lab.py) is a CPU-only reference. Reproduce the measured 100-update fixture comparison:

~~~bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m dongxi_llms.sft_lab --steps 100 --mode full
PYTHONPATH=src OMP_NUM_THREADS=1 python -m dongxi_llms.sft_lab --steps 100 --mode lora
~~~

These commands train a random 6,120-parameter decoder on four known symbolic copy requests. The adapter variant adds 384 trainable Q/V scalars. They verify mechanism behavior and supply observed plots; neither command loads or tests Qwen.

## Prepared Spark route

The [CUDA runner](../../scripts/run_chapter09_spark_sft.py) supports a bounded opt-in experiment, with explicit revision, template, objective, runtime and memory controls. It requires the platform-validated CUDA/BF16 environment, Transformers and PEFT for adapter mode. Inspect installed versions and use the platform lock; the course does not modify a shared environment automatically.

Official base revision snapshots verified on 2026-10-04:

| Model | Exact repository revision |
|---|---|
| [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd) | da87bfb608c14b7cf20ba1ce41287e8de496c0cd |
| [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base/tree/ea980cb0a6c2ae4b936e82123acc929f1cec04c1) | ea980cb0a6c2ae4b936e82123acc929f1cec04c1 |

Generate the original English dataset as described in Lab 8. The following is the concrete smoke/profile command, prepared but not executed as part of course writing:

~~~bash
OMP_NUM_THREADS=1 python scripts/run_chapter09_spark_sft.py \
  --model Qwen/Qwen3-0.6B-Base \
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --tokenizer-revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --template experiments/data/instruction_interface_v1.jinja \
  --train outputs/course-sft-interface-v1/train.jsonl \
  --dev outputs/course-sft-interface-v1/dev.jsonl \
  --output outputs/course-sft-0.6b-full-smoke \
  --mode full --updates 20 --microbatch 1 --accumulation 4 \
  --max-length 256 --learning-rate 0.00002 \
  --runtime-seconds 900 --reserve-gib 25 --seed 1212
~~~

This performs full-development answer NLL and the first eight development IDs' greedy completions before/after training. Both share a 64-token budget; any prompt-plus-generation length exceeding the declared context is rejected. It logs every update and stops on nonfinite state, runtime cap or sampled memory-reserve violation.

Generation stops on the explicit end-of-message token or the pinned tokenizer's EOS. The result distinguishes the expected message ending from a generic EOS stop. This matters because a base tokenizer's EOS need not equal its chat-template end marker.

The runner's memory/time checks occur between model operations; an external job supervisor is needed for a hard kill deadline. Stop known GPU inference services before authorizing training. Prepared commands do not launch jobs.

After profile acceptance, a proposed learning run uses a new output directory, 400 updates and a 3,600-second cap. The same supervised data order is the fixed-recipe control. For the adapter comparison, use another new directory and add --mode lora --rank 8. The matched learning rate is a controlled setting, not a claim of method-optimal tuning. A separately tuned comparison needs a declared search budget.

The larger transfer check uses the pinned 1.7B-Base revision above and a new run identity only after the smaller run passes the health and regression gates in the specification. Its memory and runtime remain unmeasured until profiling.

## Output and genealogy

Each real run writes config.json with revisions and source hashes, metrics.jsonl, result.json with raw baseline/candidate completions, checkpoint.pt with model/AdamW/RNG/data cursor state, and a policy directory.

The checkpoint is atomically replaced after every20 completed updates by default, and at the final boundary. A failure during the next save retains the previous completed checkpoint. Use --checkpoint-every to declare a different interval. A failure before the first checkpoint has no resumable optimizer state.

In full mode, policy is an HF-loadable model plus tokenizer. In LoRA mode, policy is a PEFT adapter plus tokenizer and requires the recorded pinned base. A downstream trainer that requires a full HF checkpoint must explicitly load and merge that adapter, then save a new derived checkpoint with genealogy; an adapter directory is not silently interchangeable with full weights.

Use --resume with the trusted local checkpoint and the same configuration to continue an interrupted budget. Model weights alone reproduce neither AdamW moments nor data cursor. The saved result's supervised-target count is for that invocation, so aggregate resumed segments explicitly.

Each metric row records an invocation ID and the restored update boundary. If a failed attempt logged updates after its last saved checkpoint, their retried update numbers remain visible under a different invocation. Do not concatenate those attempts as though they were extra completed optimizer steps in one trajectory.

## Interpret the outcome

Report answer accuracy, output-format validity and natural termination separately. Preserve all fixed-grid outputs, including regressions. The runner's eight-item grid is a bounded qualitative diagnostic, not publication evaluation. Final claims use the frozen 120-item suite with source-group uncertainty and task slices; do not use it to tune the recipe.

Defend the recipe as choice → rationale → evidence → trade-off → failure risk → next experiment. The actual CPU report verifies mechanics. No real-model capability or measured Spark SFT throughput is implied before those commands are authorized and completed.
