# Lab 9 — From an answer gradient to a defended recipe

## CPU lesson route

Study [objective and gradient paths](../../notebooks/day-12/01_sft_objective_and_gradient_paths.ipynb), then [accumulation and restart](../../notebooks/day-12/02_accumulation_and_checkpoint_identity.ipynb). Continue with [tiny assistant training](../../notebooks/day-13/01_tiny_assistant_training.ipynb) and [full versus LoRA](../../notebooks/day-14/01_full_sft_lora_and_recipe_defense.ipynb).

The reusable [sft_lab.py](../../src/dongxi_llms/sft_lab.py) is a CPU-only reference. Reproduce the measured 100-update fixture comparison:

~~~bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m dongxi_llms.sft_lab --steps 100 --mode full
PYTHONPATH=src OMP_NUM_THREADS=1 python -m dongxi_llms.sft_lab --steps 100 --mode lora
~~~

These commands train a random 6,120-parameter decoder on four known symbolic copy requests. The adapter variant adds 384 trainable Q/V scalars. They verify mechanism behavior and supply observed plots; neither command loads or tests Qwen.

## Response-distillation CPU extension

The separate [complete-response distillation notebook](../../notebooks/day-26/03_response_level_distillation.ipynb) is a Chapter 15 extension with a Chapter 9 objective bridge. It fits an actual larger tiny teacher, retains its unforced sampled responses, and compares same-initialization complete-response and answer-only students. It checks response/EOS labels, exact single shift, padding, parent provenance, source leakage and printed-step versus final-answer outcomes. Its 15-token symbolic grammar is not pretrained language reasoning.

The immutable [specification](../../experiments/specs/2026-10-04-response-distillation.md) precedes the [measured report](../../experiments/reports/2026-10-04-response-distillation.md). Reproduce only into a new output directory; the runner refuses an existing one and preserves partial attempt/fit journals on ordinary failure:

~~~bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m dongxi_llms.response_distillation_lab \
  --fixture fixtures/response-distillation/items.json \
  --spec experiments/specs/2026-10-04-response-distillation.md \
  --output outputs/response-distillation-new-run
~~~

This fixed CPU command runs all three seeds, not a pretrained teacher. Both student arms use 80 updates but unequal supervised-token budgets. Missing eligible teacher coverage blocks both student arms without gold repair. All actual attempts, invalid formats, cap stops and selection decisions remain in `responses.jsonl`, `events.jsonl` and `results.json`. The notebook also includes adjacent reference answers, a fresh fixed-seed numeric replay and inspected figures.

### Local response-distillation transfer gate — not executed

The source function `prepare_local_transfer()` in [response_distillation_lab.py](../../src/dongxi_llms/response_distillation_lab.py) only validates a bounded future protocol. It requires two distinct existing local full/merged checkpoint directories with saved config/tokenizer and safetensors names, disjoint source groups, at most eight teacher attempts per item, a maximum 256-token teacher response, at most 200 student updates and at least 25 GiB host reserve. It explicitly rejects execution; no model is loaded. Filename inspection is not validation of actual weight contents.

Before a separately authorized Spark run, hash all local weight files and freeze each checkpoint's actual tokenizer/template/ending interface. Re-encode teacher text with the student's tokenizer; do not copy token-ID arrays between vocabularies. Freeze a reviewed natural-language rationale task, reject truncation and ambiguous endings, preserve prompt-excluded response/EOS labels, and charge all teacher attempts and rejections separately from student work. Hardware profiling and external timeout/memory supervision remain pending. See the [prepared transfer protocol](../../experiments/specs/2026-10-04-response-distillation-spark-transfer.md); the symbolic STEP checker does not certify natural-language explanations.

## Actual native profile and fixed replay acceptance

Later measured case: the [fresh full/LoRA400 study](../../experiments/reports/2026-10-05-native-sft400-comparison.md)
now completes the fixed learning recipe and original120-item publication panel.
Compare its9,321 matched training labels with the different trainable parameter
spaces, development NLL, whole responses and message endings. The
[assistant-study card](../../docs/model_cards/native-assistant-interface-study.md)
names the actual selected full400 artifact and separates it from profile,
recovery and merged-adapter exports. Do not mistake the historical short-run
limits below for evidence that the400-update runs are still unexecuted.

The [actual20-update profile](../../experiments/reports/2026-10-05-native-base-profile.md)
now passes its declared runtime/interface/resource checks. Recompute both NLL
panels from the retained raw sums/counts, then inspect exact-answer and ending
rates separately. Both remain0/8 despite lower NLL; do not describe the model
as a working assistant. The frozen publication test has not been used here.

The subsequent [fixed replay specification](../../experiments/specs/2026-10-05-native-sft-replay.md)
compares native full and rank-8 Q/V LoRA completed-update recovery at the same
20-update horizon. Its explicit Spark-only collector is not invoked by notebook
or CPU verification. Both native full and LoRA update10→20 replays now match
their uninterrupted numerical tails, serialized tensor entries and final
token/stop records. The first full replay was stopped by a concurrent
runner-named CPU process; its failure remains alongside the successful fresh
retry using the same journals and caps. Read the
[actual replay report](../../experiments/reports/2026-10-05-native-sft-replay.md)
and compare numerical training targets455 with cumulative physical targets684.
Then inspect the failed BF16 merge and separately declared successful FP32
merge/reload. Why does the latter not prove original BF16 logit equivalence?
Neither exact recovery nor compatible export establishes useful answering;
both final generation panels still score0/8 exact answers and message endings.

## Prepared base-model Spark route

The [CUDA runner](../../scripts/run_chapter09_spark_sft.py) supports a bounded opt-in experiment, with explicit revision, template, objective, runtime and memory controls. It requires the platform-validated CUDA/BF16 environment, Transformers and PEFT for adapter mode. Inspect installed versions and use the platform lock; the course does not modify a shared environment automatically.

Official base revision snapshots verified on 2026-10-04:

| Model | Exact repository revision |
|---|---|
| [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd) | da87bfb608c14b7cf20ba1ce41287e8de496c0cd |
| [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base/tree/ea980cb0a6c2ae4b936e82123acc929f1cec04c1) | ea980cb0a6c2ae4b936e82123acc929f1cec04c1 |

Generate the original English dataset as described in Lab 8. The following is
the historical prepared smoke/profile command, with deliberately unresolved
resource placeholders. The later actual profile above used fully specified,
hashed declarations through the external supervisor; this illustrative command
is not its execution receipt:

~~~bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/run_chapter09_spark_sft.py \
  --model Qwen/Qwen3-0.6B-Base \
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --tokenizer-revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --template experiments/data/instruction_interface_v1.jinja \
  --train outputs/course-sft-interface-v1/train.jsonl \
  --dev outputs/course-sft-interface-v1/dev.jsonl \
  --output outputs/course-sft-0.6b-full-smoke \
  --mode full --updates 20 --microbatch 1 --accumulation 4 \
  --max-length 256 --learning-rate 0.00002 \
  --runtime-seconds 900 --reserve-gib 25 --seed 1212 \
  --snapshot-max-bytes "$APPROVED_SNAPSHOT_MAX_BYTES" \
  --work-limits "$APPROVED_SFT_WORK_LIMITS_JSON" \
  --work-journal-max-bytes "$APPROVED_WORK_JOURNAL_MAX_BYTES" \
  --snapshot-io-limits "$APPROVED_SNAPSHOT_IO_CONTRACT_JSON" \
  --snapshot-io-ledger "$APPROVED_SNAPSHOT_IO_LEDGER_PATH" \
  --environment-lock /home/dongxi/dgx-spark-dongxi/uv.lock
~~~

`APPROVED_SNAPSHOT_MAX_BYTES` is a required separately sized/approved byte
envelope, not a supplied measurement or the 16MiB CPU fixture limit. The file
buffer, restored tensors and save overhead require their own resource budget;
this prepared command is incomplete until that value is declared.

Declare all sixteen v2 SFT caps separately: the original twelve cover complete updates, selected
examples/cursor steps, targets, sequence tokens, policy calls/positions,
evaluation calls/positions and generation calls/prefix bounds/returned tokens.
The runner reserves complete accumulation windows and complete observer panels;
it does not shorten the objective to fit remaining capacity. Its original cached
greedy full/LoRA generation is unchanged; a conservative uncached allowance is
not reported as the actual cached position count. The four new caps are
`recovery_validation_operations`, `recovery_history_rows`,
`recovery_tensor_elements` and `recovery_rng_states`. Each actual runner semantic
check reserves its own panel before scans/probes; the saved prefix includes that
charge. Old twelve-cap files refuse without migration. These units do not bound
the shared reader's pre-callback hashing/deserialization/tree checks, save
cloning/serialization, FLOPs, backward recomputation or all output storage.

The separate I/O JSON is a full strict `dongxi-snapshot-io-work-v1` contract,
not another SFT16 cap file. It declares all nine shared-operation limits,
whole-operation payload/node/element/tensor-byte/primitive-byte envelopes and
its journal byte bound. Its payload limit must equal the approved snapshot
limit. The named ledger belongs in a privately owned directory; on resume it
must be the same physical journal. No fixture value is a production default.
Read the [I/O source boundary](../../experiments/reports/2026-10-05-snapshot-io-readiness.md)
and size the complete commit/inspect/load schedule before approval. Bootstrap
metadata/journal processing, caller state capture, application, other outputs
and physical resources remain outside this separate logical ledger.

This performs full-development answer NLL and the first eight development IDs' greedy completions before/after training. Both share a 64-token budget; any prompt-plus-generation length exceeding the declared context is rejected. It logs every update and stops on nonfinite state, runtime cap or sampled memory-reserve violation.

Generation stops on the explicit end-of-message token or the pinned tokenizer's EOS. The result distinguishes the expected message ending from a generic EOS stop. This matters because a base tokenizer's EOS need not equal its chat-template end marker.

The runner's memory/time checks occur between model operations; an external job supervisor is needed for a hard kill deadline. Stop known GPU inference services before authorizing training. Prepared commands do not launch jobs.

### Prepare the external wrapper separately

The [fixed-profile adapter](../../scripts/prepare_native_sft_profile.py) prepares
only this pinned20-update recipe. Its default mode constructs and hashes a
command without starting it; it accepts no arbitrary extra argv, acquisition,
resume or pilot recipe. Supply explicit absolute input/interpreter/lock paths,
new output and private journal paths, complete SFT16/I/O9 declarations and the
snapshot byte envelope. Positive declared limits do not establish that those
limits cover the actual encodings or that the model fits.

Inspect the preparation interface without starting a job:

~~~bash
PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/prepare_native_sft_profile.py --help
~~~

Its separately explicit execution route requires operator scope text and a
new supervision-evidence directory. That text is retained evidence, not an
authenticated permission grant. The [source verification contract](../../experiments/specs/2026-10-05-native-profile-watchdog.md)
uses inert CPU children to test observer/logging failures and bounded owned
shutdown; it does not load the intended Base checkpoint. At that source
checkpoint, local Base bytes, tokenizer sizing and approved envelopes were
unresolved. The subsequent
[approved acquisition and CPU sizing](../../experiments/reports/2026-10-05-base-tokenizer-sizing.md)
now verify the exact local Base bytes and 1,175 scheduled targets: 455 training
plus two development panels of 360. All eight prompts plus the fixed generation
cap fit the context. The original records were regenerated in memory; native
input-file preparation, model-dependent validation/snapshot/I/O allowances and
launch approval remain pending. No GPU execution follows from this receipt.
Do not infer GPU clearance, continuous memory
safety or a physical quota from a prepared JSON command.

After profile acceptance, a proposed learning run uses a new output directory, 400 updates and a 3,600-second cap. The same supervised data order is the fixed-recipe control. For the adapter comparison, use another new directory and add --mode lora --rank 8. The matched learning rate is a controlled setting, not a claim of method-optimal tuning. A separately tuned comparison needs a declared search budget.

The larger transfer check uses the pinned 1.7B-Base revision above and a new run identity only after the smaller run passes the health and regression gates in the specification. Its memory and runtime remain unmeasured until profiling.

## Output and genealogy

Each real run writes config.json with revisions and source hashes,
recovery-contract.json with stable scientific identity, metrics.jsonl,
result.json with raw baseline/candidate completions, immutable
checkpoint-UPDATE.pt payloads with adjacent .commit.json markers, a
latest-completed-snapshot.json pointer, and a policy directory.

The selected lock path above is Spark-specific; select the existing declared lock
on another host. No lock installation is implied. Every invocation additionally
writes `identity-<invocation>.json`: Git/dirty state, source/input/actual cached-model
hashes, Python/packages/lock, command, hardware/driver and interface. Stable config
includes source/lock/interface identity, but not volatile journal timestamps.
The actual local input snapshot is hashed before model loading. Cached inputs are
the default; only `--allow-download` explicitly authorizes acquisition. Cooperative
guards cannot interrupt one blocking acquisition or model operation.

An initial completed-zero snapshot precedes baseline observation. Later
snapshots commit after every 20 completed updates by default and at the final
boundary, before final evaluation/export. Use --checkpoint-every to declare a
different interval. Payloads and markers are exclusive; no successful boundary
is overwritten. A failed save retains its partial files and the earlier
completed boundary. Failure before the initial commit supplies no resumable state.

In full mode, policy is an HF-loadable model plus tokenizer. In LoRA mode, policy is a PEFT adapter plus tokenizer and requires the recorded pinned base. A downstream trainer that requires a full HF checkpoint must explicitly load and merge that adapter, then save a new derived checkpoint with genealogy; an adapter directory is not silently interchangeable with full weights.

Use --resume with a trusted-local completed payload plus --resume-contract,
--resume-sha256 and --resume-bytes from separately retained expectations. The
resume additionally requires `--resume-io-receipt`, the bounded independently
retained work receipt from the selected accounted snapshot, plus the same main
work journal and I/O journal. Bind their prefixes before payload verification
and model allocation; keep later failed spending. A new output path does not
create a new allowance. The marker is untrusted metadata, not a replacement
for the expected receipt.
Before model allocation, current observable identity and actual payload bytes
are checked; before applying state, the loop checks parameter/optimizer binding,
RNG, fixed order/cursor, history and cumulative target/position invariants.
Legacy unrestricted checkpoints are not implicitly migrated. Resume keeps the
same fixed update horizon; extending it is a separately declared experiment.

Supply `--work-journal` on resume with the same retained physical private journal.
Its trusted snapshot prefix validates before numerical state is applied, while
later failed/observed work remains charged. The new output directory is evidence
isolation, not a new allowance. Initial parsed cap bytes are independently
SHA-bound and rechecked; do not substitute a later changed file's hash.

Use a **new empty output directory for each invocation, including resume**. The
trusted resume checkpoint's bytes are input-hashed; failed compatibility cannot
overwrite the previous run's configuration or exports. A new invocation does not
change the stable recipe. Keyboard interruption finalizes its partial journal;
SIGKILL/power failure requires external supervision and cannot promise a final write.

For a local adapter, [merge_course_adapter.py](../../scripts/merge_course_adapter.py)
requires explicit local base/adapter paths, their recorded base revision, selected
lock and a new output. It writes a full checkpoint under `output/policy`; parent
hashes, supported tokenizer interface and actual tiny CPU merge/reload checks are
documented in the [identity report](../../experiments/reports/2026-10-04-run-identity.md).
Size and authorize a real merge separately. This course preparation does not run it.

Each metric row records an invocation ID and the restored update boundary.
Completed numerical history travels with the snapshot. If a failed attempt
logged updates after its last saved checkpoint, their retried update numbers
remain visible under a different invocation. Do not concatenate them as extra
completed optimizer steps. Cumulative supervised targets and physical training
positions belong to the restored trajectory, not a fresh zero counter.
The [source recovery specification](../../experiments/specs/2026-10-05-sft-runner-recovery.md)
uses actual tiny random local HF full/LoRA loops; its source checks are separate
from approved pretrained compatibility and Spark/BF16 numerical recovery.

## Interpret the outcome

Report answer accuracy, output-format validity and natural termination separately. Preserve all fixed-grid outputs, including regressions. The runner's eight-item grid is a bounded qualitative diagnostic, not publication evaluation. Final claims use the frozen 120-item suite with source-group uncertainty and task slices; do not use it to tune the recipe.

Defend the recipe as choice → rationale → evidence → trade-off → failure risk → next experiment. The actual CPU report verifies mechanics. No real-model capability or measured Spark SFT throughput is implied before those commands are authorized and completed.
