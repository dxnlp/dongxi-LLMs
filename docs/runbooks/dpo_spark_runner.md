# DPO and matched chosen-SFT operations

Use this reference after the [guided lab](../../book/labs/11-direct-preference-optimization.md).
The commands and receipts below retain their original historical scope. The CPU
notebook route reads existing evidence; model-scale execution has a separate
checkpoint, frozen recipe and bounded-resource contract.

## Matched chosen-only source path

The [matched helper](../../src/dongxi_llms/chosen_sft_control.py) and
[local-only adapter](../../scripts/run_matched_chosen_sft.py) reuse the native
SFT cross-entropy and DPO encoding/update paths. They match actual parent,
source/encoded interface, replacement draws and chosen targets. They do not
match rejected supervision, reduction or total compute. The
[protocol](../../experiments/specs/2026-10-05-matched-chosen-sft-control.md)
uses the unchanged Chapter 11 location fixtures and an explicit scenario-group
sidecar, not an easier substituted task.

```bash
PYTHONPATH=src:tests OMP_NUM_THREADS=1 python -m unittest \
  test_chosen_sft_control -v
python scripts/run_matched_chosen_sft.py --help
```

The [measured two-seed CPU comparison](../../experiments/reports/2026-10-05-matched-chosen-sft-control.md)
retains all six arm rows, raw greedy outputs and 0/4 exact match throughout.
Inspect the actual matching receipts and unequal training calls/positions,
then compare natural stops with correctness. Same chosen exposure is not proof
that either objective learned useful behavior. Chapter 11 exercise 15 supplies
the sampler/reduction explanation.

The adapter takes local checkpoint/tokenizer paths and an explicit
`--arm unchanged|chosen-sft|dpo`; all three commands must use the same
independently bound parent, data, group sidecar and frozen recipe. Its explicit
`--device cpu|cuda` defaults to CPU and never falls back or downloads.
CUDA/pretrained execution is unverified and requires separate authority.
The sampled memory/deadline guard is not an external supervisor; interrupted
chosen-SFT updates are poisoned, not accounted resumptions. Production
recovery/accounting and stage-specific exposure/resource limits remain gates
before the intended Spark pilot. Do not use tiny fixture limits as a model-scale
budget or fill the planned campaign's actual fields from these CPU results.

The separate [native counted replay](../../experiments/reports/2026-10-05-native-chosen-replay.md)
has now passed clean/source/fresh-resume numerical and metric-tail equality on
the selected full400 parent. Compare its 40-label numerical trajectory with 60
physical source/resume presentations and its 0/4 correct,4/4 naturally stopped
diagnostic answers. This exercise reads existing evidence only. It does not
retrofit the older adapter's accounting, run the 100-update pilot, or supply the
common DPO/chosen retained-capability comparison.

The [actual 100-update pilot report](../../experiments/reports/2026-10-05-native-preference-comparison.md)
adds a read-only native case study for Chapter 11 exercise 17. Compare the exact
chosen exposure with the unequal rejected/reference work. Then read each raw
four-answer diagnostic: favorable DPO margins, natural stops and strict answers
are different observations. The pilots have accepted technical receipts, but
their own FP32/BF16-autocast panels are not the common BF16-loaded evaluation.
The completed common comparison now supplies those separate receipts and
source-bound figures: location 0/4,4/4,1/4; instruction 120/120 in all three arms;
annotated reasoning 5/20,6/20,6/20. Every response stops naturally. Inspect the
unchanged frozen parser, overlap slices and per-item outcomes; do not mistake
the larger DPO margin for more exact answers. The
[CPU figure consumer](../../scripts/plot_native_preference_comparison.py) joins
actual producer/identity/raw-record evidence without loading a model, mutating
the source or producing new scores. Its
[accepted output](../../experiments/reports/native-preference-figures-20261005-run-01/acceptance.json)
is already retained; do not rerun an exclusive directory or launch another GPU
job just to study these results.

## Runnable Spark route

The optional `run_chapter11_spark_dpo.py` accepts a **local full or merged HF-format SFT
checkpoint** (an unmerged LoRA adapter is rejected), an explicitly identified tokenizer and exact 40-character revision,
train/validation preference JSONL, and independent generation JSONL. The base
checkpoint should come from Chapter 9's base-to-assistant experiment; using an
already post-trained assistant is a different experiment and must be identified.
The Chapter 9 base route pins `Qwen/Qwen3-0.6B-Base` at
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; its tokenizer metadata includes the
chat template. The runner restores the **audited template saved with the SFT
checkpoint** from `chat_template.jinja` (or the legacy tokenizer JSON field),
because Chapter 9 explicitly supplies its own template rather than silently
adopting the Hub template. A missing template or a mismatch against checkpoint
genealogy is rejected.
The runner verifies the actual saved parent's semantic fingerprint and then the
proposed tokenizer: token-to-ID meanings, serialized encoding/wrapper rules,
special IDs, template, stops and recorded source declaration must agree. Shape or
model-name equality is insufficient. A legacy checkpoint without this fingerprint
fails closed unless `--allow-legacy-interface` explicitly adopts its audited saved
semantics; adoption cannot prove an old revision. Generation stops on either the
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
  --updates 100 --accumulation 4 --beta 0.1 --lr 5e-7 --max-length 512 \
  --snapshot-max-bytes "$APPROVED_SNAPSHOT_MAX_BYTES" \
  --work-limits "$APPROVED_WORK_LIMITS_JSON" \
  --work-journal-max-bytes "$APPROVED_WORK_JOURNAL_MAX_BYTES" \
  --snapshot-io-limits "$APPROVED_SNAPSHOT_IO_CONTRACT_JSON" \
  --snapshot-io-ledger "$APPROVED_SNAPSHOT_IO_LEDGER_PATH" \
  --snapshot-artifact-max-bytes "$APPROVED_SNAPSHOT_ARTIFACT_BYTES" \
  --snapshot-artifact-max-entries "$APPROVED_SNAPSHOT_ARTIFACT_ENTRIES" \
  --snapshot-artifact-journal-max-bytes "$APPROVED_SNAPSHOT_ARTIFACT_JOURNAL_BYTES" \
  --environment-lock /home/dongxi/dgx-spark-dongxi/uv.lock
```

The required `APPROVED_SNAPSHOT_MAX_BYTES` must be separately declared before
an approved run. It is not a measured production size or the 16MiB fixture cap.
Size the file buffer, restored tensors and save overhead explicitly; this
prepared command is incomplete until that envelope is approved.

The work-limit JSON and journal envelope also require separate declaration.
The JSON supplies the exact 19 v2 dimensions in `BUDGET_KEYS` of the runner, with
bounded integer values. The original 14 cover updates, draws/examples, targets/logical tokens,
policy/reference calls and positions, evaluation calls/positions and generation
calls/prefix-position bound/output tokens. Several dimensions intentionally
overlap; do not add their totals as FLOPs. Use actual encoded masks/lengths and
the full baseline/final panels to size a recipe, not an update-count shortcut.
Conservative reservations never refund unused or failed work. These limits cover
cooperative logical operations, not CUDA memory or arbitrary exports. The new
five are `recovery_validation_operations`, `recovery_history_rows`,
`recovery_tensor_elements`, `recovery_rng_states` and `recovery_sampler_draws`.
Reserve each actual initial/periodic/final/load-callback/restore semantic panel;
duplicate checks are not free. Replayed history uses a private generator, never
the live sampler. Old 14-cap/v1 files refuse without migration. Generic shared
reader hashing/deserialization/tree work and save cloning/serialization remain
outside this semantic envelope. They use a separately explicit nine-dimensional
I/O contract; see the [reader source boundary](../../experiments/reports/2026-10-05-snapshot-io-readiness.md).
The full `dongxi-snapshot-io-work-v1` JSON supplies its logical limits,
whole-operation payload/node/element/tensor-byte/primitive-byte envelopes and
journal bound. Match its payload limit to the approved snapshot limit; do not
borrow the 16MiB test cap or append/refill the original `DPO19` dimensions.
The named I/O ledger must stay in a private directory and be the same physical
journal on resume. The separate pure schedule calculator does not authenticate
the supplied geometry or grant permission to execute it.

The three snapshot-artifact capacities are also explicit and separately sized.
They cover direct payload/marker/staging/work-receipt pathnames, including peak coexistence
and retained failures; the journal's full byte capacity is charged up front.
They do not cover metrics, identity/status files, HF exports, scratch or arbitrary
library/child writes. A cooperative envelope is not a physical filesystem quota.

The command is a proposed run and is not executed when the course material is
created. The authored fixture files exist; the SFT checkpoint path is a product
of the Chapter 9 run, not evidence fabricated during this preparation. The small
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
with parent-checkpoint genealogy, and exclusive completed policy/reference/
Adam/RNG snapshots under checkpoints, each with a .commit.json marker. Initial
completed-zero and periodic/final snapshots precede baseline observation,
metric publication at saved boundaries, final evaluation and HF export.
Independent exact match
is deliberately strict; a fuller frozen rubric can be applied to saved outputs.
It is not tuned to the reward-margin metric.

Resume uses a new output directory and independently retained --resume-contract,
`--resume-sha256` and --resume-bytes alongside --resume.
An accounted resume also requires `--resume-io-receipt` and both original
physical work journals. The bounded retained receipt binds payload bytes and
its pre-save I/O/runner prefixes before a full hash or model allocation; the
later journal suffix remains spent. Source/lock/raw and encoded
data/parent/interface/objective identity is compared before allocating models;
actual effective model/reference configuration and frozen-reference tensor
identity are then recomputed before loading/applying the restricted snapshot.
Do not read an unchecked marker to manufacture those independent expectations.
The fixed update horizon remains part of this same-recipe contract.

Budgeted recovery additionally requires `--work-journal` pointing to the same
physical retained work journal, outside the new invocation output. A snapshot
attests its prefix; restore validates that prefix before applying numerical
state and retains later charged attempts. Changing limits, copying the journal
or creating a new allowance does not produce a same-recipe resume. Baseline and
final evaluation and uncached greedy generation reserve full panels before work;
their conservative bounds and actual successful/partial calls remain distinct.
See Chapter 14 section 14.8 for why numerical and resource clocks can diverge.

Snapshot-budgeted resume also requires `--snapshot-artifact-root` for the same
retained flat private root and `--snapshot-artifact-receipt` for its separately
retained identity/prefix JSON. The loaded state's prefix must match that active
ledger before application; later saves and failed partials remain charged.
New invocation snapshot names are exclusive in the old root. Payload SHA/bytes
and science-contract expectations remain independently required. The
[actual CPU integration report](../../experiments/reports/2026-10-05-dpo-snapshot-artifacts.md)
tests these distinctions with the original random fixture, not a pretrained run.

The snapshot retains the *original* reference, policy, Adam, pair/global RNG,
cumulative branch work and complete numerical history. Recovered metrics are
kept separately from new invocation attempts. Resetting the reference or work
counters changes the experiment; a failed metric/export does not erase the
last durable boundary. --checkpoint-every controls periodic saves, not the
objective. The [actual-loop CPU specification](../../experiments/specs/2026-10-05-dpo-runner-recovery.md)
uses original random local HF models, not a pretrained SFT checkpoint or a
Spark recovery result.

The [separate activation-checkpointing control](../../experiments/reports/2026-10-05-dpo-activation-checkpointing.md)
uses the released runner's actual checkpoint-enabled CPU FP32 path. Seven tests
verify original-equation parity, same/fresh-process replay and wrong-mode refusal;
the off/on comparison passes its predeclared tolerance. Logical scored-forward
counts exclude recomputation dispatched during backward. This is not a measured
memory-saving, CUDA/BF16 or model-scale result. Legacy preference files without
source-group metadata remain explicitly unverified for source independence;
distinct IDs and disjoint encoded prompts cannot supply missing provenance.

The common invocation journal retains source/input/lock identity before hardware
preflight and actual parent-byte/interface identity before model loading. Driver
queries can be marked unavailable rather than erasing other evidence. Keyboard
interruption records its last stage; no in-process journal guarantees a SIGKILL
write. The identity and tiny local merge tests are recorded in the
[CPU report](../../experiments/reports/2026-10-04-run-identity.md), not Spark evidence.

Preserve an untouched SFT checkpoint and run matched controls under a separate
approved budget. Model-scale correctness, template compatibility, performance,
and sample quality remain unverified until an actual Spark smoke and learning
run are recorded. The bounded CPU report verifies the shared objective and mask
mechanisms; it cannot replace these hardware checks.

## Counted chosen-only recovery companion

The [chosen-only recovery API](../../src/dongxi_llms/chosen_sft_control.py)
has 36 focused CPU checks and two retained fresh-process 2→6 continuations.
Read its [measured report](../../experiments/reports/2026-10-05-chosen-sft-accounted-recovery.md)
alongside the original matched-control report: it changes neither the chosen
objective/draws nor the negative generation outcomes. A deliberately failed
later second forward remains charged after recovery. Reproduce the focused
controls in the locked CPU environment; they load only local tiny fixture models:

~~~bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -p test_chosen_sft_recovery.py -v
~~~

Do not invoke the legacy comparison CLI and assume these APIs were adopted:
it remains unaccounted by default. Actual counted generation/observer integration
and pretrained/CUDA smoke remain separate prerequisites for its campaign pilot.
