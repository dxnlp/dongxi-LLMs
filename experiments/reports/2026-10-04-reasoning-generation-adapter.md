# Local checkpoint generation verification

The local-only evaluation adapter passed a real CPU HF checkpoint
load → forward → response ledger → offline CLI replay. The model is an original
randomly initialized decoder with 4,080 parameters and an authored sixteen-ID
tokenizer, not a downloaded or pretrained model. This verifies the measurement
path; pretrained evaluation and independent behavioral review remain pending.

The [premeasurement specification](../specs/2026-10-04-reasoning-generation-adapter.md)
precedes execution. [Final portable evidence](2026-10-04-reasoning-generation-adapter-final.json)
retains actual commands/exits, environment, input/source hashes, eight real
response records, complete contracts/interfaces, costs and six separately labeled
scripted stop/error controls. The
[first successful reference](2026-10-04-reasoning-generation-adapter.json) is
preserved as historical evidence; the final verification adds explicit control
records and fresh notebook evidence without overwriting that first report.

## Actual random checkpoint outputs

Two original development prompts each receive two seeded attempts. Both raw and
chat modes use the same random checkpoint, temperature1, full support, output
cap4 and context24. Chat's authored template explicitly passes the disabled
thinking-template keyword. Each attempt's random stream comes from its item ID
and sample coordinate, not its execution order.

| Mode | Records | Generated actions | Completed full-prefix positions | EOS | Turn stop | Task success |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 4 | 12 | 47 | 2 | 2 | 0/4 |
| Chat | 4 | 12 | 83 | 2 | 2 | 0/4 |

These are actual unforced model events. The arithmetic attempts decode as
`THINK [EOS]` and `+ [BOS] [EOS]`; the greeting attempts contain
`user 4 [TURN]` and `NO_THINK blue [BOS] [TURN]`. All IDs, including BOS and the
chosen stopping action, remain in the evidence. Only the final declared stop is
excluded from the separately stored grading text. The arithmetic outputs are
unsupported by the exact grammar; the greeting outputs are incorrect. No
successful answer is selected or substituted. Format validity is2/4 in each
mode because the greeting slice has no restrictive format requirement.

The larger chat forward-position count reflects its serialized prefix, not a
change in generated actions. Measured wall times are retained per record but
do not establish a throughput comparison: this tiny no-cache implementation
includes different warmup and serialization boundaries. Template disabling also
does not prohibit the random model from choosing the literal `THINK` token.
An interface keyword is not a learned behavioral guarantee.

The interface-bound checkpoint labels differ between raw and chat, while their
saved weight-file hashes agree. The label identifies weights plus the actual
tokenizer/serialization interface, not two separately trained models. Each run
rechecks actual local artifact and suite/contract bytes before declaring success.

## Instrument and failure checks

Fifteen focused generation tests and the existing fourteen grading/replay tests
pass. They cover real HF save/reload/forward execution, raw/chat tokenization,
the thinking-template keyword, deterministic per-item seeds, transformed
sampling likelihoods, greedy ties and rejection of silently ignored settings.
An external permuted vocabulary fails even when a contract was frozen from that
wrong tokenizer: the model's saved tokenizer provides a second compatibility
boundary. A changed template fails before weights load.

Declared controls distinguish first/generated EOS, a turn stop, EOS inside the
prompt, EOS-valued padding, output-cap truncation, context exhaustion and an
expired deadline. The final JSON stores six scripted records separately from
the eight unforced HF outputs. A forward failure and KeyboardInterrupt after
one chosen token preserve that token, its raw text, attempted/completed work
and failure stage. Invocation tests verify fsynced partial/final records and
the unattempted item/sample coordinates. Restarting remains a new invocation,
not an exact-resume claim.

An oversized raw response remains in storage while the bounded grader returns
INVALID. Stop, text, setting and interface tampering are rejected. A suite-byte
change after generation leaves all response rows but invalidates the invocation.
Existing output directories and frozen-contract files cannot be overwritten.
Setup failures retain actual identity and load stage, rather than fabricating
generated rows or a successful run.

The first development pass had one failing test: installed Transformers returns
a BatchEncoding by default for direct chat tokenization. Requesting
`return_dict=False` explicitly corrected the comparison with rendered-text IDs.
That failure is documented rather than silently treated as success.

## Environment and reproduction

Execution used `/home/dongxi/dgx-spark-dongxi/.venv/bin/python`, Linux/aarch64,
Python3.12.14, Torch2.13.0+cu130 and Transformers5.16.1. Every tensor and model
forward in this report used CPU float32; the installed CUDA build is not evidence
of GPU use. Both CLI replays equal the corresponding saved evaluation JSON.
Nine subprocess checks exit0, including the focused tests and book math check.

Fresh Day10 kernels execute all21 source code cells across four existing
notebooks and emit11 figures without source edits or execution errors. No new
notebook or animation was needed: the existing grading/response microscope and
the new Chapter7 section/lab explain the adapter boundary. The final JSON binds
the actual notebook manifest and its source hashes. Report-process RSS is a
Linux process measurement, not a subprocess/GPU peak or continuous memory trace.

Run a new bounded reference from the course root:

```bash
PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  /home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/verify_reasoning_generation.py \
  --report /tmp/NEW-generation-adapter.json
# Optionally add --notebook-manifest /tmp/ACTUAL-fresh-run/manifest.json.
```

The verification script writes a new temporary random-checkpoint workspace and
refuses to overwrite historical report JSON. Production generation uses
[generate_reasoning_records.py](../../scripts/generate_reasoning_records.py);
offline replay uses
[evaluate_reasoning_records.py](../../scripts/evaluate_reasoning_records.py).
The [Chapter7 lab](../../book/labs/07-evaluation-is-a-contract.md#local-checkpoint-generation)
documents complete settings and separate freeze/generate/replay commands.

## What remains unestablished

This completes the local adapter's bounded CPU checks, not all of DXI-02.
Genuine pretrained checkpoint comparison, independent behavioral review and
approved model-scale Spark profiling/supervision remain pending. CUDA and Mac
execution are unverified by this report. A forward-boundary deadline is not a
hard external process supervisor, and the full-prefix loop is not an optimized
inference framework. No acquisition, installation, paid API, service, GPU task,
Git mutation, publication or animation rendering occurred. Learner position
remains Day9; prepared material does not establish mastery.
