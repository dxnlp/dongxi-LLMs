# Local checkpoint generation and response evidence

This specification precedes the bounded CPU verification of a local Hugging Face
generation adapter. The objective is to connect a frozen evaluation contract to
actual model events without acquiring a model, using a service, or running a GPU
experiment. The positive control is an original, tiny, randomly initialized
decoder and tokenizer saved locally. Its outputs are not pretrained capability
evidence. Independent behavioral review remains pending.

## Frozen interface and predictions

Generation consumes an existing suite and a frozen contract. An explicit
preparation command may freeze that contract from settings and the actual local
tokenizer, without loading model weights. Raw input and chat serialization are
different interfaces. A chat contract records the complete template and the
thinking-template keyword; a raw contract has neither an implicit template nor a
thinking toggle. Record temperature, top-k, top-p, greedy versus sampling, output
and context caps, declared EOS and turn-stop IDs, dtype, device and attempt count.
Greedy contracts reject unused sampling transformations rather than ignoring them.

Hash actual input files, local checkpoint/tokenizer artifacts and the observed
tokenizer encoding and special-token semantics. Compare a separately supplied
tokenizer with the one saved alongside the model, and compare both with the frozen
interface before loading weights. No remote identifiers, downloads, pickle
weights, automatic device mapping or custom remote model code are accepted.

Predict that a compatible locally saved random HF model will produce replayable
token records. A permuted same-size tokenizer and a changed template must fail
before model loading. Sample seeds derive from the frozen seed, item ID and sample
coordinate, so changing iteration order cannot change an item's random stream.

## Execution and measurement boundaries

Implement a transparent single-sequence autoregressive loop using actual HF
forward calls, without KV-cache optimization. Greedy chooses the first maximum;
sampling applies temperature, top-k, renormalization, top-p and renormalization,
with stable action-ID tie order. Never send reference answers or grading results
to prompt serialization or sampling. Every sampled stop token remains in the raw
token ledger and token cost; raw decoding preserves special tokens.

Classify chosen EOS, chosen turn stop, output cap, context boundary and execution
failure separately. An EOS inside the prompt or an EOS-valued padding ID is not a
generated termination event. If the context budget ends first, do not call it an
output-cap stop. Preserve partial responses on forward errors or interruption.

Use a new output directory only. Persist input identity before model loading,
append and flush/fsync progress and final response events, and never overwrite
existing evidence. An interrupted invocation retains saved progress; restarting
is a new invocation, not a claim of exact resume. Store every attempted response
and explicitly list planned coverage if setup fails before attempts begin.

Measure prompt/generated tokens, attempted and completed forward token positions,
forward calls, total per-record wall time and forward time. These counts are not
FLOPs, billing units or valid-answer tokens. Offline grading uses no model scoring
tokens. Time a CUDA device only if explicitly selected and allowed, synchronize
that device at measured boundaries, and enforce the platform host reserve before
loading; this verification never selects CUDA. CPU times are host wall times,
not hardware-independent performance comparisons.

## Acceptance checks

- Execute a real tiny random saved HF model and local fast tokenizer on CPU;
  preserve its actual IDs, decoding, measured costs and replay scores.
- Verify seeded sampling reproducibility and order-independent attempt seeds.
- Exercise raw/chat and thinking-template validation, same-size vocabulary
  permutation, template changes, context exhaustion, first-token EOS/turn stops,
  EOS-as-padding, nonfinite logits and errors after a partial continuation.
- Keep all stop IDs and costs, reject silently ignored greedy settings and
  unknown generation settings, and preserve oversized raw text while the bounded
  grader declares it invalid rather than discarding it.
- Verify a new-only output journal, partial interruption evidence, load-stage
  failures and mutated input-artifact detection. No claim of exact resume.
- Replay records without importing/loading a model; existing strict integer
  grading and authored response fixtures remain unchanged.

Record commands, actual check exits, source/input hashes, interpreter/packages,
tiny model parameter count, measured durations and limits in a new report. No
pretrained run, model acquisition, GPU task, installation, server, independent
human study, Git mutation or animation production is part of this specification.
