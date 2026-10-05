# Local response-distillation transfer — prepared, not executed

This is a bounded follow-up protocol, not measurement, acquisition permission or
a claim that local pretrained checkpoints are available. The source validator
`prepare_local_transfer()` never loads a model and rejects `execute=True`.
Its unit tests inspect dummy files in temporary directories only; they do not
validate a real safetensors file or model compatibility.

## Inputs that must be frozen before authorization

Select two distinct existing local full/explicitly merged checkpoints. Record
actual weight-file digests, model configuration, separate saved tokenizers,
separate chat templates and end-marker policies. The source validator currently
hashes config/tokenizer bytes and lists weight filenames; actual weight identities,
template behavior and hardware compatibility remain explicit pending gates.
There is no download or installation fallback.

Freeze unique source-group IDs with disjoint train/test membership, including
multiturn/template siblings in one split. Write original prompts, an independently
scored final-answer contract and an explicit review rubric for printed reasoning.
Do not generalize the tiny sum/parity STEP checker to natural-language rationale
verification or internal causal faithfulness. Preserve teacher initialization and
training genealogy if the local checkpoint was produced by another course run.

## Bounded proposed run

The default proposal is four teacher attempts per item, at most 64 generated
tokens each, and 80 student updates. The validator permits at most eight attempts,
256 tokens and 200 updates, with at least 25 GiB available-host reserve. Freeze the
chosen values, teacher/student seeds, decoding support and exact source IDs before
any collection or fit; these upper bounds are not an estimated affordable GPU run.

Keep every attempt, exception, malformed ending, reviewer verdict and rejected
response with its work cost. Choose a predeclared format/review rule without using
the final test split. Missing eligible source coverage is explicit; no silent
substitution or truncation. Derive answer-only versus complete-response examples
from the same selected parent records and disclose unequal target-token exposure.

Teacher text must be re-encoded with the student's tokenizer. Teacher token IDs
are not portable across vocabularies. Check exact prompt-loss exclusion, one
causal shift, response EOS inclusion and padding before training. The complete
response arm and answer-only arm begin from identical student initialization.
Evaluate original student, teacher and both fitted students on a frozen independent
raw-generation suite; final correctness and printed-rationale review are separate.

## Execution gate and costs

A new specification must bind actual local checkpoint interfaces and a timed
Spark hardware profile. Separate teacher loading/fitting/data-generation from
student fitting/evaluation and selection/review work, including failed attempts.
Declare supervised tokens, attempted generated tokens, throughput timing boundaries,
actual peak device memory and sampled host reserve. A smaller parameter count
does not establish deployment speed or total distillation cost savings.

Authorize the run separately after finite-value, local reload, no-truncation,
source-independence, checkpoint/restart and memory-reserve gates. External
supervision is needed for a hard timeout; cooperative checks cannot interrupt one
blocking GPU operation. No execution, model loading, GPU profiling, installation,
download, service or publication occurred while preparing this protocol.
