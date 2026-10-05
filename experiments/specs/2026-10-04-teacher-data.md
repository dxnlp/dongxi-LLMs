# Teacher attempts and matched rejection sampling

This specification and `fixtures/teacher-data/protocol.json` are frozen before
teacher collection or student fits. The recipe is an original bounded CPU
control, not a pretrained/API teacher, human-feedback study, or general
distillation benchmark. No recipe will be changed after held-out results.

## Inputs and identities

Use twelve train, four development, four test, and four control prompts in the
protocol. Half of each split uses two response words; half uses three. Source
groups are the actual copy/reverse operation plus color tuple, independent of
polite prefixes. Gate normalized prompt and actual encoder-ID aliases before
collection; no group may cross splits. The fixed teaching vocabulary is the
existing instruction-data interface, not a newly learned tokenizer.

The programmatic teacher has eight declared sample slots per train prompt:
correct, wrong/same length, wrong/one extra word, empty, overlength, unsupported
symbol, no END, and a declared failure. Only a recorded error receives one retry,
which emits a duplicate correct answer. This produces nine attempts per prompt
when collection completes: 108 total. The teacher performs no model forward or
API call. Its raw trace/final text, symbolic final IDs when representable, stop,
error, measured elapsed time and serialized-word costs are retained separately.
Do not call these inference token costs or billed units.

Attempt identity includes frozen contract, item ID, sample slot and retry index.
The contract fingerprints actual fixture/source bytes, vocabulary, serialization,
teacher, sampler, verifier, filter and student settings. Persist execution-start
events and fsynced result lines. Resume cannot duplicate committed attempts or
silently accept a changed contract. An explicitly requested partial-tail recovery
first preserves the exact corrupt bytes and recovery reason, then truncates only
that uncommitted tail. Complete corrupt records reject recovery. A started but
uncommitted execution may be retried, with that repeated start and unknown lost
cost disclosed; this is no-duplicate records, not exactly-once physical calls.

## Frozen candidate and selection contract

Filter only train attempts. Retain every applicable reason for error, empty,
overlength, unsupported/malformed answer, missing END, source leakage, and
within-prompt exact normalized duplication. Acceptance **does not require
correctness**. The expected candidate pool contains correct and wrong well-formed
responses. Pool and per-attempt audit identities are frozen before selection.

The fixed training-task verifier computes copy/reverse correctness from the
training prompt and response. It cannot consult the fixture reference field,
held-out prompts, teacher mode, or a hidden correctness label. Top per prompt
chooses one candidate by correctness, then shorter symbolic supervised length,
then stable attempt identity. Compare with one seeded-random accepted candidate
per prompt, using seed912 and item-keyed streams. Also fit a predeclared random
sensitivity restricted to the top answer's supervised length for that prompt.
Report availability rather than inventing a fallback if this stratum is empty.

All three arms use the identical frozen pool and scorer. The primary comparison
matches twelve samples and all twelve prompts. It may not match supervised-token
budget; report final-answer/END counts and actual eighty-update exposure. The
length-stratified sensitivity matches those counts where available. Do not
change objective normalization to disguise an exposure difference.

Audit global top3/top6/top12 separately, including source, difficulty, length,
correctness, and coverage. Their example budgets differ and they are not extra
trained student arms. A global ranking can exclude prompts while a per-prompt
ranking guarantees coverage; show actual selected IDs rather than assuming it.

## Student protocol and independent evaluation

Reuse `instruction_data_lab.encode_messages/collate`, `sft_lab.token_loss_sum`
and `decoder_lab.TinyDecoder`. Serialize system `short answer`, user prompt,
assistant final answer and END. Supervise assistant body and real END, not roles,
prompt or right padding. Trace is provenance, not a training target.

Randomly initialize one-layer width24/heads3/MLP48 decoders with seeds
1101/1102/1103, float64 CPU, no dropout, maximum context32. Pair initialization
across top, random and length-random arms. Train each for80 full-batch AdamW
updates, lr0.015, weight decay0, clip1, using global valid assistant-token mean.
Keep every update and first-backward evidence. Baselines take zero updates.
No best checkpoint or seed selection, pretrained acquisition, GPU, services,
installations, Git mutation or publication is included.

Before and after training, retain one greedy and four independently item/sample-
seeded full-vocabulary temperature1 responses per prompt, max6 new tokens. END
is the learned stop, not a forced grammar action. Record raw IDs, input IDs,
errors, natural termination, cap, correctness, format, and measured full-prefix
forward/token/elapsed costs. Gold references are read only by this evaluation,
not by teacher-data selection. Test/control are never used for fitting or recipe
selection. Shared token vocabulary limits the held-out claim to new combinations
and the declared polite-prefix control, not new vocabulary or general reasoning.

## Acceptance

Test immutable identity, partial collection/resume, duplicate/corrupt/tail handling,
error retry, all filter reasons, split/source/actual-ID leakage, pool and seed-order
invariance, length-stratum availability, exact mask/causal shift and END retention,
finite student gradients, frozen initial controls, and independently regraded
held-out generations. Preserve negative outcomes and unmatched budgets.
Execute the new Day11 notebook in a fresh locked CPU kernel, export and inspect
its data-backed figures, verify focused tests/math/source hashes, and record
actual commands/environment. Existing learner cells remain untouched. New math
and flow animation opportunities are proposals for Mac-side production only.
