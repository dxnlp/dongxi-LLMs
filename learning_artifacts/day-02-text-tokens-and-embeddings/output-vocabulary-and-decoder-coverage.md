# Output vocabulary and decoder coverage

Recorded on 2026-10-05 from two retained native failures. This is a later
evidence bridge to [the Day2 BPE discussion](bpe-training-and-byte-coverage.md)
and [Chapter2](../../book/chapters/02-text-tokens-and-embeddings.md), not a new
mandatory lesson or a claim of live learner mastery. The active learner position
remains Day9. No model was loaded or run to prepare this note.

## The useful distinction

The learner previously established that a tokenizer with complete byte coverage
can encode unfamiliar Chinese text without introducing an unknown token. That
is a statement about the **input encoder**: it turns text into IDs that belong
to its mapping, possibly using many byte pieces.

It is not a guarantee that every numeric row in a model's output projection
has an entry in that tokenizer's decoder. Generation selects a categorical
index from a logit vector. Converting that index back to text is a separate
interface operation. Token IDs are addresses, not numerical meanings.

```text
Input text → encoder → known token IDs → embeddings → Transformer
                                                    ↓
                                          all output-row logits
                                                    ↓
                                            sampling → chosen ID
                                                    ↓
                                           tokenizer ID mapping
                                          /                    \
                                   mapped piece            unmapped ID
                                        ↓                       ↓
                                    text decode             ERROR record
```

For an illustrative four-row model, suppose the output indices are
`[0, 1, 2, 3]`, but the decoder only maps `0 → cat`, `1 → dog` and
`2 → EOS`. Selecting `3` is a valid output-vector index, yet the decoder cannot
look up its piece. This is not an input unknown word, nor an out-of-bounds
embedding index. The example is invented to explain the mechanism; its tokens
and dimensions are not the production model below.

## Three different vocabulary counts

The actual preparation binds the local cache snapshot for
`Qwen/Qwen3-0.6B-Base`, revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.
The following counts were independently read from its small cached JSON files,
whose complete-file SHA-256 values match the preparation bindings. No model
weight body or tokenizer runtime was loaded by this review.

| Object counted | Actual count or range |
| --- | ---: |
| `config.json` configured `vocab_size` |151936 output rows|
| `tokenizer.json` BPE base vocabulary, identical to `vocab.json` |151643 entries|
| `tokenizer.json` added-token records |22|
| Bare `tokenizer.json` union |151665 IDs,0–151664|
| `tokenizer_config.json` added-token decoder records |26|
| Union including that sidecar |151669 IDs,0–151668|
| Configured indices without an entry in that combined mapping |267 IDs,151669–151935|

The sidecar contributes four additional IDs,151665–151668, beyond the bare
tokenizer JSON. Consequently “the vocabulary size” can mean different things:
configured model width, BPE base entries, or the effective tokenizer mapping.
The native observed interface agrees with the combined151669-ID mapping; the
sampling records retain a support size of151936.

The selected ID151768 is below151936, so it is **inside the configured numeric
output range**, but absent from all the checked tokenizer mappings. The
configuration declares tied word embeddings; that declaration is not a new
runtime tensor or weight-tying check. These metadata and records do not explain
why the267 extra rows exist, how they were trained, or whether another decoder
would constrain them. Do not label their cause “padding” or infer a universal
rule for Qwen or other model families.

Primary metadata sources are the [cap32 preparation](../../experiments/reports/native-reasoning-base-chat-cap32-20261005-run-01/preparation.json),
[cap128 preparation](../../experiments/reports/native-reasoning-base-chat-cap128-20261005-run-01/preparation.json)
and [observed native interface](../../outputs/native-reasoning-base-chat-cap32-20261005-run-01/sample-1019/observed-interface.json).
The preparations retain the immutable local cache path and hashes; their
upstream correspondence was not remotely reverified by this review.

## Positive probability does not imply decodability

For finite logits $z_i$ and configured output width $V$, the model's
full-support distribution is

$$
p_i=\frac{e^{z_i}}{\sum_{j=0}^{V-1}e^{z_j}}.
$$

That denominator is over numeric output rows, not automatically over the
decoder's known-ID set. The actual sampler used temperature1, no top-k limit
and top-p1. It retained support151936, with raw-model and behavior logp equal
for every action in the failed trajectory.

At the failing prefix, the selected ID151768 had retained logp
$-14.50478178687033$, hence conditional probability approximately
$5.019417364343248\times10^{-7}$. This is an observed, finite, positive
probability for **that one ID at that one prefix**. It is not the total mass of
all unmapped IDs, a failure frequency, or evidence about the model's
pretraining mechanism. A low-probability action can still be sampled.

## What the two actual invocations retained

Both `native-reasoning-base-chat-cap32-20261005-run-01` and
`native-reasoning-base-chat-cap128-20261005-run-01` failed in `sample-1019`,
item `math-6`. The error record's derived attempt seed is
4213949931270527619. In both invocations the **24th retained action** is
ID151768, with `error_stage=decode`, `stop_reason=error` and
`truncated=false`. It is not a25th action after24 completed actions.

The two failing records have identical24 IDs and raw/behavior logp trajectories.
They are separate predeclared cap profiles, but the same prefix/seed failure,
not two independent rare-event draws from which to estimate a rate. Both fail
before either output cap. Increasing the cap did not remove a mapping mismatch.

| Retained invocation quantity | Cap32 | Cap128 |
| --- | ---: | ---: |
| Planned responses |100|100|
| Observed response rows, including the decode ERROR |40|40|
| Missing responses |60|60|
| Generated actions in those40 rows |1244|4491|
| Forwarded model positions in those40 rows |48378|377459|
| Actual child exit code |1|1|
| Actual child interval, seconds |56.54799546097638|132.66686874401057|
| Sampled minimum `MemAvailable`, bytes |122019704832|120977350656|

The one failing attempt itself retains24 generated actions,24 forward calls
and828 forwarded positions in each invocation. These are real incurred costs,
not refunded work. The source appends the selected ID, raw/behavior logp,
support size and action count before checking the decoder mapping. Its explicit
mapping guard records the error rather than silently discarding the action.

Both supervisor records retain no deadline/resource stop reason, no supervisor
failure and no cleanup or journal errors. The native child exited1 because the
required panel was incomplete/failed. Do not reinterpret this as an out-of-memory
failure, a successful100-response panel, or0/100 correctness:60 responses are
missing, and one of the40 observed records is an interface ERROR. Process
acceptance and answer grading remain separate questions.

The raw journals, including all24 IDs, selected likelihoods and costs, remain
unchanged. No valid-ID filter, resampling, retry or repaired pass was introduced.
The cap128 profile does not retroactively repair the cap32 failure.

Evidence:

- [Cap32 acceptance](../../experiments/reports/native-reasoning-base-chat-cap32-20261005-run-01/acceptance.json),
  [returned supervision](../../experiments/reports/native-reasoning-base-chat-cap32-20261005-run-01/returned-supervision.json)
  and [raw `sample-1019` journal](../../outputs/native-reasoning-base-chat-cap32-20261005-run-01/sample-1019/responses.jsonl).
- [Cap128 acceptance](../../experiments/reports/native-reasoning-base-chat-cap128-20261005-run-01/acceptance.json),
  [returned supervision](../../experiments/reports/native-reasoning-base-chat-cap128-20261005-run-01/returned-supervision.json)
  and [raw `sample-1019` journal](../../outputs/native-reasoning-base-chat-cap128-20261005-run-01/sample-1019/responses.jsonl).
- [Native generation source](../../src/dongxi_llms/reasoning_generation.py),
  `serialize_prompt`, `choose_token` and `generate_record`: known input-ID
  validation, full-vector sampling, append-before-decode and preserved errors.
  Both preparations bind source SHA-256
  `15d25f40776b5268cf1c41521fe5c35aabff2b095c5c51f0b68384cbe53ca5c0`.

## Why a possible future constraint would be a new experiment

Let $D$ be the decoder's declared known-ID set. A future decoder could choose
to constrain support and renormalize:

$$
p'_i=
\begin{cases}
\dfrac{e^{z_i}}{\sum_{j\in D}e^{z_j}}, & i\in D,\\
0, & i\notin D.
\end{cases}
$$

This is a different behavior distribution and support contract, not merely a
cosmetic text conversion. It would need explicit settings, likelihood/cost
accounting and newly bound evidence. It must not replace the retained
full-support result, silently retry until an ID decodes, or label the old
invocations passed. This note proposes no code change or new run.

## Reuse and the next deep question

Extend existing `CAND-ANIM-001`: show known input bytes taking the successful
encoder lane, then all output rows, full-support sampling and the selected
ID151768 branching to the missing-mapping ERROR. Keep the24 actions, their
likelihoods and costs visible instead of erasing the failure. The alternative
support constraint belongs in a visibly separate, hypothetical contract lane.
Any animation production remains separately approved on Mac Studio.

Suggested extension to `X-EVAL-001`, not a drafted or approved article:
“How a valid model output index can still fail to become text.” Link this
interface failure to the existing distinctions between decode errors, stopping,
missing responses and answer grading. Do not claim a new model-quality ranking.

Question for a later live discussion: **If a sampler only admits decoder-known
IDs, which probability distribution should its likelihood record describe—and
why must that decision be declared before comparing it with full-support
sampling?** No learner answer to this new question has been assessed yet.
