# Actual Base and short-SFT response replay

Two genuine local pretrained checkpoints emitted all thirty planned responses
under one frozen development contract. Offline replay and card export preserve
their complete texts, tokens, failures, formats, stops, slices and paired
source groups. This closes a model-event evidence gap; it does not establish a
useful assistant, broad reasoning improvement or behavioral safety.

The [specification](../specs/2026-10-05-pretrained-evaluation-replay.md) was
declared before generation. The original fifteen authored development items
and fourteen source groups are unchanged. The contract SHA-256 is
`099520a69562abbe93bf45982c844f4658125e483331727d7b2f0afc4cbcdec7`.
Both runs use the exact saved instruction template, greedy decoding, one
response per item, a64-token output cap, context512, BF16 on Spark and the
same EOS/message-ending IDs. The thinking setting is `template-default`;
the template has no `enable_thinking` switch. No publication-test examples
or later400-update checkpoints were evaluated.

## What actually happened

| Observation | Acquired Base | Disposable20-update full SFT |
|---|---:|---:|
| Retained attempts |15 |15 |
| Automatic rubric/answer passes |0/15 |1/15 |
| Required format valid |12/15 |12/15 |
| Natural stops |0/15 |0/15 |
| Token-limit truncations |15/15 |15/15 |
| Returned generation tokens |960 |960 |
| Completed full-prefix forward positions |47,328 |47,328 |
| Actual external exit |0 |0 |
| Actual externally observed elapsed seconds |35.833090 |36.279153 |
| Minimum externally sampled MemAvailable bytes |123,204,276,224 |123,148,771,328 |

The Base checkpoint record identity is
`local-hf-sha256:b39c8fc43535a8b4fa6fa129bb6faad1494a1e3ffba8f980e57859079d7e3aa3`;
the short-SFT identity is
`local-hf-sha256:1e68a13306bb879c26730ff8217a6b85cff26ece4d071373f55c02375bccad73`.
These are local input-map identities, not individual weight-file hashes.
Their actual input identities, source digests, observed interfaces and raw
records remain in the respective
[Base](2026-10-05-pretrained-evaluation-replay/base/input-identity.json) and
[SFT](2026-10-05-pretrained-evaluation-replay/sft20/input-identity.json) bundles.
The Base's exact upstream revision and local acquired bytes were independently
checked in the [acquisition report](2026-10-05-base-tokenizer-sizing.md).
The SFT parent and training work are in the
[actual profile report](2026-10-05-native-base-profile.md).

Both generation processes completed successfully under an external900-second
deadline and25GiB reserve. Their returned supervision records retain owned
watchdog/helper shutdown. The table uses external whole-child timing and
sampled memory, not a claim of continuous minimum memory. Generation-only
recorded wall time is21.369555s for Base and19.908780s for SFT; it is a different
boundary, not an optimized serving benchmark. Offline grading adds no model
forwards. All1,920 generated actions and94,656 full-prefix positions remain
charged; the adapter intentionally recomputes prefixes without a KV cache.

## Why one automatic pass is not one good assistant answer

The SFT `benign-a` response contains the required phrases `terminate` and
`task manager`, so the frozen phrase rubric awards its one pass. It also
contains repeated `.nasa` fragments and ends mid-sentence at the token cap:

> task manager is a built-in tool in most operating systems that allows you to

Do not repair the text, remove the cap failure or retrofit a stricter metric
after inspection. Preserve the automatic score and add the independent
whole-response review as separate evidence. An inspectable positive/negative
phrase rule is useful for instrument testing, but it is not a semantic judge
of helpfulness or appropriate disagreement. Twenty responses are classified
`UNSUPPORTED` by the bounded parser. That reports extraction/grammar coverage,
not proof that the model cannot perform their underlying mathematics.

The [independent review](2026-10-05-pretrained-evaluation-replay/independent-behavior-review.md)
reads every complete response and passes107 offline/hash/instrument checks.
It finds genuinely partly useful conceptual content in `benign-a`, not just
keyword echo, while confirming artifacts and an unfinished ending. No response
is judged a clean completed requested answer. This is an explicitly unblinded
AI review by a different agent, not a human panel or calibrated semantic metric.
All23 grammar fixtures, six extraction cases, seven unchanged strict-integer
cases and the original grades/card/bootstrap replay also reproduce.

The paired automatic task-success difference is0.066667; the2,000-draw
source-group bootstrap with seed1010 gives the interval[0,0.214286]. Fifteen
items from fourteen intentionally authored groups are a small development
instrument. Neither the point difference nor that interval establishes broad
capability improvement, calibrated uncertainty or a successful training recipe.
All thirty responses failed natural termination under this exact contract.
Most items use an unconstrained `any` format policy. Consequently12/15
format-valid records do not mean twelve clean, useful or well-formed answers.

## Retained evidence and boundaries

The [complete paired raw ledger](2026-10-05-pretrained-evaluation-replay/paired-responses.jsonl)
contains the actual emitted records, not the authored fixture responses.
The [offline card](2026-10-05-pretrained-evaluation-replay/cards-01/model-card.md)
and full replay preserve all denominators and per-task results. Its immutable
card identity is
`557de73ef1237b01916f8e47fec072878f05f8f16a1ef38490a4fcc5acd3b8e4`.
The exporter's statement that independent review, training and broad safety
are unavailable concerns what its consumed inputs alone establish. Separate
producer/parent reports and the independent review supply their own narrower
evidence; they must not be silently rewritten into the original card.

This evaluation is not the staged campaign's separate math/publication panel,
the120-item instruction test, a selected400-update policy, human annotation
agreement, a broad safety assessment or release authorization. Negative
behavioral results are a valid completed measurement; they are not permission
to claim that the whole campaign or improvement goal is complete.
