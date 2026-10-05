# Independent review: the completed common preference comparison

**Scoped approval.** No substantive defect was found in the completed
[comparison](native-preference-evaluation-20261005-run-01/comparison.json), its
[preparation](native-preference-evaluation-20261005-run-01/preparation.json),
[acceptance](native-preference-evaluation-20261005-run-01/acceptance.json), or the
432 completed response records. All 52 acceptance checks are true and exactly
match the comparison's checks. This is a review of one fixed local experiment,
not completion of every campaign maximum or a universal policy ranking.

Only this report was written. Review used bounded metadata/JSONL reads, source
inspection, source-file SHA checks and model-free grading/bootstrap replay.
No model or tokenizer was loaded; no model/corpus bodies were rehashed; no
producer, figure consumer, GPU job, acquisition, installation, Git mutation
or full CPU suite was executed. Full supervisor observation/event arrays were
not printed. The previously tested figure controls were not rerun here. A scoped
read-only `git diff --check` exited 0; all three report links resolve locally.

## Producer, interface and invocation joins

The three predeclared roles are the unchanged selected full-SFT400 checkpoint,
its chosen-only100 child, and its DPO100 child. The two trained exports have
accepted native exit-zero pilot receipts, exactly 100 ordered metric rows,
completed-update100 genealogy, unchanged reference/no reference gradients, and
clean retained work/I/O journals. Their original input roles match this common
preparation. All three declare the same selected full400 parent and tokenizer
semantics; none is substituted with Base, LoRA, or the separate Instruct branch.

The actual chosen-only and DPO histories consumed the same 400 sampled example
indices and have equal encoded training identities. This establishes the
matched fixture/sampler comparison, **not equal computation**: DPO additionally
consumes rejected responses and reference forwards. The common consumer does
not select a favorable checkpoint or change their training objectives.

All 14 current source/test hashes match the frozen common preparation. Key
identities are:

| Source | Frozen and current SHA-256 |
| --- | --- |
| `scripts/run_native_preference_evaluation.py` | `1c1bb663db33c4ee6b840d5732e4236cb3a03d41aded0c7d17c61643f7939023` |
| `src/dongxi_llms/reasoning_generation.py` | `15d25f40776b5268cf1c41521fe5c35aabff2b095c5c51f0b68384cbe53ca5c0` |
| `src/dongxi_llms/reasoning_evaluation.py` | `f163c5d03a87bc02474b48b3b55f651a10f8c0f0a6b9ab7c5974968961ca29da` |
| `src/dongxi_llms/dpo_lab.py` | `ac87d6c439f0a7ca21ca46c6c70854cdef0980f4e51d6da095440764252cad5e` |

The 18 frozen input roles, selected export declarations and their historical
byte receipts are joined through the preparation and generation identities.
All nine generation receipts have valid self-SHAs and exact source, items,
contract, config, command, selected checkpoint, tokenizer and interface joins.
Their saved contracts and observed tokenizer semantics match the common
contract; every completed row joins its own receipt and unique item/sample0.
This review checked these joins, not a new independent tensor-byte inventory.
The producer's source gates perform those actual byte/interface checks before
and after execution; their declarations are not treated as newly remeasured
weights in this review.

The original location splits contain 8/4/4 independent scenario groups. The
assistant splits contain 240/60/120 rows in 80/20/40 groups, pairwise disjoint.
The actual common location population is the four evaluation prompts, not the
four validation preference pairs. Saved split evidence records no encoded
prompt collisions.

## Matched generation and distinct likelihood precision

Generation uses the same saved custom chat template, context512, greedy64,
one sample, seed1010, no top-k filter, top-p1, and no prompt truncation for every
role. The recorded `thinking_mode` is `template-default`; it is not a native
Instruct thinking-on/off experiment. Actual model-loaded events report
`Qwen3ForCausalLM`, 596,049,920 parameters, CUDA and `torch.bfloat16` in every
panel. The generation source explicitly uses eager attention and uncached
full-prefix forwards. No optimized inference, forced-MATH kernel, FLOP or
throughput claim follows.

Preference likelihood scoring is different: the source loads policy/reference
FP32 weights with SDPA, enters BF16 CUDA autocast, and casts logits to FP32 for
log-softmax/gather. SDPA is an interface declaration, not proof of a particular
dispatched kernel. It sums the original body plus terminal-marker/separator
target log probabilities after exactly one causal shift; it does not report
length-normalized mean-token likelihood. Each validation pair has five chosen
targets and four rejected targets. Each role records eight policy and eight
reference forwards, 256 positions per network, and 36 valid targets across the
four pairs. Final pair events exactly match each completed summary.

| Policy | Mean chosen sequence logp | Mean rejected sequence logp | Mean **unscaled** reference-relative logp margin |
| --- | ---: | ---: | ---: |
| Unchanged full400 | -11.486888 | -25.312824 | 0 |
| Chosen-only100 | -0.003524 | -27.318045 | 13.488587 |
| DPO100 | -6.642914 | -91.838970 | 71.370116 |

The saved `reference_relative_margin` multiplies the final column by beta0.1.
These quantities must not be mislabeled as each other, as probabilities, or as
a generation-quality score. The unchanged arm's four zero relative margins
and log2 losses are an actual same-reference control. DPO's larger margin does
not imply the best greedy location result: its observed result below is 1/4,
whereas chosen-only gives 4/4.

## Grading and uncertainty replay

Independent model-free replay checked all 432 completed records and found zero
stored metric mismatches. The populations remain separate:

| Fixed population | Source groups | Unchanged | Chosen-only100 | DPO100 |
| --- | ---: | ---: | ---: | ---: |
| Independent location4: strict exact response | 4 | 0/4 | 4/4 | 1/4 |
| Assistant120: strict exact response | 40 | 120/120 | 120/120 | 120/120 |
| Annotated reasoning20: bounded parser-confirmed correctness | 20 | 5/20 | 6/20 | 6/20 |

Location and assistant correctness use trim-only, case-sensitive whole-response
equality. They do not use the generic grader's case/whitespace normalization.
Reasoning uses the existing bounded canonical answer parser, with explicit
extraction, rational/set/interval and unit boundaries; no Python evaluation,
symbolic rescue or last-number guessing occurs. Unsupported reasoning outputs
remain in the denominator: 14/14/12 by role. Supported-but-incorrect counts
are 1/0/2. An unsupported result is a verifier-coverage boundary, not an
independently established judgment of every answer's mathematical meaning.

All 432 responses end at the first recorded turn-stop token151645; none ends
at the cap, errors, or truncates. The stop token remains in raw token IDs/costs
and is removed from scored `response_text`. All unconstrained `format_policy`
checks are true; this is not evidence of demanding format adherence. Natural
termination is separate from correctness and rationale faithfulness.

Exactly 2,000 source-group resamples with seed1010 reproduced all 136 saved
panel correctness/format/natural/truncation comparisons and all eight
validation likelihood/margin comparisons, including every percentile endpoint.
Examples are chosen-only location +1 [1,1], DPO location +0.25 [0,0.75], and
both trained arms' reasoning arithmetic +0.125 [0,0.375]. These are percentile
uncertainty summaries on fixed diagnostic populations, not general capability
confidence or extra model samples. In particular, assistant120 has 40 source
groups, not 120 independent groups. No pooled cross-population score is made.

Nonblocking future reproducibility note: panel pairs are built from unsorted
sets, so seed alone does not guarantee the same cluster draw order in every
process. This caused no mismatch in any current saved interval; canonical
item/group ordering would make future resampling order explicit. No producer
source or existing result was changed to address it here.

## Actual supervision and cost boundaries

The returned and terminal receipts join exact commands, PIDs, status and native
exit0; the acceptance embeds the same returned receipts. All helpers finish
cleanup with no errors, and both queue feeders end for each child. Every
retained conflict probe has no conflict, is read-only/redacted, and reads no
environment or retained raw command lines. The minima below were independently
rederived from the retained samples, not described as continuous guarantees.

| Role | Native PID | Native child seconds | Minimum sampled available GiB | Generation tokens / forwards | Full-prefix positions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged | 295676 | 258.947192 | 107.657482 | 1,033 | 39,563 |
| Chosen-only100 | 296336 | 254.523008 | 107.597668 | 887 | 33,575 |
| DPO100 | 296999 | 246.119601 | 107.577000 | 859 | 32,505 |

Every generated record has matching attempted/completed calls and positions;
positions equal the exact sum of its successive prompt-plus-prefix lengths.
Scoring-token counts are zero; selected-token log probabilities are generation
readouts, not additional model-scoring passes. There are 2,779 generated tokens
including the 432 terminal stop tokens. Persisted per-token partial events are
the same evolving attempts, not extra independent completions or additional
generation-token spending.

The three distinct native child runtimes sum to 759.589801 seconds. The parent
reported an outer console duration of 1,086.313452 seconds; it is a different
scope, not independently reconstructed from the acceptance, and not added to
the child sum as model cost. Inner generation-record times, model loading,
repeated byte verification, pair scoring, panel summaries and supervisor times
overlap or cover different work. No GPU memory peak, FLOP count, optimized
serving speed, universal assistant ability or full32/128 reasoning publication
result is inferred from this common greedy64 comparison.
