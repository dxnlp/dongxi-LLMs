# Equal exposure does not imply equal adaptation

![Actual fixed-recipe training NLL and gradient norm versus cumulative training labels](native-sft400-comparison-figures/learning-curves.png)

The [plot input receipt](native-sft400-comparison-figures/inputs.json) binds both
actual400 metric streams. These are training curves, not publication accuracy.

Two fresh pinned Qwen3-0.6B-Base runs completed the original400-update recipe
on Spark. Full tuning and rank8 Q/V LoRA consumed the same original instruction
train/dev bytes, custom template, seed1212, learning rate0.00002,
microbatch1/accumulation4 and length256. Neither inherited profile or recovery
weights. Their actual acceptance receipts are
[full400](native-sft-full-pilot400-20261005-run-01/acceptance.json) and
[LoRA400](native-sft-lora-pilot400-20261005-run-01/acceptance.json).

| Observed boundary | Full | Rank8 Q/V LoRA |
|---|---:|---:|
| Successful updates |400|400|
| Successful training labels |9,321|9,321|
| Processed training positions |63,378|63,378|
| Initial development NLL |4.061965|4.061965|
| Final development NLL |0.000276425|1.320048|
| Eight fixed development responses exactly correct |8/8|0/8|
| Eight fixed development responses naturally message-ended |8/8|0/8|
| Development responses exhausting64-token cap |0/8|8/8|
| Trainable parameters |596,049,920|1,146,880|
| CUDA peak allocated bytes |6,027,842,048|2,055,486,464|
| External child seconds |519.034|439.013|
| Minimum internally sampled MemAvailable, GiB |105.034|111.404|
| Serialized snapshot bytes, five committed saves |17,055,101,053|7,577,555,233|
| Actual child exit |0|0|

Successful training labels exclude the720 development-label presentations.
Processed positions include the padding processed by training, not evaluation
or generation. Snapshot serialization/hash/clone counters overlap the same
payloads: their sum is not total disk traffic. External supervision includes
startup/exit handling; runner timers are517.316/437.312 seconds and have a
different boundary. Allocated CUDA peaks are not total unified-memory usage;
sampling cannot establish a continuous lower bound on free memory.

All declared runtime, finite-state, completed-journal, committed-snapshot and
export checks passed. Both have snapshots0/100/200/300/400; LoRA exports an
adapter rather than pretending it is a standalone full policy. The full export
has an actual byte-bound tokenizer/template/stopping interface and is the
predeclared downstream preference parent. The LoRA full merge and original
120-item publication evaluation are separately executed steps, not results
supplied by these eight prompts; their observations follow below.

The full run's final responses are four or seven generated tokens, including
their message-end token. The LoRA run often prints the requested prefix before
continuing into unrelated text. Answer likelihood improved without successful
whole-response behavior. Raw completions are retained in each acceptance
receipt and ignored model-output directory; no suffix was removed to improve
these development scores.

## Original held-out-value publication comparison

The [actual120-item/40-source-group comparison](native-assistant-comparison-20261005-run-02/README.md)
now joins unchanged Base, accepted full400 and explicitly FP32-merged LoRA400.
All three use the original template and identical actual prompt token IDs,
greedy seed1010, context512 and cap64. Generation is BF16 with an explicitly
requested eager backend in all arms; FP32 names the LoRA merge arithmetic and
stored artifact, not its publication generation precision.

| Observed publication boundary | Base | Full400 | Merged LoRA400 |
|---|---:|---:|---:|
| Strict whole decoded answers |0/120|120/120|0/120|
| Copy / reverse / extract |0/40 each|40/40 each|0/40 each|
| Natural message-end stops |0/120|120/120|0/120|
| Responses exhausting64-token cap |120/120|0/120|120/120|
| Response errors |0|0|0|
| Emitted tokens / completed forward calls |7,680|760|7,680|
| Completed full-prefix positions |515,840|29,720|515,840|
| Summed response-attempt seconds |159.574|27.540|157.873|
| External supervision seconds |173.432|45.216|170.574|

Strict scoring is case-sensitive stripped whole-answer equality after a
terminal special stop is excluded only from scoring text; raw tokens and text
retain it. No favorable prefix is extracted. Generic case-folded parser replay
is separate, although correctness agrees on these records. All responses pass
the generic `format_policy=any` check, showing why unconstrained format validity
is not answer accuracy. The full arm's actual terminal token is151645; Base and
LoRA have no natural stops. All three actual producers exited0.

Fixed2,000-draw seed1010 paired resampling retains all120 greedy IDs within40
source groups. The descriptive strict difference intervals are full−Base
+100 percentage points[+100,+100], LoRA−Base0[0,0], and LoRA−full−100[−100,−100].
Constant outcomes make these intervals degenerate; they do not capture
between-run variability or uncertainty about new task templates. Publication
values are held out, but the three deterministic task templates are shared with
training. Source/input/model/outcome receipts are byte-bound before and after
the CPU comparison, and distinct physical receipts are never declared identical.

The [actual LoRA400 merge acceptance](native-sft-lora-pilot400-merged-20261005-run-01/acceptance.json)
passes all eight fixed-prefix FP32 checks at the predeclared0.002 absolute/0.001
relative tolerance and bitwise fresh-reload logit equality. This is a narrow
numerical/interface handoff check, not proof of equivalent BF16 adapter behavior
or successful answers. The development NLL values above remain observed
pre-merge training metrics, separate from publication generation. Full400's
short correct outputs reduce the measured full-prefix work; the transparent
uncached producer is not an optimized inference throughput benchmark.

This is a fixed-recipe, single-seed, tiny synthetic-interface intervention.
It shows a useful local capacity/optimization contrast, not that full tuning is
universally better than LoRA or that either model is a general assistant. The
shared learning rate may be unsuitable for the constrained adapter; rank,
targets and learning rate were not retuned after inspection. These are
development observations and the separately measured held-out-value panel remain
different populations; neither establishes performance on unrelated templates.
The older20-update failures, first recovery conflict and BF16 merge mismatch
remain valid historical evidence. A400-update success does not erase them.

The [continuation specification](../specs/2026-10-05-spark-campaign-continuation.md)
keeps acquisition, branch ancestry, runtime and evaluation limits explicit.
This report does not advance learner mastery, claim actual Mac execution or
authorize Git/model publication.
