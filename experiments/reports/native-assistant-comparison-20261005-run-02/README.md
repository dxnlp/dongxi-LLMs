# Original held-out values: full SFT learns the interface, this LoRA recipe does not

The actual frozen publication panel contains120 items: copy, reverse two words,
and extract a labeled value for each of40 held-out lexical-value source groups.
These values were not used in the original240-item training or60-item development
split. The three task templates are shared across splits; this is not a general
assistant benchmark or a human preference assessment.

The CPU-only [saved-record comparison](comparison.json) joins the actual
[Base](../native-assistant-publication-20261005-base-run-01/strict-results.json),
[full400](../native-assistant-publication-20261005-full400-run-01/strict-results.json),
and [merged LoRA400](../native-assistant-publication-20261005-lora400-fp32-run-01/strict-results.json)
producers. All completed with exit0,120 attempts and no response errors. The
[closing receipt](closing-bindings.json) verifies unchanged source, model-byte,
input and outcome bindings before and after this comparison. Historical
[comparison run01](../native-assistant-comparison-20261005-run-01/comparison.json)
is retained; run02 adds an explicit derived accepted-checkpoint-ID join without
changing a generation or its scores.

| Actual publication observation | Base | Full400 | Merged LoRA400 |
|---|---:|---:|---:|
| Strict case-sensitive whole decoded answers |0/120|120/120|0/120|
| Copy / reverse / extract strict answers |0/40 each|40/40 each|0/40 each|
| Natural message-end stops |0/120|120/120|0/120|
| Responses reaching64-token cap |120/120|0/120|120/120|
| Response errors |0|0|0|
| Emitted tokens, including retained stop tokens |7,680|760|7,680|
| Completed full-prefix forward calls |7,680|760|7,680|
| Completed full-prefix input positions |515,840|29,720|515,840|
| Summed response-attempt seconds |159.574|27.540|157.873|
| Summed synchronized forward seconds |99.402|10.310|98.825|
| Generation invocation seconds |172.469|44.316|169.626|
| External supervision seconds |173.432|45.216|170.574|

All three use the original custom template, identical actual prompt token IDs,
greedy seed1010, context512, cap64, EOS151643 and message-end151645. Generation
loads BF16 weights and explicitly requests eager attention. Raw token IDs,
special stop tokens, decoded text, likelihoods, truncations and overlapping cost
boundaries remain retained. Strict scoring compares the stripped whole decoded
answer after only a terminal stop token is excluded from the scoring text; it
does not extract a favorable answer prefix or discard a capped response.

Generic saved-response parser replay is reported separately. Its case-folded
text correctness happens to agree with strict correctness on these actual
records. The unconstrained `format_policy=any` assigns valid format to all120
responses in every arm, including the wrong verbose answers. That format result
does not establish answer correctness or successful termination.

The fixed paired source-group bootstrap uses the truly aligned120 greedy IDs
and40 source groups,2,000 draws and seed1010. Strict full-minus-Base accuracy is
+100 percentage points with descriptive interval[+100,+100]; LoRA-minus-Base
is0[0,0]; LoRA-minus-full is−100[−100,−100]. These degenerate intervals reflect
constant outcomes in this panel, not certainty about unseen tasks, alternative
recipes or training seeds. Physical checkpoint/input receipts remain distinct;
only a derived common logical-plus-actual-encoded contract is used for the
analysis rows. No physical receipt is rewritten to make the bootstrap agree.

## Likelihood and precision are separate observations

Both fresh400-update training runs saw9,321 supervised training labels and63,378
processed training positions under seed1212, learning rate0.00002 and the same
original train/dev/template bytes. Observed60-item development NLL fell from
4.061965 to0.000276425 for full tuning and to1.320048 for rank8 Q/V LoRA.
These are teacher-forced development measurements from the actual training
results, not NLL recomputed on the publication panel or on the merged artifact.
The adapter's improved likelihood therefore coexists with0/120 whole-answer
success and120 caps in the separately generated held-out-value panel.

The [actual accepted LoRA400 FP32 merge/reload](../native-sft-lora-pilot400-merged-20261005-run-01/acceptance.json)
used CPU FP32 merge arithmetic and stored FP32 weights. All eight fixed
development-prefix checks met the predeclared absolute0.002/relative0.001
tolerance, and the freshly reloaded merged logits matched bitwise. Publication
then loaded that explicit artifact in BF16, just like the other two arms. The
merge check is not a proof of equivalence to every BF16 unmerged-adapter forward,
nor a quality score; the earlier BF16 merge failure remains retained.

This establishes a controlled local outcome, not a universal full-versus-LoRA
ranking. A common learning rate does not optimize two different trainable
subspaces. Rank, Q/V targets, learning rate and seed were fixed, not retuned after
inspection. The lower generation cost of full400 partly follows from its short
naturally ended answers, rather than an inference-engine speed advantage: this
transparent producer recomputes the whole prefix without KV caching. Logical
positions, call counts and times overlap and must not be summed into FLOPs,
unique work, unified-memory consumption or an optimized throughput claim.

## Reproduction boundary

The completed CPU command was:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/dongxi/dgx-spark-dongxi/.venv/bin/python \
  scripts/compare_native_assistant_evaluations.py
```

It loads no model or tokenizer, launches no GPU process, acquires no data and
uses no network or Git operation. The output is exclusively created; rerunning
does not overwrite it. The fixed publication input IDs remain `run-01`; only
the comparison output ID is `run-02`. The19 authored CPU tests passed in0.400s.
The [preparation](preparation.json) binds the implementation and tests:

- Script SHA256: `62f34b7ec6ab48ce2db2e4858fde242e6e8c73e3edf820156dc3b6514d966033`.
- Test SHA256: `a02cfd43989d152b087b14f13f426dd3d95efd88a84a34cfc4b6bf02bc299296`.

This report does not certify learner mastery, actual Mac execution, general
assistant safety or permission to publish model weights externally.
