# A Qwen checkpoint trained on three instruction templates

This educational study measures fresh full and rank8 Q/V LoRA adaptation of
pinned Qwen3-0.6B-Base. Full tuning learns the three course tasks and their
message ending on the declared held-out-value panel. The matched LoRA recipe
improves development likelihood without successful complete answers. Neither
result establishes a general assistant, a universal method ranking or safety
for deployment. No public model release is implied.

## Actual artifacts and ancestry

Both runs start afresh from `Qwen/Qwen3-0.6B-Base`, revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; neither inherits disposable profile
or recovery weights. The
[full acceptance](../../experiments/reports/native-sft-full-pilot400-20261005-run-01/acceptance.json)
identifies the local standalone policy at
`outputs/native-sft-full-pilot400-20261005-run-01/policy`. Its observed
`model.safetensors` SHA-256 is
`c7a3932fc8602311dea07e295d3d0c2fe9c538ce2dfbde21eed9806c6cb21928`.
This is a weight-file digest, not a hash of the whole interface.

The LoRA training export is an adapter. Its
[separate FP32 merge and reload](../../experiments/reports/native-sft-lora-pilot400-merged-20261005-run-01/acceptance.json)
binds the exact Base and accepted400-update adapter. FP32 names merge arithmetic
and stored weights; publication generation is BF16. The eight-prefix tolerance
and bitwise reload checks do not prove equivalence to unmerged BF16 behavior.
The earlier BF16 merge mismatch remains a distinct failed experiment.

The original400-update full export is the predeclared preference-training parent,
not a parent selected after publication scores. Any later chosen-SFT or DPO
descendant requires its own actual export, ancestry and acceptance receipt.
The separate Instruct-based reasoning/RLVR branch and randomly initialized story
branch are not descendants of this assistant checkpoint.

The fixed chosen-only100 and DPO100 descendants now each have their own
accepted native recovery gate, fresh training pilot, export and genealogy.
Both start from this exact original full400 policy, not recovery weights, and
present 2,047 matched chosen targets. DPO additionally sees 1,600 rejected
targets and frozen-reference scores; objectives and compute are not matched.
Their [common comparison](../../experiments/reports/2026-10-05-native-preference-comparison.md)
binds all three accepted policies and their shared interface rather than
assuming ancestry from a checkpoint name.

## Data and interface

Training uses240 programmatically authored copy, two-word reverse and
labeled-value extract instructions. Development has60 items; publication has120 items in40
lexical-value source groups. Publication values are held out from course
training, but the three task templates are shared. This is not120 independent
tasks, proof of absence from upstream pretraining, or an estimate for arbitrary
user instructions.

The accepted custom-template SHA-256 is
`f26fd6284d6bc05d057a4b5800e2a268a85df6546edf484d979e23a51ed7886d`;
the semantic checkpoint-interface SHA-256 is
`e869c7e93ad366530f30cb62e78122577d84f0f95fe2dad5a0e0923d4d031f63`.
Tokenizer ID meanings, prompt construction, assistant-label masks and stops are
part of the contract. Generic EOS151643 and message-end151645 are distinct;
only a correct completed response supports the answering-and-stopping claim.

## Measured behavior

| Same original120-item publication panel | Base | Full400 | Merged LoRA400 |
|---|---:|---:|---:|
| Strict whole decoded answers |0/120|120/120|0/120|
| Natural message endings |0/120|120/120|0/120|
|64-token caps |120/120|0/120|120/120|
| Emitted tokens |7,680|760|7,680|

Generation uses identical actual prompt token IDs, greedy decoding, context512,
cap64 and BF16. Strict correctness compares the case-sensitive stripped whole
decoded answer after terminal-stop removal only; original token/text archives
remain available. Generic `format_policy=any` passes all120 responses in every
arm and is not evidence of accuracy.

The [source-bound comparison](../../experiments/reports/native-assistant-comparison-20261005-run-02/README.md)
retains all items, physical producer identities and source-group resampling.
Constant per-group outcomes give degenerate descriptive intervals; they do not
measure training-seed, recipe or new-task uncertainty. Development NLL separately
changes from4.061965 to0.000276425 for full and1.320048 for LoRA.

The later common greedy64 evaluation measures location extraction at 0/4 for
unchanged full400, 4/4 for chosen-only100 and 1/4 for DPO100. Both descendants
retain all 120 original instruction answers and natural endings; no response
hits its cap. The annotated 20-item reasoning diagnostic changes 5/20→6/20 for
both, adding only `math-10` without losing an initially correct item. Nine
seen/development items and eleven controlled held-out items remain separate;
the latter change 0/11→1/11. The bounded whole-output parser does not validate
all mathematical steps or faithful reasoning. These are actual BF16-loaded
generation observations, not broader competence or native-Instruct thinking
results. DPO's much larger four-pair likelihood margin does not imply more
strict whole location answers. No held-out score selects a new checkpoint,
decoding setup or preferred intervention.

## Recipe and cost boundaries

Both recipes use400 updates, seed1212, learning rate0.00002, microbatch1 with
accumulation4 and length256. They present9,321 valid training labels and process
63,378 training positions each. Full tuning updates596,049,920 parameters;
rank8 Q/V LoRA updates1,146,880. Their optimizer spaces differ, and the shared
learning rate was not tuned separately for the adapter.

External child durations are519.034/439.013 seconds. Measured CUDA peak
allocated bytes are6,027,842,048/2,055,486,464. These are not total unified-memory
use. Sampled host availability remains above the declared25-GiB reserve, but
does not establish a continuous floor. Validation and eight-prompt generation
are additional operations inside those child durations, not extra time to add
again. Merge, publication and offline comparison are separate invocations.
Snapshot clone/hash/serialization
counters overlap; do not sum them as independent disk traffic. The
[full study report](../../experiments/reports/2026-10-05-native-sft400-comparison.md)
gives the separate measured boundaries.

## Intended and excluded use

Use these local artifacts to study how label masks, stopping, trainable
parameter space and teacher-forced likelihood relate to generation. They are
not safety-reviewed services, factual assistants or general reasoning systems.
Transfer, adversarial robustness, privacy, broad multilingual capability and
human preference quality are unassessed. Source/data terms and redistribution
must be checked for any separately proposed release. Actual Mac execution and
cross-machine model reload are not established by this Spark evidence.
