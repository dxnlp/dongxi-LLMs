# DongxiGPT TinyStories learning checkpoint: evidence-limited model card

This card describes the actual Day9 learning run, not a public release or a new
controlled comparison. Prepared post-training branches are not descendants of
this checkpoint. The authoritative measurements are the
[completion report](../../experiments/reports/2026-09-14-tinystories-learning-result.md)
and its [portable JSON](../../experiments/reports/2026-09-14-tinystories-learning-result.json).

## Model and intended use

DongxiGPT is a randomly initialized66,638,848-parameter modern decoder trained
on the prepared TinyStories corpus on DGX Spark. It uses the GPT-2 tokenizer,
not pretrained GPT-2 model weights. Intended use is this owner's educational
study of next-token learning, loss, generation and experiment evidence.

It is not an instruction assistant, reliable factual source, general reasoning
system or safety-reviewed deployment. Do not use its outputs for consequential
decisions. No public serving, weight upload or external publication is implied
by this card. Architecture, recipe and prepared-data identity are recorded in
the portable JSON; source/data licensing and redistribution must be reviewed
for any separately proposed release.

## Actual checkpoint identity and lineage

The measured final checkpoint is reported at
`outputs/day09-learning-01/update-014000.pt`. Its parent is random initialization;
the run contains no SFT, DPO or RLVR stage. The portable archive hashes the raw
files used for its report but contains neither model weights nor a complete
token archive. This card does not invent a final weight digest or claim a
cross-machine reload. Actual local checkpoint-byte verification is a separate
gate before reuse, handoff or publication.

CPU mechanism checkpoints elsewhere in the course have their own interfaces
and tasks. They must not be attached to this genealogy merely to make a complete
training-stage diagram. The assistant and reasoning branches remain separate.
The [original staged preparation](../../experiments/reports/2026-10-04-staged-spark-campaign.md)
retains its then-null checkpoint fields as historical evidence, not a current
inventory. Later [actual full and LoRA training](../../experiments/reports/2026-10-05-native-sft400-comparison.md)
does not create descendants of this September story checkpoint.

## Training and cost boundary

The run completed14,000 updates and48,839,975 valid target presentations.
It processed229,376,000 tensor positions; padding is not additional language
supervision. Trainer duration was12,102.9132 seconds, with actual child exit0.
Earlier corpus preparation is outside that timing boundary. The original
prepared corpus contains1,792,647 documents and390,708,926 valid targets, so
training did not present every prepared target once.

Minimum sampled host availability was104.7032 GiB against a25-GiB reserve,
sampled at0.2 seconds. CUDA peak allocated/reserved memory was11,293,046,784 /
12,490,637,312 bytes. These are distinct measurements, not quantities to add
into physical RAM use. Sampled availability is not a continuous guarantee.

## Evaluation and supported claims

The fixed development selection contains512 windows and107,264 valid targets,
chosen with seed909. Its token-weighted NLL fell from10.904913 at initialization
to1.674315 at the final observation. This supports improved prediction on that
development slice under the declared measurement, not a broad capability claim.
The online last500 batch mean1.661964 is not a matched frozen-checkpoint
train/development comparison.

Archived examples show recognizable English story constructions alongside
repetition, malformed language and unexplained role changes. The compact
sample subset was retrospectively selected by checkpoint coordinate and first
configured prompt, not by quality. Those inspected examples are not blinded
coherence ratings or an untouched publication test. EOS in the final two
first-prompt examples demonstrates stopping in those cases, not story quality.

## Known failures and missing evidence

Repetition alone does not prove overfitting. The original run lacks a matched
frozen-checkpoint train/development gap, seed replication, controlled trained
recipe comparison and blinded multi-prompt coherence scoring. Exact-normalized
deduplication is not a near-duplicate contamination audit. Broad behavioral
safety, robust transfer and rationale faithfulness are unassessed.

The later [publication-opening audit](../../experiments/reports/2026-10-05-story-panel-audit.md)
has actually inspected the pinned corpus under explicit exact and lexical-near
definitions. It is not a retrospective blinded quality score for this September
checkpoint. A [fresh matched learning-rate comparison](../../experiments/reports/2026-10-05-native-story-first400-comparison.md)
now has two accepted first 400 tranches, four publication children, 192 actual
continuations and two separately blinded AI-reader reviews. Both fresh arms
present 1,389,548 valid training targets in 6,553,600 positions and retain the
original 14,000-step horizon. Their fixed 64-window development NLLs at 400 are
3.388772 (control) and 3.652813 (half-rate), on 13,132 valid labels, not the
September selection. Publication uses automatic SDPA/BF16 forward autocast,
distinct from the deterministic MATH-only training/recovery entry. Control's
45/48 EOS stops coexist with a mean ending score of only 0.020833 on a 0–2
scale; half-rate stops at EOS in 25/48 and has ending mean 0. The actual five-axis
ratings retain 56 candidate disagreements and do not establish human consensus
or reliable coherence. Their paired intervals resample twelve openings, not
training seeds. Neither fresh arm inherits this historical checkpoint, and the
288 planned cells at 4000/8000/14000 remain missing. See that report for current
execution and quality evidence rather than extending this card's claims.
