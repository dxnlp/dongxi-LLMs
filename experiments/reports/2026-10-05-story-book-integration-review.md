# Story evidence integration: text and exercise review

Read-only review on 2026-10-05 checked the new story material in Chapters 6,
7 and 15, their worked solutions and labs, the
[first400 report](2026-10-05-native-story-first400-comparison.md), the
[measurement archive](native-story-publication-20261005-01-archive/acceptance.json),
the [supplied-rating consumer](native-story-publication-20261005-01/ratings-evaluation-01/report.json)
and the [current story card](../../docs/model_cards/day09-dongxigpt.md).
The current handoff was checked for the same summary claims. Only this review
file was written; the reviewer did not edit book or producer files.

No remaining material scientific inconsistency was found in the reviewed current
story prose after the owner's two localized corrections below. This is text and
schema inspection, not another semantic rating, the main independent numerical
review, final campaign acceptance or evidence of learner mastery.

## Claims that match the retained evidence

- Each fresh arm reaches its declared 400-update boundary with 1,389,548 valid
  training targets and 6,553,600 processed training positions. The unchanged
  14,000-update horizon/200-update warmup remains unfinished. The September
  completed run is a separate random-initialization lineage, not either arm's
  parent, missing later checkpoint or substitute matched control.
- Fixed NLL uses 64 matched windows per split: 13,285 training-source and 13,132
  development targets, each in 65,536 physical positions. Control/half-rate
  update400 development NLLs are 3.388772/3.652813. These are target-weighted
  slices, not whole-corpus losses, proof every selected training-source window
  was presented, or an overfitting diagnosis.
- Publication declares FP32 loaded weights, CUDA BF16 forward autocast, FP64
  likelihood arithmetic and automatic causal SDPA. This is distinct from the
  deterministic MATH-only training/recovery entry; no selected per-call kernel
  or cross-backend bitwise equivalence is claimed.
- Four available checkpoints supply 192 continuations, 48 each. Initializations
  have no natural EOS; control400/half-rate400 have 45/25. Their ending means
  remain 0.020833/0 on the 0–2 rubric. Natural stopping is not plot resolution,
  and freedom from repetition is not general language competence.
- Both separate AI submissions cover 192 candidates with no abstentions.
  There are 56 candidates with at least one disagreement and 64 dimension-level
  disagreements, including 41 on repetition. The declared mean is not a third
  review or human consensus. Separate instances do not establish independent
  underlying models or perfectly successful blinding.
- The update400 half-rate-minus-control causal-continuity delta is −0.104167,
  with interval approximately [−0.177083, −0.010417]. The retained consumer uses
  800 draws/seed1010 and twelve source-opening groups, carrying four recipes
  together. Recipes and raters do not create additional independent openings;
  these intervals do not estimate between-training-seed or human-population
  uncertainty.
- Original coverage remains 480 planned/192 actual/288 missing cells. Later
  4000/8000/14000 comparisons are incomplete with null equal-recipe means,
  not zero scores, interpolated quality or forecasts.
- The portable archive retains NLL, child/supervision documents and original
  metadata/coverage/source bindings, not ratings, model weights or corpus bodies.
  Reviews are retained by the separate rating report/receipt. Archive semantic
  digest `5c710289f621a2915c1136b7ad3c341de4dc5ac7d2e3ba7d5873adac639201ac`
  and acceptance's file-byte digest have different definitions.

## Corrections and instructional flow

The owner corrected Chapter15's original statement that the measurement archive
retained reviews; current prose separates the two evidence locations. The owner
also corrected the Chapter15/current-summary wording that called 56 a count of
axis disagreements. The retained consumer confirms 56 candidates versus 64
dimension-level disagreements. No ratings or empirical artifacts were changed
by these prose corrections.

All 199 local Markdown-link occurrences across the 13 inspected text files
resolve. Chapter6 exercise21/solution21 leads into the fresh-tranche evidence lab;
Chapter7 exercise23/solution23 connects the authored empty-rating control with
actual supplied reviews; Chapter15 solution10 and its lab preserve the distinct
story and pretrained-assistant genealogies. The Day8 Chapter6 foundation lab
remains separate from the later Day9 evidence-reading lab.

The added Lab6 metadata block compiles, and its referenced archive/coverage/rating
fields exist in the retained documents. Its block SHA256 is
`3e8a7edcfdc14d3f3177ec2ff190ee8df37ba39688db4d6306fd195006b7c367`.
The block was not executed as a new consumer or empirical run. No notebook,
bootstrap, plotting, model, rating collection or generation was rerun.

The owner also resolved both optional editorial clarifications after review:
the [Lab6 introduction](../../book/labs/06-reading-a-pretraining-run.md) and
Chapter6 routing now include sections6.14–6.21; the
[Lab15 prompt](../../book/labs/15-distill-evaluate-and-defend.md) now names
separately submitted AI-reader means and explicitly says separate instances do
not establish independent human or model populations. These were clarity edits,
not new scientific acceptance gates. No outstanding finding remains in this
review's scope.

Inspection used bounded text/JSON field projections and local-path checks. No
Torch/model load, corpus or weight-body rehash, raw-corpus scan, GPU, network,
installation, Git or external publication occurred. It did not revalidate all54
archival input files or repeat the separately retained numerical/source-join
review. Original scientific criteria, dependencies, resource ceilings, missing
work and the learner's Day9 position are unchanged.
