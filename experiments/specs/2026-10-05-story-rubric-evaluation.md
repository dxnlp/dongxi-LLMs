# Frozen story ratings and paired comparisons

This CPU implementation fills a missing consumer for the original DXI-03
story contract. It does not generate stories, load weights, assign fabricated
human ratings or change the publication panel. Verification uses explicitly
authored controls; actual campaign scores remain missing until real responses
and independent reviews exist.

## Inputs and blinding

Consume the exact `story-publication-v1` contract from the staged campaign:
twelve unchanged openings, the five0–2 dimensions, greedy decoding and
temperature0.8 sampling at seeds909/1909/2909, context1024,256 new-token cap
and predetermined updates0/400/4000/8000/14000. Bind each actually collected
record to its arm/checkpoint/update, original opening/source group, decoding
recipe, interface, tokens, selected likelihoods, stop reason and measured work.
Never label a development opening or a missing planned generation as an
observed publication response.

Freeze the checkpoint comparison and rater/adjudication policy before grading.
Publish a shuffled anonymous packet containing the opening and complete
continuation; keep the arm/checkpoint codebook out of the reviewers' packet.
Retain the private codebook and content identities so scores can be joined
back without relabeling outputs after inspection. Blinding metadata does not
prove that a reviewer never saw a recognizable output elsewhere: disclose
reviewer type, independence and any known exposure.

## Ratings and disagreements

Require exactly the declared independent raters and original dimension names.
Accept only integer0/1/2 ratings, rejecting booleans, invalid values, unknown
IDs, duplicates and mismatched packet/contract identities. Empty templates
remain empty: no default0, midpoint or inferred rating. Preserve each rater's
raw decisions and disagreement, including incomplete rating coverage.

Apply only the predeclared adjudication rule. A declared mean is not agreement;
an unresolved disagreement stays missing for any adjudicated estimand. If a
separate adjudicator is used, retain the original two scores and the additional
decision separately. A capped continuation is graded as the actual incomplete
response, not silently completed or discarded. Natural EOS and narrative
closure are separate outcomes.

## Comparisons and uncertainty

Keep all predetermined checkpoint/decoding cells in coverage tables. Compare
only aligned source openings under matched updates and decoding recipes;
missing responses or ratings stay null. Show each rubric dimension separately,
with disagreement and stop rates rather than a single invented coherence score.

The independent grouping unit is the source opening, not an individual seeded
continuation or reviewer. For paired intervals, resample opening groups with
both arms and all included attempt scores kept together. State the included
population and denominator. A complete recipe has twelve opening groups;
forty-eight outputs across four recipes are not forty-eight independent
opening groups. Do not promote a partial aligned cohort to the whole panel.
Use bounded deterministic CPU bootstrap draws with retained seed/settings.

## Controls and evidence

Test malformed inputs, changed contracts/interfaces, duplicate/missing attempt
IDs, invalid ratings, missing rater coverage and adjudication misuse. Use
authored continuations with disclosed synthetic rating controls to demonstrate
blinding, disagreement and source-group pairing. Independently replay the
consumer and retain actual commands, exits, source hashes and control summaries.
Reusing a nonempty output must refuse rather than overwrite evidence.

The original contamination/development audit is a separate prerequisite.
Its negative exact/defined-near results cannot establish global semantic
novelty. This implementation and its authored controls cannot close the actual
trained story comparison or the Mac platform requirement. Original18-package
acceptance/dependencies and the learner's Day9 remain unchanged.

## Declared real-response review procedure

Before any publication response is graded, freeze the
[two AI reader declarations](2026-10-05-story-publication-raters.json).
Use two separate fresh subagents with no inherited conversation, no access by
instruction to the private codebook/producer labels and no shared consultation.
Each reads and judges the complete anonymous response independently; no scores
are inferred from training loss, stop status or authored controls. Retain each
original integer0/1/2 decision and its explanation. Declare unknown runtime
model identity and the possibility of shared underlying models. These are AI
ratings, not human consensus or validated human preference measurements.

Use the original five dimensions and the predeclared `two-rater-mean` rule.
Freeze the control-versus-half-LR comparison at each of the five original
updates. The achieved first400 tranche can supply actual responses at0/400;
later checkpoint cells and their quality scores remain null. Use800 bootstrap
draws with seed1010, resampling the twelve source-opening groups, to stay
within the existing CPU work ceiling for all five comparisons. Disagreements,
caps, missing cells and negative differences remain reported. This protocol
does not supply a rating or claim perfect blinding by itself.
