# Meteor Corpus: BPE Counts, Ties, and Token IDs

- Date: 2026-09-03; Mac Studio content-production follow-up to Day 2.
- Book destination: Chapter 2 BPE worked-example companion; final visual
  integration awaits user review and does not alter the active Day 3 lesson.
- Task: `ANIM-BPE-002`, requested and approved for production by the user.
- Executable source: `src/dongxi_llms/meteor_bpe.py`, reusing `tiny_bpe.py`.
- Evidence: `visuals/animations/projects/meteor-bpe/trace.json`.

## Initial framing

The learner supplied a literary contrast between human experiences associated
with language and a tokenizer's numeric representation, and requested the
sequence `流=1, 星=2, 雨=3, 流星=4, 流星雨=5` from five small corpus entries.
The goal is an animation of BPE training and encoding. The literary judgment
about AI's eventual semantic limits is not an empirical conclusion of this
experiment. No developmental claim about how humans acquire words was tested.

## Predictions and declared rules

The production brief was written before running the corpus. Each supplied string
appears once; every full string is a separate segment; there is no normalization
or cross-string merging. The base alphabet consists of characters, not bytes.
The stopping budget is two merges.

Selection uses maximum weighted adjacent-pair count. Ties prefer `(流星, 雨)`
only if it is among the maxima, then fall back to Unicode lexicographic order.
This explicitly chosen teaching preference produces a reproducible example; the
corpus alone does not force the second merge.

## Observed mechanism

| Stage | Selected pair | Count | Selection evidence | Corpus token count |
|---|---|---:|---|---:|
| Initial | — | — | 17 distinct characters in five strings | 22 |
| Round 1 | `流 + 星` | 2 | Unique maximum | 20 |
| Round 2 | `流星 + 雨` | 1 | One of 15 tied maxima; declared preference applies | 19 |

The final segmentations are:

```text
[流星] [蝴] [蝶] [剑]
[带] [你] [去] [看] [流星雨]
[流] [水]
[星] [辰] [大] [海]
[雨] [一] [直] [下]
```

All original character tokens remain in the vocabulary. Adding `流星` and
`流星雨` gives 19 vocabulary entries. Assigning IDs happens after the vocabulary
is constructed in this demonstration: the requested five tokens receive 1–5;
the remaining 14 characters receive 6–19 in first-appearance order. These are
arbitrary categorical identifiers, not learned semantic magnitudes or production
model IDs.

During encoding, `流星雨` follows the two learned merges and becomes `[5]`.
`星雨` remains `[2, 3]`, because no `(星, 雨)` merge was learned. All supplied
strings round-trip exactly under the complete toy mapping.

## Evidence and limits

- Nine focused BPE tests passed, including five new meteor-corpus tests.
- The trace stores every pair count, all tied maxima, corpus rewrites, the full
  vocabulary, and eight encoding examples.
- The tie preference cannot override a pair with a strictly larger count.
- Character-level coverage is limited to the known toy alphabet; this example
  has no complete byte fallback or special tokens.
- The five displayed entries are a vocabulary subset, not the whole tokenizer.
- Neural language-model training and semantic understanding are not measured.
- The animation retains constituent entries, labels the tied maximum, and
  separates training from encoding to prevent three common misconceptions.

## Reuse and next action

The editable scene, production brief, render script, local review, media, and
metadata are under `visuals/animations/projects/meteor-bpe/`. Review the visual
candidate before integrating it as a Chapter 2 companion or an article asset.
The earlier byte-level BPE animation remains unchanged.
