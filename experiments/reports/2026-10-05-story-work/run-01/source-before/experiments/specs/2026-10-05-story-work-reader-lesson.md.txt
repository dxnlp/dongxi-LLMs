# Story work clocks in the existing Day9 notebook

This predeclares an original CPU companion to the persistent-work integration.
It extends `notebooks/day-09/01_read_training_clocks.ipynb` without altering any
existing cell, output or preview. The historical TinyStories result remains a
separate measured case; this fixture does not reproduce its language capability.

Use the actual new accounted Session and the existing authored boundary geometry:
vocab16, width8, one layer, context8, tied modern causal MHA, AdamW peak0.003 and
floor0.0003, seed909, five-update horizon, warmup1, accumulation2, no activation
checkpointing. The explicitly ordered windows have3/6/1/7 valid labels and real
EOS15; the first two groups contain9 and8 targets in16 positions each. The
lesson uses two successful updates, not a new or extended training recipe.

Freeze caps before execution: reserve exactly25 attempted training targets and
use explicit1000 capacities for the remaining declared logical dimensions, with
a1MiB journal. Successful-target cap1000 stays unchanged in every arm. Run the
unaccounted native two-update reference first; preserve its original numerical
records and full model/Adam/stream/RNG/counter state for exact comparison.

In the accounted arm, complete the first9-target update and save a boundary.
Admit the next8-target group; after its first real forward/backward, deliberately
fail before the second forward. Preserve the failed ticket and poison the
session. Close and reopen the same physical journal from its independent saved
receipt. A fresh Session restores the first boundary and retries the exact
eight-target group. Require the final native numerical state and the retried
record to match the original reference exactly, excluding time/resource metadata
only. Report both completed successful labels17 and reserved training targets25.

Attempt the next complete training group at the exhausted attempted-target cap.
It must refuse before model work, preserve the stream/gradients/mode/LR/RNG/state
and leave the admitted journal prefix unchanged. This is a stopping instrument,
not a positive model-quality result. Data planning, byte reading/serialization,
metadata, memory, storage and physical containment stay outside these units.

The visual must use returned actual counters: one panel compares successful
target exposure with reserved target places across complete/fail/restore/retry;
the other separates completed, known-partial and uncertain attempted logical
calls. Include a prediction before code, adjacent reference explanation, labeled
axes and a controlled cap change. Restore the caller's Torch RNG; remove only
the lesson's owned temporary files after reading retained counters. No caller
source, output, historical weights or global environment may be mutated.

Acceptance uses the existing isolated CPU interpreter/kernel, offline flags,
one Torch thread,25GiB sampled reserve and at most180 seconds per child. Keep
failed attempts, exact command exits and source hashes. Execute the extended
notebook in a fresh kernel; export only its new preview to a fresh filename,
inspect it visually, and verify the old preview remains byte-identical.
Additional tests cover caller RNG, cap refusal, poisoned state, real native
forward count, retained old-prefix spending and exact comparison. Notebook
execution is material readiness, not learner mastery or Mac/GPU/hosted evidence.
Extend the existing CAND-ANIM-019 proposal only; media production is not approved.

Before measurement, add the controlled capacity24 arm to the helper. Change only
the attempted-target cap, with a fresh independently bound journal: after9 plus
the failed8, the same eight-target retry has only7 allowance and must refuse
without numerical work. Return that refusal and actual clocks9/17 for the
notebook to plot. The original model/Adam/stream/RNG completed boundary remains
intact; no successful second update or exact two-update final comparison is
claimed in this arm. Default capacity25 retains the previously declared full
retry and17/25 result. The reader changes a notebook argument, not course files.
