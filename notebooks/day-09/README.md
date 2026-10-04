# Day 9 — Read the completed pretraining run

These three sessions form a mechanism → failure → evidence route for
[Chapter 6](../../book/chapters/06-pretraining-as-a-controlled-system.md).
Run on CPU on Mac or Spark. No model download or dashboard server is required.
Each reference is adjacent to its explanation; record a prediction first.

1. [What our TinyStories budget actually purchased](01_read_training_clocks.ipynb) — The model consumed 48.84M valid targets while processing 229.38M padded positions.
2. [Packing must preserve the prediction problem](02_masking_and_padding_control.ipynb) — We can investigate padding efficiency without another Spark training run.
3. [Prediction improvement and narrative quality](03_story_quality_and_decoding.ipynb) — Read complete saved continuations and separate grammar, entity consistency, event continuity and termination.

Use the chapter's worked solutions after discussing the interventions. Figures
are generated from the current computations or explicitly stored observations.
Reference execution checks material readiness, not independent learner mastery.
The full verification report identifies the source revision and plotted results.
