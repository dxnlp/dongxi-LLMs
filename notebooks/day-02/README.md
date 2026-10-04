# Day 2 — Byte BPE and learned embeddings

These three sessions form a mechanism → failure → evidence route for
[Chapter 2](../../book/chapters/02-text-tokens-and-embeddings.md).
Run on CPU on Mac or Spark. No model download or dashboard server is required.
Each reference is adjacent to its explanation; record a prediction first.

1. [How English-trained byte BPE still encodes 数](01_byte_bpe_training_and_encoding.ipynb) — A tokenizer can represent a character it never saw without knowing that character's meaning.
2. [Vocabulary compression changes the computation](02_vocabulary_and_sequence_cost.ipynb) — Why can a larger vocabulary shorten a sequence but increase output-head cost? Use the same text under different merge budgets, then keep parameter storage, per-position output work and attention interactions separate..
3. [An embedding receives learning through several paths](03_embedding_gradient_paths.ipynb) — Input IDs are categorical addresses.

Use the chapter's worked solutions after discussing the interventions. Figures
are generated from the current computations or explicitly stored observations.
Reference execution checks material readiness, not independent learner mastery.
The full verification report identifies the source revision and plotted results.
