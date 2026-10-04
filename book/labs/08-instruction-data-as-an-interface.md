# Lab 8 — Audit the interface before training

Use the three Day 11 notebooks in this order:

1. [Roles, templates and masks](../../notebooks/day-11/01_roles_templates_and_masks.ipynb): follow one assistant target from message ownership through the causal shift.
2. [Padding and packing](../../notebooks/day-11/02_padding_packing_boundaries.ipynb): inspect visibility and loss independently; change a segment boundary and predict the opened links.
3. [Mixtures and cards](../../notebooks/day-11/03_mixtures_and_data_cards.ipynb): change response lengths and observe token exposure.

The symbolic encoder in [instruction_data_lab.py](../../src/dongxi_llms/instruction_data_lab.py) provides IDs, ownership and labels. It deliberately rejects unknown teaching words and incomplete training conversations. It is a transparent microscope, not a multilingual tokenizer.

Generate the optional original English task dataset without downloading a model:

~~~bash
python scripts/prepare_chapter09_instruction_fixture.py --output outputs/course-sft-interface-v1
~~~

This produces 240 training, 60 development and 120 publication-test JSONL records, plus a data card with exact content hashes. Every value/source group stays in one split. Shared task forms mean the held-out result concerns unseen values under familiar copy/reverse/extract instructions. The generator rejects a nonempty destination to preserve earlier outputs.

Before any GPU run, inspect the first/last examples and every task family. Assert there are no cross-split groups, render the actual pinned tokenizer template, and print ID/label pairs including end markers. The [explicit Qwen-style template](../../experiments/data/instruction_interface_v1.jinja) uses existing message tokens and must pass prefix-consistency checks. The runner rejects changed prefixes and overlength examples.

Packing is explored in the notebook but is not enabled in the real-model runner. Its production path uses independent right-padded conversations with attention masks and answer labels. Do not infer a packed backend from a visibility diagram.

Acceptance evidence consists of role validation, target counts, end-marker preservation, boundary inspection, hashes and the card's limits. Correct serialization alone cannot establish that demonstration answers are semantically correct.
