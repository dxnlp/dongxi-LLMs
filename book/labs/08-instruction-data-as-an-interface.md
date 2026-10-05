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

## Audit a teacher before teaching from its answers

The fourth [Day11 session](../../notebooks/day-11/04_teacher_attempts_and_matched_rejection_sft.ipynb)
adds a transparent programmatic teacher, resumable attempt journal, format-only
filter, frozen candidate pool and actual tiny sequence SFT comparison. It keeps
well-formed wrong candidates so top and random selection remain distinct. The
trace is provenance; only final-answer/END tokens are supervised.

Inspect the stored original
[journal](../../fixtures/teacher-data/reference-journal/attempts.jsonl), then
predict which failures can share several rejection reasons. Compare all selected
IDs, prompt coverage and assistant-token budgets before viewing student results.
The [specification](../../experiments/specs/2026-10-04-teacher-data.md) precedes
collection, and the [report](../../experiments/reports/2026-10-04-teacher-data.md)
retains the failed transfers and cost boundaries.

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m unittest discover -s tests -p test_teacher_data_lab.py -v
CUDA_VISIBLE_DEVICES= PYTHONPATH=src OMP_NUM_THREADS=1 python -m dongxi_llms.teacher_data_lab --report /tmp/NEW-teacher-data.json --journal /tmp/NEW-teacher-journal
```

Use unused paths for a new measurement. Ordinary journal resumption skips
committed records, rejects changed contracts, and never reruns a committed
attempt. Explicit partial-tail recovery through `collect(..., recover_tail=True)`
first preserves corrupt bytes and a reason. It cannot repair complete corrupt
records or promise exactly-once physical execution. The journal supports one
writer; no teacher API, model download, GPU or server is involved.

The primary comparison controls samples/prompts, not all token exposure. Its
length-random sensitivity controls target counts where eligible alternatives
exist. Global coverage is a separate audit, and tiny held-out scores cannot
establish a universal ranking of selection methods.
