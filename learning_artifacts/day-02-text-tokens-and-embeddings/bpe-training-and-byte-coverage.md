# BPE Training and Byte-Level Coverage

- Day: 02
- Date opened: 2026-08-30
- Status: demonstrated
- Book destination: Chapter 2 sections on BPE and unknown text
- Related evidence: `experiments/reports/2026-08-30-qwen3-multilingual-tokenization.md`; `experiments/reports/2026-08-31-tokenizer-mechanics.md`; `visuals/animations/rendered/bpe-byte-merges-manim.mp4`
- Related production tasks: `X-BPE-001`, `ANIM-BPE-001`

## Questions that drove the discussion

- How is a tokenizer vocabulary decided?
- Is BPE trained in an offline phase?
- Is encoding merely dictionary lookup?
- How can a tokenizer trained only on English encode the Chinese character `数`?
- What does Chinese BPE look like when there are no suffixes analogous to
  `like + ed`?

## Learner's initial model

The learner first imagined tokenization mainly as checking strings against a
vocabulary and expected an English-only tokenizer to turn Chinese into unknown
tokens. English morphology made subwords intuitive, but the corresponding
mechanism for Chinese was unclear.

## Refined mental model

BPE has two distinct phases:

1. **Offline tokenizer training:** start from base symbols, count eligible
   adjacent pairs over a representative corpus, merge a selected frequent pair,
   rewrite the representation, and repeat under a finite merge/vocabulary budget.
2. **Frozen encoding:** normalize and pre-tokenize new text, convert it to base
   symbols, replay the learned merge priorities, and map the final pieces to IDs.

Encoding does not add tokens. Vocabulary membership alone is insufficient to
describe segmentation because merge ranks and pre-tokenization boundaries also
matter.

In byte-level BPE, all 256 byte values provide reversible coverage. Learned
merges provide compression. Therefore, `no unknown token` and `efficient
tokenization` are separate claims, and neither establishes model understanding.

## Concrete examples and derivations

UTF-8 represents one visible Chinese character as three bytes:

```text
数 → E6 95 B0
```

A byte-level tokenizer trained only on ASCII English probably has no learned
merge for this sequence, so it can fall back to:

```text
[E6] [95] [B0]
```

With sufficient Chinese exposure, ordered merges can compress the same bytes:

```text
[E6] + [95]       → [E6 95]
[E6 95] + [B0]    → [数]
```

Later statistical merges may cross character boundaries:

```text
[数] + [据]        → [数据]
[数据] + [库]      → [数据库]
```

Chinese BPE pieces may therefore be partial byte sequences, complete characters,
multi-character words, or frequent expressions. BPE does not know their meaning;
it exploits recurring adjacent patterns.

## Demonstrated understanding

- Correctly identified tokenizer training as an offline phase.
- Correctly reasoned that a frequent `数据治理` sequence may become a token only
  if its constituent merges exist and it wins competition for finite merge
  budget during training.
- Understood that constituent tokens remain available after a merged token is
  added.
- Correctly distinguished runtime segmentation from vocabulary learning.
- Explained back that an English-only byte tokenizer can preserve Chinese bytes
  even though the resulting model may not understand Chinese.

## Evidence and limitations

The byte mapping for `数` is a deterministic UTF-8 fact. The merge progression is
a transparent teaching example, not a claim that every production tokenizer
learned those exact intermediate rules. The pinned Qwen3 report now verifies
multi-character Chinese spans in one worked example, including `小型`, `语言`,
and `下一个`; it does not reconstruct the tokenizer's historical merge sequence.

Tokenizers without complete byte coverage or byte fallback may emit `<unk>` for
unsupported characters.

A later executable teaching corpus now verifies three complete pair-count rounds:

```text
h + u   → hu    count 10
hu + g  → hug   count 10
hug + s → hugs  count 3
```

The first maximum was tied, so the run also demonstrates that a declared tie
policy is needed for deterministic training. Its frozen encoder replays those
rules in rank order. This character-level example establishes generic BPE
mechanics; it remains separate from the illustrative Chinese byte merges and does
not reconstruct Qwen's tokenizer history.

## Open edges

- Compare production implementations only when a claim requires details beyond
  the transparent trainer.
- Run a representative corpus before making language-level efficiency claims.

## Reuse opportunities

- Central mechanism section of Chapter 2.
- X article `X-BPE-001`.
- BPE animation `ANIM-BPE-001` and its planned Mac/Manim refinement.
- Exercise predicting token counts under byte-, character-, and phrase-level
  merge inventories.
