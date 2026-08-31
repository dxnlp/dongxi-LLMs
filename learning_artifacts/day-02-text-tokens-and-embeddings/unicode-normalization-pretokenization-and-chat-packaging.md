# Unicode Normalization, Pre-tokenization, and Chat Packaging

- Day: 02 enrichment
- Date opened: 2026-08-31
- Status: demonstrated; pinned Qwen3 tokenizer-only evidence preserved
- Book destination: Chapter 2 tokenizer-input pipeline and special-token bridge
- Related evidence: `experiments/reports/2026-08-31-tokenizer-mechanics.md`
- Related production task: expanded `X-BPE-001`

## Questions that drove the discussion

- Is a visible character always one Unicode code point?
- Can two visually equivalent strings contain different code points and bytes?
- Does a successful tokenizer decode always reproduce the exact source sequence?
- Can adding one leading space change token identity without changing token count?
- How much model input can a chat template add around one visible user message?

## Predictions before execution

- NFC `café` would contain four code points and NFD `café` five, while both
  would contain four extended grapheme clusters.
- The family emoji `👨‍👩‍👧‍👦` would be one grapheme cluster composed from seven
  code points and 25 UTF-8 bytes.
- The pinned tokenizer might segment NFC and NFD differently, but both were
  expected to decode exactly to their original source strings.
- A leading space was expected to change the first token piece or ID.
- Chat-template packaging was expected to use more tokens than raw `Hello`.

## Refined mental model

The model input boundary is a pipeline, not a direct word lookup:

```text
source code points
→ normalization
→ pre-tokenization and boundary policy
→ subword or byte-level encoding
→ token IDs
→ optional special-token and chat-template packaging
```

Each stage can change what the next stage sees. Exact source equality, Unicode
canonical equivalence, visual similarity, token equality, and decode equality are
different properties and must not be collapsed into “the text is the same.”

A grapheme cluster is the reader-perceived unit that the earlier five-unit
taxonomy lacked. It may contain one code point, a base plus combining marks, or a
zero-width-joiner sequence. Python string length counts code points, not grapheme
clusters.

Special tokens and chat templates are frozen interface policy. They can add many
model positions without changing the visible user content, but this is not BPE
learning at runtime.

## Verified examples

### Unicode units

| Input | Code points | UTF-8 bytes | Grapheme clusters |
|---|---:|---:|---:|
| NFC `café` | 4 | 5 | 4 |
| NFD `café` | 5 | 6 | 4 |
| `👨‍👩‍👧‍👦` | 7 | 25 | 1 |

### Normalization

The pinned Qwen3 tokenizer mapped NFC and NFD `café` to the same IDs
`[924, 58858]`. NFC decoded exactly. NFD decoded as NFC and therefore failed
exact source-string equality while remaining canonically equivalent. The NFD
offsets also failed to cover the final combining mark in the original source.

### Leading spaces

```text
token  → ID 5839 → internal piece token
 token → ID 3950 → internal piece Ġtoken
```

Both used one token, proving that equal token count does not imply equal token
identity.

### Chat packaging

Raw `Hello` used one token. The pinned one-message template with a generation
prompt rendered role and boundary controls around that content and occupied nine
tokens. In this snapshot, `<|im_end|>` is EOS ID 151645 and `<|endoftext|>` is PAD
ID 151643.

## Demonstrated understanding

- Distinguished grapheme clusters from code points and bytes.
- Preserved the failed NFD exact-round-trip prediction rather than redefining
  normalized equality as exact equality.
- Treated normalization and leading-space behavior as tokenizer identity.
- Distinguished chat-template packaging from subword learning.
- Kept EOS and PAD as different semantic and batching roles.

## Evidence and limitations

The Unicode counts were computed mechanically with extended grapheme segmentation.
The tokenizer results used the exact pinned Qwen3-0.6B revision and loaded no
model weights. They establish behavior only for these fixed strings, template,
library versions, and tokenizer revision.

The leading-space example demonstrates a boundary effect but does not fully
characterize Qwen's pre-tokenization implementation. The chat example does not
generalize to other models or templates. No claim about model understanding or
generation quality follows from these token counts.

## Reuse opportunities

- Expand the X article from the narrow byte-fallback story to the complete
  source-text-to-ID boundary.
- Use the NFC/NFD failure as a concrete Chapter 1 evidence-discipline callback.
- Create a minimal pipeline diagram and a raw-versus-chat token strip.
- Revisit chat templates and assistant-only loss policy in Chapter 8.

## Remaining gaps

- Compare left/right padding only when batching becomes the active topic.
- Inspect additional normalization cases only if the article needs broader scope.
- Run a representative corpus before making language-level efficiency claims.
