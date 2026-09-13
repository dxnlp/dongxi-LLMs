# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `Why LLMs Cache K and V—but Not Q`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `49`
- Images: `7`

## Reverse Insertion Order

### IMAGE 07

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-attn-kv-001/assets/global-sharing.gif`
- Exists: `True`
- Insert after block index: `43`
- Target text: Shared KV representation changes the two-role layout. Inside the reference core-attention kernel, the same selected vec...
- Next text: These dependencies enable a different prefill strategy. For an illustrative 100,000-token project history, the encoder...

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-attn-kv-001/assets/memory-growth.gif`
- Exists: `True`
- Insert after block index: `29`
- Target text: At 8,192 positions it becomes 384 MiB. At 4,096 positions with four KV heads it becomes 96 MiB. These are calculated pa...
- Next text: Grouped-query attention, or GQA, lets several query heads share K/V heads. For example, four query heads could use two...

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-attn-kv-001/assets/math-memory.png`
- Exists: `True`
- Insert after block index: `26`
- Target text: For a conventional cache with separate, equal-width K and V tensors, identical layer geometry, and T retained positions...
- Next text: The factors are batch size B, layer count L, retained positions T, the number of KV heads per layer, head width d, and...

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-attn-kv-001/assets/decode-step.gif`
- Exists: `True`
- Insert after block index: `22`
- Target text: Decode extends it one position at a time. The new position passes through every layer. Zoom into one attention head: on...
- Next text: Notice the timing: append the new K/V before computing attention, so the query can use its own position too. After this...

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-attn-kv-001/assets/append-edit.gif`
- Exists: `True`
- Insert after block index: `17`
- Target text: Consequently, each layer can retain its earlier K/V instead of recomputing them.
- Next text: The cache belongs to a particular prefix, position, layer, and model execution. It is not a vocabulary dictionary with...

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-attn-kv-001/assets/math-attention.png`
- Exists: `True`
- Insert after block index: `9`
- Target text: For one head at a new position t, the core calculation is:
- Next text: Here d is the query/key width. The subscript “≤ t” includes both earlier positions and position t itself; the superscri...

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-attn-kv-001/assets/prefill-decode.gif`
- Exists: `True`
- Insert after block index: `5`
- Target text: Notice the timing: selecting “ blue” does not yet create its K/V. Those are computed when that token is fed back into t...
- Next text: Why the name contains K and V
