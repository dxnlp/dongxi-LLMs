# X draft transfer — 2026-10-29 (Chinese version)

User-provided target: https://x.com/compose/articles/edit/2115777748404760576
(Chinese version, at the learner's request: "不要发布"). Browser: the user's signed-in Chrome via
the Claude in Chrome extension. **Publish was never clicked.** The draft started empty: no title,
an empty body and no media.

## Post-reload verification

- Title: 一颗 token 的诞生：从你的问题到模型的回答
- Cover: 1 (`assets/cover-token-birth.png`; X stores it at 1200×480, 5:2).
- Body:
  - 9 H2 and 9 blockquotes: the 8 key quotes plus the chat template.
  - After every segment, the pasted text matched the cumulative text of `zh/x-transfer-body.html`
    (4,206 characters).
  - The editor text after the reload equals the text recorded just before it (4,734 characters
    with captions).
  - Placeholders: 0.
- Inline media: 7 GIFs, which X converted to looping video. Each sits after its planned preceding
  text.
- Captions: all seven saved, matching `zh/image-upload-order.md`, in image order. No empty caption
  placeholder remains.
- Status badge "Draft". Publish was not clicked.

## Method

Same as `../x-post-training-004/X_DRAFT_TRANSFER.md`: synthetic HTML paste per segment with exact
cumulative-text checks, a staged file input for each GIF, and captions typed after a real click.
One addition: X Articles drops native code blocks on paste. `zh/x-transfer-body.html` is the clean
body with the two code blocks converted:
- the chat template becomes a blockquote;
- the 13 token IDs become a plain paragraph.

## Incidents and recovery

- **Template quote merged into one line.** The three `<p>` lines of the template blockquote were
  pasted as one block with no line breaks. The text check caught it (2 characters short). Two soft
  line breaks were inserted (Shift+Return after `user` and after `<|im_end|>`), and the text then
  matched exactly.
- **GIF 2 placed too early.** After that repair, the editor's cursor was still inside the
  blockquote, so GIF 2 was pasted directly after the template. The check caught it. Only that
  media block was removed, with its own Remove media control; the text was unchanged. After a real
  click at the end of the last paragraph, GIF 2 was pasted again in the right place.

No other retries were needed. The English version has not been transferred.

The learner then published the Chinese draft on X themself (reported 2026-10-29).
