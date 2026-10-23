# X draft transfer — 2026-10-22 (Chinese version)

User-provided target: https://x.com/compose/articles/edit/2113363792482480128
(Chinese version, at the learner's request). Browser: the user's signed-in Chrome via the
Claude in Chrome extension. **Publish was never clicked.** Started from an empty draft.

## Post-reload verification

- Title: 指令微调：同一个损失，不同任务
- Cover: 1 (`assets/cover-post-training-02.png`; X stores it at 1200×480, 5:2).
- Body: 80 text blocks, 11 H2, 4 blockquotes; normalized text equals the cumulative text of
  `zh/x-editor-clean-body.html`. Placeholders: 0.
- Inline media: 10, all in their planned slots: nine GIFs (X converted them to looping
  `<video>`, all readyState 4) and the static three-forms summary PNG (image 6).
- Captions: all ten saved in X's caption fields, matching `zh/image-upload-order.md`.
- Status badge "Draft"; Publish button still offered and untouched.

## Method

Same as `../x-post-training-001/X_DRAFT_TRANSFER.md`: title via the textarea setter; cover
through X's cover file input plus the cover dialog's **Apply** only; the clean body split after
each image's `target_block_html` into 11 segments; each segment pasted as a synthetic HTML
`ClipboardEvent` into the empty block after the previous image, with exact cumulative-text
checks; each original image file staged in a temporary hidden file input and pasted at the
end of the body, requiring an exact +1 media block, the right preceding text and a trailing
empty block; captions typed after a real click on each placeholder and saved only after an
exact text match, then checked to have landed on the intended image.

## Incident and recovery

One text segment (the FLAN paragraph) was dropped while X was still processing the LIMA GIF.
The batch went on and pasted the FLAN GIF directly after the LIMA GIF. Only that newly
created media block was removed (its own Remove media control), the count returned to 2 with
the text unchanged, and the FLAN paragraph and GIF were pasted again successfully. After that,
every step waited 3 s after each image and threw an error on any failed check, so a batch
would stop instead of continuing. No other retries were needed.

The English version has not been transferred.
