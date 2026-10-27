# X draft transfer — 2026-10-24 (Chinese version)

User-provided target: https://x.com/compose/articles/edit/2113912265061593088
(Chinese version, at the learner's request). Browser: the user's signed-in Chrome via the
Claude in Chrome extension. **Publish was never clicked.** Started from an empty draft.

## Post-reload verification

- Title: LoRA 与 QLoRA：用一块 GPU 微调大模型
- Cover: 1 (`assets/cover-post-training-03.png`; X stores it at 1200×480, 5:2).
- Body: 9 H2, 3 blockquotes. The pasted body text matched the cumulative text of
  `zh/x-editor-clean-body.html` after every segment (3,764 characters), and the editor text
  after the reload equals the text recorded just before it (4,363 characters with captions).
  Placeholders: 0.
- Inline media: 11, each pasted after its planned preceding text: ten GIFs and the static
  pattern-lens PNG (image 11, under 小结).
- Captions: all eleven saved in X's caption fields, matching `zh/image-upload-order.md`, in
  image order; no empty caption placeholder remains.
- Status badge "Draft"; Publish button untouched.

## Method

Same as `../x-post-training-002/X_DRAFT_TRANSFER.md`: title via the textarea setter; cover
through X's cover file input plus the cover dialog's **Apply** only; the clean body split after
each image's target block into 12 segments, each pasted as a synthetic HTML `ClipboardEvent`
with exact cumulative-text checks; each image staged in a hidden file input and pasted at the
end of the body, requiring an exact +1 media block and the right preceding text; a 3 s wait
after each image and a thrown error on any failed check; captions typed after a real click on
each placeholder and saved only after an exact text match on the intended image.

No incidents: no segment or image needed a retry. The final segment begins with the summary
list, so its list-start check reported a list block there; that is the planned structure.

The English version has not been transferred.
