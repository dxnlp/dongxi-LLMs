# X draft transfer — 2026-10-28 (Chinese version)

User-provided target: https://x.com/compose/articles/edit/2115385031132635138
(Chinese version, at the learner's request: "不要提交"). Browser: the user's signed-in Chrome via
the Claude in Chrome extension. **Publish was never clicked.** Started from an empty draft (no
title, empty body, no media).

## Post-reload verification

- Title: 写给 LLM 读者的强化学习术语：一个分数如何分到每个 token
- Cover: 1 (`assets/cover-post-training-04.png`; X stores it at 1200×480, 5:2).
- Body: 12 H2, 5 blockquotes. After every segment, the pasted text matched the cumulative text of
  `zh/x-editor-clean-body.html` (8,684 characters). The editor text after the reload equals the
  text recorded just before it (9,697 characters with captions). Placeholders: 0.
- Inline media: 13 GIFs, which X converted to looping video. Each was pasted after its planned
  preceding text; images 12 and 13 are adjacent under 总结, as planned.
- Captions: all thirteen saved in X's caption fields, matching `zh/image-upload-order.md`, in image
  order; no empty caption placeholder remains.
- Status badge "Draft"; Publish button untouched. (The word "Published" on the page is the
  sidebar tab name.)

## Method

Same as `../x-post-training-003/X_DRAFT_TRANSFER.md`, with three additions:

- The clean body was split after each image's target block into 14 segments, one of them empty
  between images 12 and 13. Each paste also checked the cumulative H2 and blockquote counts.
- Image 13 was pasted into the empty block after image 12, and the check required the previous
  block to be media.
- The last segment starts with the summary list. Its list-start check was relaxed for that segment
  only, because the list is the planned structure there.

No incidents: no segment, image or caption needed a retry.

Afterwards (reported 2026-10-29): the learner published the Chinese article from this draft.
The English version has not been transferred.
