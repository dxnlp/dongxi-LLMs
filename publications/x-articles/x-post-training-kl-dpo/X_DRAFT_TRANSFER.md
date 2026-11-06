# X draft transfer — 2026-11-06 (Chinese version)

User-provided target: https://x.com/compose/articles/edit/2118652439049986048
(Chinese version, at the learner's request). Browser: the user's signed-in Chrome via the Claude in Chrome
extension. **Publish was never clicked.** The draft started empty: no title, an empty body and no media.

## Post-reload verification

- Title: 你的 LLM 其实是一个 reward model：DPO 是怎么工作的
- Cover: 1 (`assets/cover-post-training-dpo.png`; X stores it at 1200×480, 5:2).
- Body:
  - 13 H2 and 5 blockquotes.
  - After every segment, the pasted text matched the cumulative text of the Chinese draft HTML (10,476 characters).
  - The editor text after the reload equals the text recorded just before it (11,387 characters with captions).
  - Placeholders: 0.
- Inline media: 13 GIFs, which X converted to looping video. Each sits after its planned preceding text.
- Captions: all thirteen saved, matching `zh/image-upload-order.md`, in image order. No empty caption placeholder
  remains.
- Status badge "Draft". Publish was not clicked.

## Method

Same as `../x-post-training-kl/X_DRAFT_TRANSFER.md`: the payload was loaded through a staging file input, the 14
segments were split at the `[IMAGE NN]` placeholders of the draft HTML, and each GIF was pasted at the body end with
a check on the media count and the preceding text.

Two captions were corrected before typing: a stray backslash in `π*`, and 步子 replaced by 更新 to match the article.

The cover was replaced the same day with the large DPO version at the learner's request, through X's cover file input and
the dialog's **Apply**. After a reload the editor text was unchanged (11,387 characters, 13 videos), and the draft was
still unpublished.

No segment, image or caption needed a retry. The English version has not been transferred. Before the English
transfer, decide whether to restructure it to match the Chinese rewrite.
