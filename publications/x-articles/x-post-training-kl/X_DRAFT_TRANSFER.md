# X draft transfer — 2026-11-04 (Chinese version)

User-provided target: https://x.com/compose/articles/edit/2118081140015656961
(Chinese version, at the learner's request: "不要发布"). Browser: the user's signed-in Chrome via
the Claude in Chrome extension. **Publish was never clicked.** The draft started empty: no title,
an empty body and no media.

## Post-reload verification

- Title: KL：用一个数衡量模型变了多少
- Cover: 1 (`assets/cover-post-training-kl.png`; X stores it at 1200×480, 5:2).
- Body:
  - 13 H2 and 5 blockquotes.
  - After every segment, the pasted text matched the cumulative text of `zh/x-editor-clean-body.html`
    (7,980 characters).
  - The editor text after the reload equals the text recorded just before it (8,735 characters with
    captions).
  - Placeholders: 0.
- Inline media: 12 GIFs, which X converted to looping video. Each sits after its planned preceding text.
- Captions: all twelve saved, matching `zh/image-upload-order.md`, in image order. No empty caption
  placeholder remains.
- Status badge "Draft". Publish was not clicked. The word "Published" on the page is the sidebar tab name.

## Method

Same as `../x-post-training-005/X_DRAFT_TRANSFER.md`. One change: the payload (`segs`, `ends`, `caps`,
`files`) was loaded into the page through the staging file input instead of being typed inline, and the
13 segments were split at the `[IMAGE NN]` placeholders of the draft HTML, whose joined text was first
checked against `zh/x-editor-clean-body.html`.

One check reported a false failure: image 3's expected preceding text spanned two list items, while the
check reads only the last block. The image was in the right place; the check strings were shortened and
nothing was redone. No segment, image or caption needed a retry.

# X draft transfer — 2026-11-04 (English version)

User-provided target: https://x.com/compose/articles/edit/2118088509898596353
(English version, at the learner's request: "不要发布"). Same browser and method as above. **Publish was
never clicked.** The draft started empty.

## Post-reload verification

- Title: KL Divergence: One Number for How Far a Model Has Moved
- Cover: 1 (`assets/cover-post-training-kl.png`; X stores it at 1200×480).
- Body: 13 H2 and 5 blockquotes. After every segment, the pasted text matched the cumulative text of
  `x-editor-clean-body.html` (17,526 characters). The editor text after the reload equals the text
  recorded just before it (19,128 characters with captions). Placeholders: 0.
- Inline media: 12 GIFs, which X converted to looping video, each after its planned preceding text.
- Captions: all twelve saved, matching `image-upload-order.md`, in image order. No empty caption
  placeholder remains. GIF 02's caption was corrected before transfer (the clip has no ratio dial).
- Status badge "Draft". Publish was not clicked.

No incidents: no segment, image or caption needed a retry.
