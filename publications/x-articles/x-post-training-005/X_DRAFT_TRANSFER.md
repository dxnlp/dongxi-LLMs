# X draft transfer — 2026-10-31 (Chinese version)

User-provided target: https://x.com/compose/articles/edit/2116301274433961984
(Chinese version, at the learner's request: "不要发布"). Browser: the user's signed-in Chrome via
the Claude in Chrome extension. **Publish was never clicked.** The draft started empty: no title,
an empty body and no media.

## Post-reload verification

- Title: Reward model：人的选择怎样变成一个分数
- Cover: 1 (`assets/cover-post-training-05.png`, upright variant; X stores it at 1200×480, 5:2).
- Body:
  - 11 H2 and 5 blockquotes.
  - After every segment, the pasted text matched the cumulative text of `zh/x-editor-clean-body.html`
    (8,089 characters).
  - The editor text after the reload equals the text recorded just before it (8,774 characters with
    captions).
  - Placeholders: 0.
- Inline media: 12 GIFs, which X converted to looping video. Each sits after its planned preceding
  text. Figure 1 is the reused X-POST-004 clip; 05–06 and 08–09 are the chat-model / reward-model
  pairs placed after their own paragraphs.
- Captions: all twelve saved, matching `zh/image-upload-order.md`, in image order. No empty caption
  placeholder remains.
- Status badge "Draft". Publish was not clicked. The word "Published" on the page is the sidebar tab name.

## Method

Same as `../x-token-001/X_DRAFT_TRANSFER.md` and `../x-post-training-004/X_DRAFT_TRANSFER.md`:
- the title is set through the textarea setter;
- the cover goes through X's cover file input, then the dialog's **Apply**;
- the clean body is split after each image's target block into 13 segments;
- each segment is pasted as a synthetic HTML paste, with an exact cumulative-text check;
- each GIF is staged in a hidden file input and pasted at the body end, requiring exactly one new
  media block and the right preceding text;
- each caption is typed after a real click and saved only after an exact text match on the intended
  image.

One adjustment: the plan split image 12's target inside a list, so the closing `</ul>` was moved from
segment 13 to the end of segment 12.

No incidents: no segment, image or caption needed a retry. The English version has not been transferred.
