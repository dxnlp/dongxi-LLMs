# X draft transfer — 2026-10-20 (Chinese version)

User-provided target: https://x.com/compose/articles/edit/2112617637763047425
(user chose the Chinese version for this draft). Browser: the user's signed-in
Chrome through the Claude in Chrome extension. **Publish was never clicked.**

Started from an empty draft: no title, empty body, zero media.

## Post-reload verification

- Title: 预训练之后，为什么还要后训练？
- Cover: 1 (shared English cover, `assets/cover-post-training.png`; X stores it at 1200×480, 5:2).
- Body: 55 text blocks, 8 H2, 12 blockquotes; normalized text equals the
  cumulative text of `zh/x-editor-clean-body.html` (2,639 characters).
- Inline media: 6 — five GIFs (X converted them to looping `<video>`, all
  readyState 4, durations 11.1 / 17.7 / 11.0 / 10.8 / 14.9 s) and the Chinese
  comparison PNG. All six preceding/following-text slots correct.
- Captions: all six saved in X's native caption fields, matching `zh/image-upload-order.md`.
- Placeholders: 0. Status badge "Draft"; Publish button still offered and untouched.
- X's displayed count: 122 words (platform count, not Chinese characters).

## Method (adapting `x-article-drafter` to Claude in Chrome)

The skill's Chrome route (`chrome_transfer_local_article.py`) needs AppleScript
JavaScript permission and only counts `img` media; its clipboard helper handles
PNG/JPEG only, so GIFs would lose animation. Instead, following the skill's
sequential-construction rules:

1. Title set through the textarea's native value setter; cover uploaded into X's
   cover file input, then only the cover dialog's **Apply** was clicked.
2. The clean body was split after each image's `target_block_html` into seven
   segments (Python equivalent of `buildSequentialBodySegments`; segments rejoin
   to the exact source).
3. Each HTML segment was pasted as a synthetic `ClipboardEvent('paste')` with
   `text/html` + `text/plain` into the empty block after the previous image;
   cumulative normalized text was verified after every segment.
4. Each image was staged in a temporary hidden `<input type=file>` (filled by the
   extension's file upload), then pasted as the original file at the end of the
   last text block. Each paste required an exact +1 media-block increase, the
   correct preceding text and a trailing empty block. GIF bytes stayed original,
   so animation was preserved.
5. Captions: real click on each "Provide a caption" placeholder, typed text,
   Save only after an exact text match (script clicks did not open the dialog).
6. Waited for "Last saved", reloaded the exact URL and re-audited from scratch.

No retries were needed. The English version has not been transferred; it needs
its own empty draft.
