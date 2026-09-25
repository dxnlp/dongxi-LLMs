# X draft transfer — 2026-09-14

User-authorized target: https://x.com/compose/articles/edit/2099600211739803648

Started from an empty title/body and zero media in the signed-in in-app browser.
Set `Prefill and Decode`, uploaded `assets/cover-prefill-decode-v2.png` unchanged,
and transferred the current Chinese clean-body HTML plus seven original GIFs.
All seven local captions were saved in X's native caption fields.

## Post-reload verification

- Title: Prefill and Decode.
- X's displayed count: 261 words (platform count, not Chinese character count).
- Body: 55 text blocks, 6 H2 headings; normalized text equals local clean HTML.
- Media: 7 inline animations + 1 cover = 8; all seven preceding/following-text
  slots correct. Caption text matches each local plan entry.
- Placeholder count: 0. `Last saved` observed before reload and after reload.
- X converted GIFs into looping video elements. All seven had readyState 4
  after reload, with durations 9.87, 7.27, 8.53, 16.73, 7.73, 12.47, 10.33 s.
- First clip playback was activated and observed advancing to 5.24 s with
  changed rendered frames. Other six clips were checked for loaded duration and
  persistence, not individually played end-to-end.
- Cover and first media region inspected in the actual X editor.
- Publish was never clicked. No Git commit or push.

## Transfer adaptation and recovery

The installed x-article-drafter helper counts `img` elements. During upload X
replaces a GIF image placeholder with `video`, causing its count check to stop
after the second GIF. No content was deleted or pasted again. A project-local
`transfer-inapp.mjs` adapter counts media blocks across image/video conversion,
excludes nested caption blocks, and guards the exact cumulative-text/media
boundary before appending the next segment. It uses supported cua_repl Tab APIs,
with the skill's original segmentation and clipboard utilities.

File-input chooser upload succeeded after a first chooser-button timeout.
Transient media processing exceeded short locator waits; subsequent read-only
checks verified retained uploads before continuing. The final adapter returns
pending status instead of repasting. Read-only audits separately verify text,
caption, title, cover, media count and exact slots after reload.

This is evidence for this specific transfer, not a guarantee of future platform
behavior. Future updates must inspect the non-empty draft before changing it;
the append helper does not authorize clearing or replacing existing content.
