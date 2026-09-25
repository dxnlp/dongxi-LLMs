# Production review — 2026-09-14

Seven low-resolution previews preceded the final 1920×1080, 30 fps H.264 renders.
All seven final exports have 960×540 looping GIFs at 15 fps, with individually
less than 5 MB payload. Exact durations, sizes, frame counts and hashes are in
`metadata.json`. MP4s remain local/Git-ignored; GIFs and source are reusable.

## Evidence

- All seven preview contact sheets inspected directly as local images.
- All seven final half-size posters inspected for typography and geometry;
  final contact sheets inspected for parallel positions, decode feedback,
  scheduling and layerwise handoff. Titles and labels are unclipped.
- Preview feedback incorporated input-to-projection arrows, a continuous token
  feedback path, a B-prefix cache-read pulse, and retained green completed layers.
- Clip validators check event order, cache counts and old-entry geometry,
  per-request state, and a visible handoff transfer tail.
- Independent article fixture verifies cached/full, chunked/full and
  reverse-row/batched equality, with a broken noncausal control.
- Export checks cover codec, dimensions, looping, nontrivial frame diversity and
  output/source hashes. Article checks cover seven media slots and local assets.
- Review-player tests use Node DOM stubs: seven toggles, default/reduced-motion
  startup and preference changes. The switch shows a final poster, not the paused
  current GIF frame.

## Limits

These are explanatory schematics, not measured model execution or network
timelines. Disaggregation is an alternative arrangement; the scheduling scene
transition does not claim live deployment migration. Handoff assumes compatible
model/state and compresses generation-state handling. Reset fades close an
editorial loop; they do not illustrate a cache invalidation rule.

The existing browser local-URL policy restriction was respected. No browser
HTML/mobile layout, actual browser GIF playback, X upload or X draft persistence
verification is claimed. User visual review and publication remain separate.
