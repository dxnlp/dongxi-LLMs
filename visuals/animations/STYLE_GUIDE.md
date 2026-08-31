# Dongxi animation visual system

Status: approved from the `ANIM-BPE-001` review on 2026-08-31.

Use this system as the default for Manim course animations. The reference
implementation is `manim_bpe_byte_merges.py`; copy its visual grammar, not its
topic-specific layout.

Import shared palette and geometry helpers from `manim_style.py` rather than
duplicating them in each scene.

## Core principle

Make the mechanism move. Use text only to name a stable object, mark a stage,
or state the final distinction. A viewer should understand the main transition
with the sound off, on a phone, and from a paused frame.

- Prefer one continuous computation over a sequence of explanatory slides.
- Preserve object identity while state changes; move or transform the existing
  object instead of replacing the whole composition.
- Use physical relationships to carry meaning: adjacency, splitting, fusion,
  routing, selection, and scale.
- Keep one visual question on screen at a time.
- Remove prose, tables, legends, and panels unless the animation becomes less
  accurate without them.

## Canvas and hierarchy

- Use a white `#FFFFFF` canvas.
- Keep a small title at the upper left and, only when needed, one compact status
  badge at the upper right.
- Reserve the center for the mechanism.
- A persistent scale or process track may use at most three stable anchors.
- End with no more than two compact claims and one short caveat.
- Use generous negative space; do not fill an empty region merely for balance.

## Typography

- English: Arial, normal weight, natural kerning.
- Chinese: Songti SC (`宋体`), normal weight.
- Do not simulate alignment with spaces or manually alter character spacing.
- Do not place mixed English and Chinese in one `Text` object when font metrics
  matter; compose separate objects and align their geometry.
- At 1920×1080, the BPE reference uses Manim sizes 34 for the title, 38 for
  token contents, 18 for stage labels, and 21–22 for compact badges. Treat
  these as starting points, then test at 50% scale.
- Avoid sentences in persistent screen regions. If narration is required, put
  it in audio, captions, or surrounding book prose rather than the diagram.

## Color roles

Use color semantically and keep each role stable throughout a scene.

| Role | Color |
|---|---|
| Foreground | `#111827` |
| Muted labels | `#64748B` |
| Dividers and token borders | `#CBD5E1` |
| Base or input units | `#2563EB` |
| Intermediate representation | `#7C3AED` |
| Composed or higher-level result | `#047857` |
| Focus, rule, or caveat | `#B45309` |
| Text on filled tokens | `#FFFFFF` |

Do not use color as decoration. If two objects share a semantic role, they
should normally share a color.

## Alignment discipline

- Compute rows from actual object bounds and center them as a group.
- Use `arrange`, `next_to`, geometric anchors, or a reusable positioning helper;
  never align variable-length text with inserted spaces.
- Keep persistent anchors fixed. Animate the data, not the coordinate system.
- Align labels to the objects they describe rather than to unrelated text.
- Avoid dense tables in motion graphics. If exact tabular data is essential,
  make it a separate static figure.

## Motion grammar

Use motion to expose cause and effect, not to decorate the frame.

1. Briefly focus the operands or source object.
2. Move related objects toward the operation.
3. Transform them into the result with continuity.
4. Recenter surviving objects from their measured widths.
5. Pulse the result once, then pause long enough to read it.

The BPE reference timing is a useful baseline: roughly 0.25 s for focus, 0.35 s
for approach, 0.8 s for transformation, 0.35 s for result emphasis, and 0.2 s
of rest. Keep the camera static unless camera movement communicates a real
change of scale or scope.

## Accuracy before beauty

- Every transformation must correspond to the mechanism being taught.
- Distinguish learned parameters or frozen rules from work performed on the
  current input.
- Preserve important limitations in one compact visual caveat or in adjacent
  prose; never imply understanding, causality, or empirical performance from a
  representation change alone.
- When simplification omits a branch or competing operation, record that limit
  in the metadata or companion explanation.

## Deliverables and acceptance checks

Preserve the editable source, exact render command, dependency revisions,
1920×1080 H.264 MP4, lightweight 960×540 GIF, representative PNG still, and a
metadata file with hashes and known limitations.

Before review:

1. Inspect a contact sheet covering the complete sequence.
2. Inspect the final still at full resolution and at 50% scale.
3. Confirm natural English kerning and correct Chinese glyph fallback.
4. Check that no transition introduces a false extra object or ambiguous arrow.
5. Verify source compilation, media dimensions and codec, metadata JSON, and
   recorded hashes.
6. Keep the candidate uncommitted or on its review branch until its content and
   visual style are approved.
