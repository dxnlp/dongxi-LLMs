# Two phases. Two kinds of waiting.

`ANIM-PD-002` — independently reusable animation, explicitly approved 2026-09-15.
Mac production; dirty main at ca2cfe1 preserved, pull skipped. No article edits,
X activity, model/GPU job, commit or push.

## Example and learning objective

A project brief says Friday launch is planned but payment integration is pending.
The request asks whether launch will be on time. Output: 上线 / 可能 / 会 / 延期。
One continuous arrival timeline carries the mechanism: orange TTFT spans request
send to first output arrival; three green ITLs connect consecutive arrivals.
Each word appears instantly at its interval endpoint. Previous spans remain
visible, making the two kinds of waiting directly comparable.

The title and English mechanism labels use naturally shaped Arial; Chinese uses
Songti SC. Words and phrases replace token symbols for readability. They are not
actual tokenizer segmentation or measured model output. Each card stands in for
one token arrival. Interval durations are schematic, not benchmark values.
TTFT may also include queueing, sampling and transport; the Prefill label does
not equate all request-to-first-arrival latency with prefill computation.

## Render

```sh
uv run --project visuals/animations python visuals/animations/projects/two-kinds-of-waiting/render.py --preview
uv run --project visuals/animations python visuals/animations/projects/two-kinds-of-waiting/render.py
```

Outputs: `rendered/two-waits/two-waits.mp4` (1080p), `two-waits.gif` (960×540),
posters/contact sheets; source and media hashes in metadata. MP4 stays local and
Git-ignored. Existing article and animation files are hash-protected during render.

## Verification return

The first 25.13-second document-processing version was rejected for missing the
TTFT/ITL focus. The user explicitly approved this timing-focused revision.
Validation checks TTFT starts at request send, every interval ends at its output
arrival, and each ITL starts exactly at the preceding arrival, in both time and
timeline coordinates. Media and protected-file hashes are recorded in metadata.
`review.html` is a standalone native-video player; no trained-model run is claimed.

Final revision: 18.8 seconds, 1920×1080 H.264 at 30 fps; 960×540 GIF,
282 frames / 230 distinct frames, 747,187 bytes. Timeline, bounds, media and
protected-file hash checks pass. Preview contact sheet and final full/half-size
posters visually inspected. Dashed guides stop above the arrival labels.

Follow-up refinement: document facts (plan and risk) and User query now occupy
separate vertically aligned cards. The document appears first, followed by the
question; the existing arrival timeline and TTFT/ITL boundaries are unchanged.
Card separation and text-fit assertions supplement the canvas checks.
