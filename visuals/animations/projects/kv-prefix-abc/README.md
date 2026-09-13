# One document, three requests

`ANIM-KV-007`, approved 2026-09-13. Local Mac production; no publication.

Learning objective: distinguish prefill (computation), KV cache (retained states)
and prefix caching (reuse). Asking to change a document leaves the input unchanged;
supplying the revised document changes its token prefix and reuse boundary.

## Story

- A: original project document + risk-review instruction. Compute input states.
- B: same document + budget-halving instruction. Reuse document states; compute
  the new instruction with access to the cached prefix. Budget still reads 100.
- C: supplied document now says 50. Preserve the earlier prefix, recompute the
  related suffix including unchanged textual groups, and compute the new question.

White canvas, Songti SC Chinese and naturally shaped Arial, fixed columns,
purple retained K/V, green new computation, orange actual input edit.
The diagram shows one schematic layer. Input tiles group illustrative token spans;
they are not tokenizer results or literal serving blocks. Fingerprints identify
state dependencies only, not model activations. Actual output generation omitted.
Cache reuse requires the actual full input prefix and compatible execution settings.
Recomputing a suffix does not establish that every numerical element changes.

## Source and reuse

Chapter 4 §4.10 and
`learning_artifacts/day-04-attention-and-causal-information-boundary/why-cache-keys-and-values.md`.
The existing `kv-article-loops/` project supplies visual grammar; its files remain
unchanged. Shared `manim_style.py` supplies palette and whole-string typography.

## Render

```bash
uv run --project visuals/animations python visuals/animations/projects/kv-prefix-abc/mechanism.py
uv run --project visuals/animations python visuals/animations/projects/kv-prefix-abc/render.py --preview
uv run --project visuals/animations python visuals/animations/projects/kv-prefix-abc/render.py
```

Outputs: `rendered/prefix-abc.mp4` (1080p H.264), `prefix-abc.gif` (960×540),
full/half stills, summary and sequence contact sheet. MP4s are ignored by Git.
`metadata.json` records dependency versions, source/output hashes, command,
event-order and geometry checks. Visual review is recorded separately in `qa.json`.
Manim media is temporary; editable sources and reusable exports remain here.

## Production return — 2026-09-13

Final MP4: 18.97 seconds, 1920×1080 at 30 fps, 1,727,240 bytes. GIF:
960×540, 284 frames, infinite loop, 2,137,794 bytes. Fixture, event ordering,
invariant geometry and media checks pass; preview/final sequence contact sheets
and full/half-size stills were visually inspected. Browser playback and physical
mobile-device readability have not been audited. Awaiting learner review.
Article and X draft remain unchanged; nothing published, committed or pushed.
