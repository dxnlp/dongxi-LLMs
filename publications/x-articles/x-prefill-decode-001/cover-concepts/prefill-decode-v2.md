# Prefill and Decode — cover v2

Created on 2026-09-14 with the built-in image generation tool, using the local
KV Cache v2 cover as a style reference only. User requested the same article-cover
style: black-and-white stippled ink, deep negative space, large white single-line
type. A new broad-input-to-sequential-output composition distinguishes this topic.
The artwork is an editorial metaphor, not a tensor-compression or timing diagram.

Source: `prefill-decode-v2-source.png` (1983×793), copied from the built-in output.
Final: `../assets/cover-prefill-decode-v2.png` (2000×800, exact 5:2).
The original generated output remains in Codex's generated-images directory.
Only dimension normalization was applied, without cropping or compositing:

```sh
sips -z 800 2000 publications/x-articles/x-prefill-decode-001/cover-concepts/prefill-decode-v2-source.png --out publications/x-articles/x-prefill-decode-001/assets/cover-prefill-decode-v2.png
```

`build.py` selects the new cover on every rebuild. The original white
`assets/cover.png` and its generator are preserved. Body text, seven animations,
and the existing KV Cache package are unchanged. No X upload or publication.
The final image was inspected for exact wording, readable single-line title,
black/white series resemblance, and safe title margins; dimensions checked.

## Exact generation prompt

Use case: stylized-concept.
Asset type: original editorial cover for an X Article titled Prefill and Decode, in the same visual series as the supplied KV Cache cover.
Input image: reference for visual style ONLY, not an edit target; preserve its black-and-white stippled ink aesthetic, large crisp sans-serif white typography, deep black negative space and engraved sculptural detail. Create a NEW composition for a different mechanism.
Canvas: exact 5:2 ultra-wide landscape, ideally 2000 x 800 pixels. Not 16:9.
Concept: many known inputs processed together, followed by outputs unfolding one at a time. An elegant broad fan of fine white parallel stippled ribbons flows into a compact luminous fold; emerging from that fold, a single slender trajectory carries a few distinctly separated bright bead-like marks, continuing outward in order. Retain some fine layered geometry in the fold as a subtle family resemblance to the KV Cache cover. This is an abstract visual metaphor, not literal token compression, an architecture diagram or a performance measurement.
Composition: huge single-line title across the upper-middle dark negative space, with a sculptural flowing ribbon motif across the lower half, spacious readable separation between text and artwork. Strong silhouette at thumbnail size; broad-input-to-single-sequence contrast visible without labels. Keep lettering and main artwork inside 6% safe margins; atmospheric stippling can reach the outer edges.
Text verbatim: "Prefill and Decode" exactly once, one unbroken horizontal line, very large white modern sans-serif matching the reference, naturally kerned, crisp solid letters, not made of particles. No subtitle, no other letters or numbers.
Palette: black and white only, dot-density shading and fine etched lines, no color, no smooth glossy gradients.
Avoid: boat, person, black hole, logos, credits, watermarks, arrows, technical boxes, extra text, copied reference composition.

