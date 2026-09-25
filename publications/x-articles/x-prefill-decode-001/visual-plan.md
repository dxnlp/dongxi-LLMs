# Static storyboard → seven article animations

Source: user-approved phase-focused article. Reuses CAND-ANIM-009. On 2026-09-14
the learner approved production of all seven loops as `ANIM-PD-001`, superseding
the earlier deferrals below. White canvas, natural Arial, fixed semantic colors and
object anchors follow `visuals/animations/STYLE_GUIDE.md`. Production metadata
stays outside learner-facing figures. Cover ratio is 5:2; title stays on one line.

Earlier 2026-09-14 clarification: the learner requested a Decode illustration, then
specified static images first and all animation production together later. The
feedback-loop still is now body slot 04; no animation was started or rendered.
The learner then asked why position 2 need not wait for position 1's output.
Slot 02 now traces input-derived Q/K/V and independent attention output rows;
this motion idea is also deferred for the later combined animation pass.

Latest editorial pass: follow the meeting/report example, use a hypothetical
“延期” → punctuation continuation, compare report-to-three-bullets with a long
speech, and identify active answer A versus newly submitted document B. Keep
precise dependencies while replacing abstract exposition with short questions,
examples and reading cues. Do not animate hypothetical text as measured output.

| Figure | Static question | Approved motion | Accuracy invariant |
|---|---|---|---|
| 01 | Where does the first output sit? | Input traverses prefill, y1 appears, later outputs follow | TTFT includes queue/input/network; widths not measured |
| 02 | Why needn't position 2 wait for O1? | Project h1/h2 together, route K1/V1 to both reads, produce O1/O2 in parallel | Q/K/V come from the incoming representation; no O1→O2 edge; preserve layer dependency |
| 03 | What is cached when y1 appears? | Keep x1–x4 fixed; reveal causal rows together; y1 appears outside cache, returns, appends its KV, then y2 appears | N KV entries before y1 forward, N+1 after; each layer has its own cache |
| 04 | How does output become the next input? | Feed y1 back, append its per-layer KV, select y2, repeat | y2 KV remains absent until its own forward; old KV retained |
| 05 | Read a long report or write a long speech? | Contrast many input lines/few output lines, then few input lines/many output lines | Same model; illustrative workload sizes, no measured timing or guaranteed bottleneck |
| 06 | Can B's long document interrupt A's answer? | A's decode shares iterations with B1–B3; then contrast separate phase pools | A/B keep independent state; B retains context across chunks; transfer has cost |
| 07 | What can overlap? | Stream layer KV while later prefill layers run | Show transfer tail; preserve layer identity; no zero-cost network |

## Production return — 2026-09-14

Seven independent loops are rendered and integrated, following the approved
seven-figure article order. A combined long-form video is not part of this pass.
The numerical fixture passes; timing is schematic, with no measured advantage.

| Slot | Clip stem | MP4 duration |
|---|---|---|
| 01 | `01-lifecycle` | 9.87 s |
| 02 | `02-parallel` | 7.27 s |
| 03 | `03-first-token` | 8.50 s |
| 04 | `04-decode-loop` | 16.73 s |
| 05 | `05-workloads` | 7.73 s |
| 06 | `06-scheduling` | 12.50 s |
| 07 | `07-handoff` | 10.33 s |

Canonical sources, MP4/GIF exports, per-clip traces and hashes live in
`visuals/animations/projects/prefill-decode-article/`. Article assets are copied
by `build.py --animations`. Each GIF is below 5 MB; original stills and the static
5:2 cover remain. See the animation project's `QA.md` for verification limits.
