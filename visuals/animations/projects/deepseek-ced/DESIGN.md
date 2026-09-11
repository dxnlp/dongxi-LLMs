# DeepSeek CED: two motion-first animation designs

- Requested: 2026-09-10; English animation design for two specific mechanisms.
- Status: implementation/rendering explicitly approved on 2026-09-10 after design
  review. See README, trace, render metadata and QA for return evidence. Durations
  below remain original editorial targets, not measured media.
- Owner: Mac Studio. No Spark model run is needed for the proposed small fixtures.
- Scope: two short explanatory films. No article, publication, commit, or push.
- Base checkout: `codex/visuals/looped-transformer`; pre-existing uncommitted
  animation and lesson work preserved. No pull attempted on the dirty tree.
- Reuse: CAND-ANIM-009 and CAND-ANIM-017; approved production packets are
  ANIM-CED-001 and ANIM-KV-002 in LEARNING_MEMORY.md.

## Shared visual direction

Follow `../../STYLE_GUIDE.md` and `../../manim_style.py`: white 16:9 canvas,
normal-weight Arial, large geometric objects, fixed anchors, minimal English text,
and causal motion. Target 1920x1080/30fps with 960x540 inspection. These are films,
not X covers; the 5:2 cover convention does not apply.

Both films use the same visual vocabulary:

- A token state is a short horizontal vector strip; a cache is a vertical stack
  of strips. Never use a cylinder, database metaphor, or literal word meanings.
- Blue: current input/query. Purple: global KV representation. Green: resulting
  attention output. Neutral outlines: frozen computation structure. Amber is
  reserved for a brief distinction or omitted-detail caveat.
- Local KV retains its own small outline and `Local KV` label; identity is not
  conveyed by color alone. No decorative meteor imagery or model logos.
- Stored entries remain anchored while reads send temporary signals or visibly
  transient contributions. A read never removes or permanently duplicates data.
- Keep no more than three active readers in view. Use an ellipsis and one range
  label for omitted layers/heads. Never pretend three drawn blocks are the whole
  released model, or that a few drawn cells are its full hidden dimension.
- Define source creation and read access in separate beats before showing both.
  Pause on each resolved layout; avoid simultaneous wire rewiring and token motion.
- Use Manim-native geometry for these tensor-focused films. Excalidraw remains an
  option for a future whole-system map, not a required decorative layer here.

## Film 1: Where global KV comes from

**Question:** Must every decoder layer build historical global KV from its own
hidden states?

**Target:** 55 seconds. One stable three-layer detail view with the encoder
boundary above it. Never show all 40 layers individually.

| Target time | Motion and composition | Minimal visible text |
|---|---|---|
| 0-8 s | In a generic layer-local reference, follow a state through three successive layers. Each layer independently writes its own KV bank. Reveal one write at a time; previous banks stay still. | `Layer-local KV`; `21`, `22`, `23`, `…40` |
| 8-16 s | Isolate the last bank. Trace its creation dependency backward through earlier layers, using one moving focus marker. Stop at the state entering that layer. | `Source: current layer` |
| 16-28 s | Introduce CED as an architectural change. Clear the reference banks, retaining the layer positions. Reveal final encoder states H20 at the boundary; their signals feed separate global-KV projections for the three displayed decoder layers. The main residual path remains continuous. | `CED`; `Encoder output`; `Global KV` |
| 28-39 s | Add CSA2 sharing as a distinct second change. Keep the L21-created global bank and remove the redundant *storage placeholders*. Later decoder layers' read paths now converge on that one bank. This is a configuration change, not arithmetic fusion or a claim that the earlier projected values were equal. | `+ Cross-layer sharing`; `21–40` |
| 39-50 s | Run the current token through the three visible decoder layers. Each produces a visibly different query; each also writes its own small local cache. Read the fixed shared global bank and the current layer's local entries together; produce a distinct output at each depth. | `Q`; `Local KV`; `Global KV` |
| 50-55 s | Hold the resolved composition: one encoder-derived global bank, three distinct readers and local caches, continuous depth path. | `Encoder-derived global KV` / `Layer-local Q + local KV` |

**Essential separation:** CED changes the source; CSA2 sharing changes how many
global banks are stored. The intermediate fan-out is a conceptual CED-only view,
not the actual released decoder configuration. Mark it `CED` and the actual
configuration `CED + sharing`; explain this in the companion page.

**Narration, if later requested:** “Ordinary layers construct their own memory.
CED moves the source of decoder global memory to the encoder output. Cross-layer
sharing then lets later layers reuse one bank. Each layer still forms its own
query and local memory.” No narration is required to follow the objects.

**Keep out of this film:** sparse index selection, cache precision, byte totals,
performance bars, and bounded replay. Prefill savings can be mentioned in adjacent
prose only with the local-replay exception. A separate future film should explain
the approximate replay dependency cone; do not squeeze it into the ending.

**Truth checks:** a global read is not weight sharing; the current token still
traverses the decoder; local KV does not migrate to H20; historical cache content
must be regenerated when changing the architecture rather than depicted as
numerically unchanged under new projections. The encoder has its own caches,
omitted here, so never label the global decoder bank “the model's only cache.”

## Film 2: One KV representation, two attention roles

**Question:** Can one stored vector participate both in matching a query and in
providing the information accumulated into the output?

**Target:** 55 seconds. Begin with two familiar roles; then show the released
core-attention representation. The cache is the stable center of the scene.

| Target time | Motion and composition | Minimal visible text |
|---|---|---|
| 0-9 s | A generic reference input feeds separate K and V projections. A query compares with K; the resulting weight gates V into an output. Keep score generation and payload contribution sequential. | `Separate K / V`; `K`; `V`; `Q` |
| 9-18 s | End the generic reference visibly, then introduce the shared-KV design at the same spatial anchor. A single projection emits a vector c. The old K/V arrays do not collide, shrink, concatenate, or morph into c. | `Shared KV`; `c` |
| 18-30 s | Expand to three stored rows. A query visits them for dot products; aligned score marks become normalized weights. Keep the stored rows unchanged. Show the two-dimensional toy's signed values consistently; every displayed weight must come from its calculation. | `Score`; `Normalize` |
| 30-41 s | Use those weights to send transient, scaled contributions from the *same rows* into the output. Preserve row identity across both uses. The contributions accumulate; the bank remains intact and available. | `Weighted sum`; `Output` |
| 41-50 s | Repeat with another query head, then reveal three head outputs together. All heads read the same bank but produce different weights and mixtures. Show the additional head count as a compact scope label, not dozens of wires. | `Shared across heads`; `3 of 64 heads shown` |
| 50-55 s | Return attention to the unchanged bank and the two operation paths, with three distinct outputs at the edge. | `One representation. Two roles.`; `Main attention only` |

The dimensional expansion must be honest: the numerical microscope uses three
rows and two coordinates; explicitly mark `2D example` while toy values appear.
Only the final architecture-scale beat refers to the released 64 query heads.
Do not relabel a toy two-coordinate row as an actual 512-coordinate activation.

**Precise kernel grounding:** the released sparse-attention kernel loads a single
`kv_shared` block, uses it in `q_shared @ kv_shared.T` for scores, then again in
`weights @ kv_shared` for output accumulation. This is stronger than saying K/V
are merely packed next to each other. The same selected numerical representation
serves both roles inside this kernel.

**Omitted operations:** the teaching microscope fixes the selected entries and
omits the attention sink, RoPE, output projection, quantization, and full indexer.
Actual core attention receives selected global entries together with layer-local
entries. Present the toy as a selected KV pool, not a persistent merger of those
two caches. The global/local storage distinction from Film 1 remains valid.
There is also indexer K and other runtime state; never imply the entire system
stores exactly one tensor. Avoid any unsupported “half the total memory” claim.

## Planned numerical and structural fixtures

These are predeclared tests for production, not completed experiments:

1. Film 1 fixture: a small causal lower stack and three distinct upper layers.
   Give each upper layer its own Q and local-KV projections while sharing one
   encoder-derived global bank. Verify object/storage identity, unchanged global
   entries across upper-layer reads, and differing upper-layer outputs. Reference
   versus CED variants do not have to produce equal outputs.
2. Film 2 fixture: three fixed KV rows `[[1,0],[0,1],[-1,0]]` and three queries
   `[[2,0],[0,2],[-2,0]]`. Use scale `1/sqrt(2)` and no sink. Prediction: the
   first and third queries favor opposing rows; the second favors the middle.
   Compute softmax weights and weighted sums from these exact arrays.
3. Verify that making separate equal-valued copies `K=C`, `V=C` agrees with the
   one-buffer arithmetic; do not imply those copies are independently learned.
   Mutating an independently parameterized V should break that equivalence.
4. Keep all toy sources fixed during each clip; query changes are controlled
   interventions, not training. No arbitrary glow intensity, invented attention
   score, measured speedup, or semantic interpretation of the row coordinates.

## Production gate and planned return

After approval: create two explicit ANIM sub-packets with base revision, allowed
files, source hashes, precise toy traces, and executable checks. Implement under
this project using shared Manim primitives. Make separate English players and
one series page. Deliver reproducible source, trace JSON, local 1080p MP4s,
lightweight GIFs, stills, contact sheets, and metadata. MP4 files remain ignored.

Acceptance must cover full-sequence and transition inspection, natural kerning,
phone-scale readability, no data disappearance on reads, no false K/V fusion,
one-based layer numbering, separate CED/sharing stages, codec and dimensions,
source/media hashes, and real player playback. Keep internal task identifiers,
toolchain names, and review-status language out of the media and learner pages.

## Primary sources and course placement

Canonical discussion:
`learning_artifacts/day-06-modern-architecture/deepseek-v41-causal-encoder-decoder.md`.
Chapter 5 frontier extension; no change to Day 8 learner progress.

All external source references are pinned to
`df42c109f1defefcbfcedbe7d905718a12266e40` in
`deepseek-ai/DeepSeek-V4.1-Flash`:

- [Report, sections 2.2-2.3 and Figure 4](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/DeepSeek_V41_Tech_Report.pdf).
- [Model: Attention and Compressor](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/model.py).
- [Kernel: sparse_attn_kernel, score and output GEMMs](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/kernel.py).
- [Configuration: source layers, query heads, and dimensions](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/config.json).

Draw original diagrams. Do not copy report figures or substantial external code.

Approved cover exception (2026-09-10): use the original report's Figure 3 as the
shared HTML video poster, leaving both animation streams unchanged. Preserve the
source artwork, attribution, pinned extraction provenance and repository MIT
license in `assets/`; this exception does not apply to the animated diagrams.
The kernel is read as mechanism evidence; the proposed toy will be independently
implemented. Static source inspection is not execution of DeepSeek's model.
