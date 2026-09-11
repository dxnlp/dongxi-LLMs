# DeepSeek V4.1: causal encoder, decoder, and KV dependencies

- Opened: 2026-09-10, at the learner's request; first source-grounded explanation.
- Placement: Chapter 5 frontier architecture/KV discussion, bridging Chapter 4.
- State: report and reference code inspected; learner prediction and mastery pending.
- This user-led topic takes precedence for this conversation. It neither completes
  earlier practice nor cancels Day 8's separately recorded Spark learning plan.

## Initial question and learning objective

The learner associated causal attention with decoders and asked whether a causal
encoder exists, then requested a deep dive into DeepSeek's CED and KV flow.
Separate a component's encoder/decoder role from the causal information boundary.
Explain source creation, cross-layer reuse, sparse selection, and runtime caching
as different mechanisms. Do not call cache sharing weight sharing or recurrence.

## Primary evidence

Pinned Hugging Face revision: `df42c109f1defefcbfcedbe7d905718a12266e40`.

- [Technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/DeepSeek_V41_Tech_Report.pdf):
  Figure 3 and sections 2.1-2.2, Figure 4 and sections 2.3.1-2.3.2,
  and sections 3.2.1-3.2.2. Architecture and serving claims are author reports,
  not locally reproduced performance measurements.
- [Reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/model.py):
  `Attention`, `Compressor`, `Indexer`, `SharedAttentionRuntime`, `Block.forward`,
  and `Transformer.forward`.
- [Configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/config.json).
- [Implementation boundary](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/README.md):
  explicitly a readable reference, not a production serving engine.

## Mechanism and precision boundaries

All layer numbers below are one-based; config/source identifiers are zero-based.

1. Forty causal language-backbone layers comprise encoder 1-20 and decoder
   21-40. The separate vision encoder is not the causal language encoder.
   Prompt and subsequently generated tokens both pass through the causal encoder.
2. In ordinary self-attention, each layer produces its Q and K/V from its own
   normalized layer input. CED changes the source of decoder **global** KV to
   the final encoder representation, without changing the source of main Q or
   layer-local sliding-window KV.
3. Report equation (1) gives C_l = H_20 W_KV_l and Z_l = H_20 W_Z_l for decoder
   global KV entries and compression weights. This expresses the source
   dependency before considering CSA2 sharing; it does not imply twenty stored
   decoder global banks. The released decoder uses compression ratio 1, so its
   main-KV path does not pool adjacent tokens or need compression gates.
4. CSA2 Full creates main KV and indexer K and computes selection indices.
   Reindex shares those KV/keys, but creates a new indexer query and selection.
   Reuse shares main KV and the latest selection. All three compute their own
   main Q and SWA KV. Shared indices do not imply identical main-attention weights.
5. Global banks are produced at layers 3, 9, 15 (encoder ratio 2) and 21
   (decoder ratio 1). Layer 21's bank serves all decoder layers. Decoder index
   producers are 21, 25, 29, 33, 37; intermediate layers reuse the last producer's
   selections for the same query position, not a previous token's query.
6. The decoder's first indexer creates its own Top-512 selection and a broader
   candidate pool; later reindexers choose up to 512 entries from that pool.
   Candidate pool size is not the same as final selected-entry count.
7. The reference main attention uses 64 query heads of width 512 against a shared
   KV representation of width 512. Q has logical shape [B,T,64,512], decoder
   global cache [B,T,512], and each local cache [B,128,512]. These describe tensor
   geometry, not packed physical bytes. It is not ordinary separately materialized
   per-head K and V. Hyper-Connections, normalization, low-rank projections,
   RoPE, sink terms, and quantization are omitted in the dependency schematic.
8. The main query attends to selected global entries **together with** its own
   local-window KV in one core attention operation. The indexer query performs
   selection, whereas the main query determines final attention weights.

## Prefill, decode, and approximation

For a cold prompt, run the encoder for all positions, obtain the decoder global
bank from its final outputs, then run the decoder on a bounded prompt tail to
reconstruct local KV and obtain the first output distribution. During ordinary
autoregressive decoding, each new input token traverses both halves; encoder
banks update as complete compression groups become available, decoder global KV
gets the new ratio-1 entry, and each visited layer updates its local window.
The final prompt position predicts the first generated token; processing that
generated token predicts the following one.

The report's decoder bounded replay uses the final 128 prompt positions. Exact
reconstruction has a deeper local dependency cone, on the order of decoder
depth times window width. Bounded replay truncates that cone and is explicitly
approximate. The authors report negligible response-quality impact and adaptation
during post-training; neither claim is independently reproduced here.

On an encoder prefix-cache hit with missing local state, encoder bounded replay
reuses cached global entries without overwriting them and regenerates local KV
on a recent prefix tail. Newly processed suffix states can consequently depend
on the cache-hit boundary. This is an approximation, not a violation of ordinary
exact causal-cache immutability under unchanged computations.

Important implementation boundary: the inspected public `Transformer.forward`
iterates through all backbone layers on the supplied prefill chunk. Do not claim
the minimal reference implements or benchmarks the report's bounded-replay
serving shortcut. The reference exposes the cache-sharing dependencies.

## Interactive lesson and next prediction

Conversation-native dependency view:
`/Users/yongchao/.codex/visualizations/2026/08/31/01a05939-8cfa-7672-9da0-efe2a11b0d26/deepseek-ced-kv-flow.html`.
Default layer 27; selector 21-40 derives mode, global source 21, and latest index
producer. It is an architecture schematic, not live model activations, measured
attention, or an MP4. Source read back; no browser playback verification claimed.
Thread-local path is not a cross-machine Git export.

Prediction to ask before a numerical experiment: if layers 26 and 27 share both
global KV and selected indices, must their attention weights or outputs match?
Expected reasoning: their main queries, local KV, and surrounding transformations
remain distinct. Learner response is not yet recorded.

Next lesson: trace two layers and one new token through a tiny exact causal model;
then compare full-prefill and tail-replay states. Predict unchanged shared global
KV but potentially different reconstructed local states. No experiment has run.
Canonical chapter prose and trained-quality claims remain unchanged. A book-facing
frontier sidebar can be integrated during the next Chapter 5 synthesis after the
learner confirms the explanation's scope.

## Animation opportunity

Subsequent user approval on 2026-09-10 authorized two English films, now
implemented and locally rendered: [CED and shared KV](../../visuals/animations/projects/deepseek-ced/README.md).
The independent fixture passes nine checks, including shared-storage identity,
causal prefix invariance, normalized weights, and equal-copy arithmetic.
This is new toy evidence, not DeepSeek model validation or demonstrated learner
mastery. The bounded-replay experiment below remains unexecuted and unapproved
for film production. No article or public publishing action was taken.

Extend CAND-ANIM-009, with CAND-ANIM-017's source-sharing comparison: animate the
global source moving from per-layer hidden states to H20, retain distinct local
states/queries, then show the prefill dependency cone and bounded approximation.
Source: user question plus agent automatic mathematics/dependency check. Evidence:
primary report and static code inspection only. Production requires a tiny
reference trace, verified exact-versus-approximate boundary, and explicit approval
for Mac animation production. This lesson does not authorize rendering or an article.
