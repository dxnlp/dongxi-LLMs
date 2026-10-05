# Actual pretrained full and LoRA completed-update replay

Both fixed native paths now pass fresh-process update10→20 recovery on the
exact acquired0.6B Base, original source-disjoint data and saved interface.
The ten numerical update rows, same-producer serialized tensor entries and
timing-free final token/stop records match their uninterrupted counterparts
exactly. This is same-Spark BF16/SDPA evidence, not a toy result extrapolated to
pretrained weights, cross-machine equality or arbitrary pickle-metadata proof.

The [premeasurement specification](../specs/2026-10-05-native-sft-replay.md)
keeps the20-update horizon. Both arms start fresh from Base, not the disposable
profile's trained weights. The full arm's
[acceptance](2026-10-05-native-sft-full-replay/acceptance-retry02.json) and the
LoRA arm's [acceptance](2026-10-05-native-sft-lora-replay/acceptance.json) retain
every numerical tail, serialized tensor digest and actual child outcome.
Raw likelihood/generation records are copied beside those receipts.

| Quantity | Full SFT | Rank-8 Q/V LoRA |
|---|---:|---:|
| Trainable parameters |596,049,920 |1,146,880 |
| Reference updates / shifted targets |20 /455 |20 /455 |
| Fresh replay updates / extra targets |10 /229 |10 /229 |
| Reference initial development NLL |4.061965 |4.061965 |
| Reference final development NLL |1.100189 |3.691564 |
| Reference final exact answers |0/8 |0/8 |
| Reference final message-end stops |0/8 |0/8 |
| Reference CUDA peak allocated bytes |6,027,842,048 |2,055,486,464 |
| Reference child elapsed / exit |73.494630s /0 |61.429519s /0 |
| Successful fresh replay elapsed / exit |81.304204s /0 |58.601126s /0 |
| Cumulative successful likelihood-target work |2,124 |2,124 |
| Cumulative snapshot saves / inspect / load |5 /1 /1 |5 /1 /1 |

The reference and replay each run complete before/after60-item development
panels. They add1,440 evaluation target presentations to684 actual training
target presentations. The numerical resumed trajectory's cumulative training
counter is455; the physical work journal correctly counts684 because the tail
was executed twice. A checkpoint reload does not refund its already executed
future. All reservations fit the unchanged pair caps and both physical journals
have no open or failed tickets in their final successful state.

The full arm's earlier resume was safety-stopped at4.549661s with exit-15 on a
concurrent runner-named CPU test. Its immutable failure/returned supervisor
records remain. The [retry declaration](../specs/2026-10-05-native-sft-full-replay-retry.md)
waited for clearance, used a new output and retained the same checkpoint10
expectation, independently retained contract/receipt and physical journals.
Include that failed attempt in wall cost; it is not erased by later success.

Full final generation exhausted all eight64-token caps. LoRA final generation
exhausted six caps and stopped twice on generic EOS, not the required message
ending; all eight answers were incorrect. This is not evidence that either arm
is a useful assistant, nor a fair best-recipe comparison. Both used the same
declared learning rate and short horizon, and neither was selected for quality.
The fixed400-update pilots, frozen publication comparison and selected full-SFT
parent remain separate unexecuted campaign rows.

The serialized-entry check hashes every tensor storage entry without executing
pickle metadata:1,243 entries in the full final snapshot. Numerical histories
and final token records provide additional independent replay checks. It does
not certify arbitrary Python RNG metadata, a hostile archive/deserializer,
whole-output physical quotas or other hardware/backend versions. Export and
independent archive hashing remain outside SFT16/I/O9 and are labeled separately.

## Merge is a separate handoff, not implied by recovery

The subsequent original BF16 merge diagnostic failed its predeclared logit
tolerance:5.9% of first-prefix elements mismatched, maximum absolute discrepancy
0.67578125. The [failure](2026-10-05-native-lora-merge-reload/failure.json) and
actual exit1 are retained; no merged output was exported. Exact adapter recovery
does not establish BF16 merge equivalence. A separately declared FP32 validation
artifact tests a wider representation without rewriting that failed result.
Its [specification](../specs/2026-10-05-native-lora-fp32-merge-reload.md) explains
the numerical change and excludes selected-parent or original-BF16-policy claims.

That distinct FP32 diagnostic subsequently passed all eight prefix comparisons
at its declared absolute0.002/relative0.001 tolerance. Maximum absolute logit
difference was0.00005364418029785156; every last-position argmax agreed.
Saving and freshly reloading the merged full FP32 artifact produced bitwise
identical logits on those eight prefixes and retained the complete tokenizer,
template and stopping interface. Its actual external child exited0 in41.636924s
with minimum sampled host availability117,331,619,840bytes. See the
[verification](2026-10-05-native-lora-fp32-merge-reload/verification.json) and
[returned supervisor receipt](2026-10-05-native-lora-fp32-merge-reload/returned-supervision.json).
The full weight file is2,384,234,968bytes, SHA-256
`fe09c2d32e3ce6b136d1b8b1cedec787ff198feb9e6835017ec4bad7ad65c8b7`.
Its role is an explicitly merged FP32 validation artifact, not the selected
400-update SFT parent. Both precision-specific results remain in the record:
passing FP32 merge/reload does not retroactively pass the failed BF16 comparison.
