# Native Base profile: lower likelihood loss, still unreliable answers

The fixed profile completed all20 updates on Spark with actual child exit0 in
61.636 seconds. Development answer NLL fell from4.061965 to1.100189. However,
both first-eight development generation panels scored0/8 exact answers and0/8
message-end stops: all16 continuations exhausted their64-token allowance.
This verifies the declared training/runtime path, not a successful assistant.

The [predeclared specification](../specs/2026-10-05-native-base-profile.md)
fixes the exact Qwen3-0.6B-Base revision, original240/60 source-disjoint data,
template, seed1212, full tuning, BF16/SDPA, microbatch1, accumulation4,
length256, learning rate2e-5 and20-update horizon. The
[independent review](2026-10-05-native-base-profile/independent-review-01.json)
replays the raw records and journal chains and streams all output hashes without
loading model tensors. Its [collector](2026-10-05-native-base-profile/independent-review-01.py)
and small raw records are retained beside it.

## Observations

| Quantity | Actual observation |
|---|---|
| Unique trainable parameters |596,049,920 |
| Finite-loss/gradient updates |20/20 |
| Training target presentations |455 |
| Training input positions |3,154 |
| Development likelihood targets |360 per panel;720 total |
| Total likelihood-target work |1,175; evaluation is not training exposure |
| Development NLL |4.061965→1.100189 |
| Exact answers / natural message endings |0/8→0/8 for both metrics |
| Generation attempts / returned tokens |16 /1,024 |
| Actual cached forwarded generation positions |1,546 |
| Conservative generation position reservation |66,688; not measured FLOPs |
| External child runtime / exit |61.635627s /0 |
| External sampled minimum MemAvailable |112,855,150,592B;105.10455GiB |
| External memory samples |594; not continuous enforcement |
| CUDA peak allocated bytes |6,027,842,048 |
| Initial / final committed payload |1,503,441,805B /3,887,896,124B |
| Total logical native output bytes |6,606,496,333B; not a filesystem quota |

Both work journals have no open or failed tickets. Every SFT16/I/O9 reservation
fits its declared limit. Actual semantic-validation tensor visits were
2,695,375,062, including10,144 RNG elements omitted from the header-only
projection. Both payload hashes match their commit markers and independently
retained receipt prefixes. The exported weight SHA256 is
`c165204ade77affa09b972b339ee97381424071c8600f2ab465b227b3e266816`.
The actual command, environment, device/driver, source/input bytes and complete
output inventory remain in the JSON receipts.

Owned session signals, observed descendant cleanup and helper cleanup completed
without reported errors. The saved watchdog record labels its future writer ACK
unknown-at-write. The root terminal separately observed a returned true ACK;
do not rewrite the stored record as if it could authenticate that future event.
No cgroup, hostile-process containment, hard output quota or optimized inference
benchmark is claimed.

## Interpretation and next experiment

Teacher-forced NLL measures the probability of gold continuation tokens under
gold prefixes. Exact generation also requires choosing the right first token,
remaining on a useful path under self-produced prefixes, and emitting the
declared ending. Those obligations did not pass here. Including the ending in
training labels does not guarantee that it wins during decoding after20 updates.
The samples retain repetitions and unexpected Unicode/code-like continuations;
they are not removed or repaired for the report.

This result motivates the already declared full/LoRA smoke, completed-state
recovery and fixed400-update comparison, not an adaptive sweep for a pleasing
answer. The profile's weights are disposable and are not the selected SFT parent.
The separate [replay acceptance](../specs/2026-10-05-native-sft-replay.md) keeps
the same20-update horizon and compares an uninterrupted tail with a fresh
update10→20 process. Its first uninterrupted full run exited0; the first resume
was safety-stopped at4.550s with exit-15 when a concurrent runner-named CPU test
was detected. That attempt remains retained, and no recovery pass follows from
it. Retry only after conflict clearance, in a new output, without refunding work.

The [current campaign evidence](2026-10-05-native-base-profile/current-campaign-evidence-01.json)
populates only the actual profile row and enumerates those two auxiliary
invocations. All45 planned rows survive; unexecuted rows remain null. The frozen
October4 preparation remains unchanged. A root collector's first mistaken
assumption about that report's nesting is retained separately as a collection
failure, not attributed to the model.

The18-package goal remains incomplete. This profile supplies actual pretrained
runtime/interface evidence; publication evaluation, merged LoRA, downstream
preference/RLVR comparisons, controlled story ratings, actual Mac/hosted checks
and final capability/cost/regression defense still need their own evidence.
Learner Day9 and the original acceptance criteria are unchanged.
