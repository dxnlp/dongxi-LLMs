# Systems and scaling reference notes

These original CPU companions support Assignments C and D. Each prediction has
its runnable reference and mechanism explanation directly in the notebook.
This guide collects the decisive acceptance questions; it does not replace the
learner's argument or turn a passed implementation into assessed mastery.

| Companion | Controlled change | Reference and decisive check | Evidence boundary |
|---|---|---|---|
| [Resource accounting](01_resource_accounting.ipynb), checkpoints1–2 | Matrix size or query tile, holding dtype/model dimensions fixed | `MatmulLedger`:2mkn FLOPs and b(mk+kn+mn) bytes. A[2,3]B[3,4] FP64 gives48FLOPs/208bytes. `attention_ledger` explicitly changes K/V reread count. | Traffic is modeled under a stated reuse policy; no memory-controller counter. |
| Resource accounting, checkpoint3 | Compute ceiling or bandwidth alone; asynchronous submission control | Roofline=min(peak,I×bandwidth); hypothetical100GFLOP/s and20GB/s have ridge5FLOPs/byte. `time_calls` synchronizes before/after timed work. | The hypothetical curve cannot calibrate actual CPU utilization. Raw timed matmul throughput excludes allocation and transfer. |
| Resource accounting, checkpoint4 | Existing `StoriesDecoder.activation_checkpointing` only | Same weights/tokens, FP64 loss/gradient tolerance1e-12; two extra block entries during checkpointed backward. | Saved storage includes parameters and aliases; no peak-memory or speedup inference. |
| Resource accounting, checkpoint5 | Prefix length only | Cached new-token attention work grows with prefix; projections/MLP/head run once. Reuse decoder parameter/KV estimator. | Payload and counted matmul work are separate from actual cache speed. |
| [Online attention](02_online_attention.ipynb), checkpoint1 | Omit old-statistic rescale; direct exponentiation | Scores[0,1000],values[0,1]: stable output1, omitted-rescale output0.5, direct-exp overflow. | Original arithmetic fixture, ordinary autograd. |
| Online attention, checkpoints2–3 | Tile partition, future values, ragged length or query offset | Independent SDPA output and gradients: FP64 atol/rtol1e-11, FP32 3e-6. Invalid rows output0,LSE−∞; zero masked contribution. | Tested shape/mask contracts, not a fused GPU kernel. |
| Online attention, checkpoints4–5 | Length at fixed tiles; remove causal mask | Separate modeled workspace from forward saved storage. Causal-mask removal changes semantics. | No recomputed backward and no backward-memory claim. |
| [Scaling prediction](03_scaling_prediction.ipynb), checkpoints1–2 | Compute budget, with authored surface fixed | Under C=6ND, symmetry gives N*=D*=√(C/6), exponents0.5 and−0.25. Freeze384M prediction before generating the target grid; all15 target rows retained. | Known floor and functional form are authored; no external training result. |
| Scaling prediction, checkpoints3–4 | Replicate noise, sparse budgets, architecture, data policy or grid boundary | Resample within size/budget; keep failed fits. Refuse fewer than3 calibration budgets, missing contract identity, invalid logs and boundary minima. | Conditional intervals omit extrapolation/form/hardware uncertainty; labels require source review. |
| Scaling prediction, checkpoint5 | Three tiny widths across four budgets, two seeds | Frozen CPU spec, smoke, complete24-run receipt;3 calibration budgets and1 heldout budget. Report target and actual6ND, train/dev/test, unique slots and repetitions. | Finite repeated synthetic corpus; an unbracketed optimum freezes a refusal, without retuning. |

For a Spark extension use `scripts/run_cs336_systems.py --mode profile --device
cuda --output <new receipt>` only inside the separately declared profile. Record
the actual device/precision/backend evidence and raw repetitions. Single-device
GB10 results do not reproduce Stanford's two-B200 systems leaderboard. A backend
name does not establish the exact fused kernel. Publication and distributed or
Triton production remain separate scopes.
