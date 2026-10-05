# Current model genealogy and experimental conclusions

Historical short-run snapshot: the observations below precede the accepted
400-update runs. The later
[full and LoRA comparison](2026-10-05-native-sft400-comparison.md) now establishes
actual selected full400 ancestry and the original120-item publication results.
The [literal platform review](2026-10-05-platform-scope-reading.md) supersedes
the stronger mandatory-Mac interpretation in this historical snapshot. Preserve
its old measured failures and costs; do not treat its missing-stage counts as
the current campaign inventory.

The actual evidence supports identity, recovery and evaluation-instrument
claims, but not a successful assistant or a controlled story-quality gain.
This review completes the defense that can be written from existing results.
It is not the final DXI-18 campaign defense: the selected400-update policy,
story comparison and DPO/RLVR pilots remain unrun.

## Checkpoint genealogy

| Actual artifact | Actual parent | Operation | Defensible role |
|---|---|---|---|
| September DongxiGPT | Fresh random weights; GPT-2 tokenizer only |14,000 TinyStories updates | Measured pretraining baseline, not a controlled new arm |
| Native disposable profile | Exact Qwen3-0.6B-Base revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |20 full-SFT updates | Profile/interface artifact; not the selected SFT parent |
| Full-SFT replay reference | Same acquired Base, freshly restarted |20 updates and actual fresh10→20 continuation | Completed-update replay evidence |
| Rank-8 Q/V LoRA replay reference | Same acquired Base, freshly restarted |20 adapter updates and actual fresh10→20 continuation | Adapter recovery evidence; not a merged policy |
| FP32 merged validation export | Exact acquired Base plus the20-update LoRA adapter | Separate FP32 merge/save/reload | Precision-qualified validation artifact, not a BF16-policy pass |
| Selected400-update SFT, DPO and RLVR policies | Unestablished | Unrun | No actual child identity or capability claim |

The [current campaign snapshot02](2026-10-05-native-base-profile/current-campaign-evidence-02.json)
binds actual receipts and genealogy maps. The profile's exported weight digest
is `c165204ade77affa09b972b339ee97381424071c8600f2ab465b227b3e266816`;
the FP32 export's weight digest is
`fe09c2d32e3ce6b136d1b8b1cedec787ff198feb9e6835017ec4bad7ad65c8b7`.
These are child weight-file hashes, not upstream revision names. Snapshot01's
incorrect parent-digest placement remains retained alongside the explicit02
correction. The full and LoRA references are separate fresh runs; their similar
trajectories do not make them aliases of the profile output.

## Capability and regression evidence

| Instrument and population | Measured outcome | Conclusion and limit |
|---|---|---|
| Original TinyStories fixed development prediction | Final NLL1.674315; inspected stories contain repetition and inconsistencies | Prediction improved; systematic five-dimensional coherence and a matched generalization gap are not measured |
| Native profile60-item development NLL |4.061965→1.100189 | Better teacher-forced likelihood on this fixed population, not generated answering |
| Native profile initial/final eight-answer grid |0/8 exact and0/8 message endings in both; every output capped | No measured answering/stopping gain in this diagnostic |
| Full/LoRA final diagnostic grid | Both0/8 exact and0/8 message endings; LoRA has two generic-EOS stops | Recovery succeeds despite negative behavior; generic EOS is not the required message ending |
| Common15-item Base/short-SFT development replay |0/15 versus1/15 automatic passes; both0/15 natural stops and15/15 caps | One proxy gain, not one clean useful answer or a broad capability gain |
| Independent whole-response review | All30 responses inspected; partly useful SFT content with artifacts and truncation; no clean completed requested answer | Separate unblinded AI review, not blinded human agreement or calibrated safety evidence |
| BF16 merge diagnostic | Failed declared tolerance; max absolute difference0.67578125 | Original BF16 handoff remains failed |
| Separately specified FP32 merge/reload | Eight comparisons pass; maximum absolute difference0.00005364418; fresh reload logits exact | Precision-specific numerical/interface success, not improved task behavior |

The common development difference is0.066667 with a paired source-group
bootstrap interval[0,0.214286] under the frozen instrument. These fifteen
authored items from fourteen groups cannot establish general improvement.
Twelve format-valid records per arm mostly reflect an unconstrained `any`
policy, not twelve clean answers. Parser `UNSUPPORTED` reports bounded grammar
coverage, not a general mathematical-ability failure. The
[complete response report](2026-10-05-pretrained-evaluation-replay.md) retains
raw answers and these distinctions.

## Cost boundaries and failed work

| Actual work | Observed cost | Boundary |
|---|---|---|
| Original TinyStories training |12,102.9s;48,839,975 valid training targets | September run; separate from October native jobs |
| Disposable native profile |61.635627s;20 updates/455 training targets | External whole-child timing; observers have separate work |
| Full replay pair plus failed attempt |73.494630s+81.304204s+4.549661s | Three actual invocations; the failed attempt is not refunded |
| LoRA replay pair |61.429519s+58.601126s | Two actual invocations |
| Each successful replay pair |684 training plus1,440 development targets;5 saves/1 inspect/1 load | Physical exposure is2,124 targets; numerical final training counter is455 |
| Common Base/SFT response panels |35.833090s+36.279153s;1,920 emitted tokens;94,656 full-prefix positions | Two generation children; no KV cache; offline grading adds no model forwards |
| Failed BF16 / passed FP32 merge children |14.336323s /41.636924s | Two distinct precision-specific experiments |

Do not sum overlapping target counts or add host/CUDA peak memory as if they
were one resource. The [native replay report](2026-10-05-native-sft-replay.md)
states trainable parameters, measured CUDA peaks and per-child external
sampled host availability. Sampling is not a continuous-memory guarantee.
Export/archive inspections and offline review are separately labeled work;
no universal total-compute or energy measurement is available.

## Remaining evidence

The45-row campaign keeps only its native profile row populated;44 unrun rows
remain null. Ten actual native/model invocations include two failures and nine
auxiliary jobs; auxiliary jobs do not become planned400-update or publication
results. The frozen story panel still needs real model continuations and
independent ratings before any coherence comparison is possible.

DXI-17 requires an actual supported Mac receipt. Its original CI criterion
requires honest separation of source/local checks and hosted execution, not
a new compulsory hosted success. The [Mac verification packet](../../docs/handoffs/MAC_CPU_VERIFICATION.md)
states the exact return evidence without pretending this Spark task can execute
on Mac. Original dependencies remain unchanged. When platform and branch gates
pass, extend this defense with actual matched campaign results—including
negative results—rather than replacing missing measurements with forecasts.
