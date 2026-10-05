# Accounted recovery of the chosen-only SFT control

The original chosen-only control now has an opt-in, counted completed-boundary
save and restore API. Both original fixture seeds recover in a genuinely fresh
CPU process with exact model, Adam, sampler, global RNG, numerical history and
next-draw agreement. Recovery retains the deliberately failed later update's
cost instead of restoring an earlier resource allowance. This closes a source
and tiny-CPU recovery gap; it is not a pretrained preference result.

The [premeasurement specification](../specs/2026-10-05-chosen-sft-accounted-recovery.md)
defines the unchanged objective and verification. The current
[verification receipt](2026-10-05-chosen-sft-accounted-recovery/run-02/verification.json)
contains actual source hashes, observed environment, command exits and complete
fresh-process results. Its SHA256 is
`d2fc0f0a3dd2ce531a6b0b72e21f2db991a178992b5cdfda9199ea68b9ac8978`.

## What stays scientifically unchanged

Training still uses the existing eight location preference pairs and the native
DPO private CPU-generator replacement draws. Chosen-only SFT still minimizes
native summed chosen-token NLL divided by every valid chosen target in the whole
accumulation window. The terminal token remains supervised. Each forward uses
the complete chosen sequence, whereas native DPO uses shifted input sequences.
Rejected supervision and reference forwards remain zero in the chosen arm.
The original parent, matching recipe, validation/publication splits and existing
[negative held-out results](2026-10-05-matched-chosen-sft-control.md) are unchanged.
The existing `run_arm`/CLI comparison path remains unaccounted by default; this
extension makes no new quality or equal-compute claim for that historical route.

## Recovery and accounting boundary

`make_chosen_recovery_contract` re-encodes the actual raw fixture and binds its
masks, tokenizer template/mapping/stops, original-parent files/tensors, native
sources, selected lock, observed packages, fixed recipe, effective zero-dropout
model, declared managed precision and named AdamW parameter order. Arbitrary
opaque autocast factories are refused by the counted route. CUDA/BF16 execution
has not been verified here.

`ChosenSFTLoop.bind_recovery` binds the retained science to its physical
nineteen-dimensional work journal and nine-dimensional shared snapshot I/O
journal. Before any active sampler draw, `completed_update` reserves the entire
accumulation window. Entered and completed forwards remain separate on failure.
An interrupted or partially applied optimizer update poisons the live numerical
state and cannot be saved.

Saving validates completed numerical history and its journal coverage inside a
separately charged semantic panel, then publishes the existing exclusive shared
snapshot and independent receipt. Restoring first binds that independently
retained receipt to exact payload bytes and both physical journal prefixes.
Shared loading then verifies receipt/bytes and the chosen-specific model, Adam,
draw history and RNG semantics before applying state. Restoring numerical update2
does not remove the later failed update3 ticket or give its capacity back.

These are cooperative logical work and shared I/O controls for trusted local
artifacts. They are not FLOP measurements, physical containment, an external
deadline, a whole-output quota or a model-scale launcher. Generation and held-out
observer integration into an actual counted campaign invocation remain separate.

## Actual CPU observations

The final panel passed all36 tests:14 new recovery controls plus22 existing
chosen-control tests, with no failures, errors or skips. Unittest reported9.276s;
its actual parent process exited0 in11.267s. Each additional retained fresh child
had a60-second ceiling and exited0:2.181s for seed1818 and2.182s for seed1819.
All13 selected source/specification/original-report hashes stayed unchanged
through this final collection.

Both seeds use six updates and accumulation2. The fresh child restores after
update2, with a deliberately failed second forward of update3 already retained
in the same journal, and continues2→6. Exact agreement covers model weights,
optimizer moments/steps, private and global RNG, all six numerical metric rows
and the next replacement-draw pair.

| Logical training quantity | Completed numerical trajectory | Permanent reserved capacity |
|---|---:|---:|
| Updates | 6 | 7 |
| Private sampler draws | 12 | 14 |
| Chosen target presentations | 48 | 56 |
| Full chosen input positions | 288 | 336 |
| Policy forwards | 12 | 14 |
| Reference forwards | 0 | 0 |

The failed update retained two selected draws/eight chosen targets. Its first
forward completed24 input positions; its second entered24 positions and failed.
Those second-forward backend results remain uncertain rather than invented.
Successful semantic save/load checks separately reserve two validation panels,
four history rows,93,212 tensor-verifier elements, four RNG probes and eight
private replay draws. Each retained arm records one actual shared save and load.

The negative controls verify whole-window refusal before draw/backend,
reservation-fsync uncertainty before work, poisoned partial forwards,
post-optimizer failure recovery without a refund, prior-checkpoint preservation
after serializer failure, old-boundary cap exhaustion, inadequate journal
coverage, malformed history/moments/RNG, changed masks/interface/model/precision/
optimizer/environment/source identities, mismatched independent receipts and
copied physical journals. These authored failures are not actual host shortages.

## Verification history and reproduction

The initial focused checks passed8 tests, then34 combined tests, then13 expanded
recovery controls. The first retained
[collection](2026-10-05-chosen-sft-accounted-recovery/run-01/verification.json)
passed36 tests and both fresh children at its recorded source hash. Final review
then added the directly imported identity and digest modules to mandatory source
coverage; that earlier receipt remains unchanged and historical. The final
collection above rechecks the expanded coverage. No failed verification attempt
was omitted; the retained negative tests and later training failure are expected
controls, not hidden failed acceptance runs.

The actual environment is Linux/aarch64, Python3.12.14, Torch2.14.1+cpu and
Transformers5.18.0 in the existing isolated teaching environment. Its selected
course lock is hashed, not presented as proof every installed version came from
that lock. Full observed package/lock identity is in the receipt. No package
installation, network/model acquisition, pretrained weights, GPU execution,
shared-environment change, service, Git operation, Mac/hosted run or media render
was performed by this task.

Run the focused controls with the existing CPU environment:

```bash
PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python -m unittest test_chosen_sft_recovery test_chosen_sft_control -v
```

For retained fresh-process evidence, use a new output directory:

```bash
PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python tests/test_chosen_sft_recovery.py --collect-replay outputs/chosen-recovery-check-01
```

Final helper SHA256:
`9b85d65b6ed5829901aeb667bc0a7a6fb3579764eee985f4f18fec83cacfe6ce`.
Test SHA256:
`c2852fc150deca1b63a9931f0522ec0c12fe0fdc1b722090f6ac3c631dced72a`.
