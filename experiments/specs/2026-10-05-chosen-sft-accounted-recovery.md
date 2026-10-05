# Accounted recovery of the matched chosen-only SFT control

Saved before implementation and measurements. This extends the existing chosen
control, not its objective or fixture. Preserve native SFT summed chosen-token
NLL divided by all valid chosen targets in the accumulation window, native DPO's
private CPU Torch replacement draws, complete chosen sequence forwards, AdamW,
the fixed original parent and existing held-out negative results.

## Intended implementation

- Add an opt-in completed-boundary recovery contract to `ChosenSFTLoop`.
- Bind actual encoded pairs/masks, tokenizer interface, selected source/input
  and original-parent file bytes, numerical environment, frozen matching recipe,
  effective zero-dropout model, named optimizer order and exact Adam metadata.
- Reuse the existing nineteen-dimensional logical-work journal and nine-
  dimensional shared snapshot I/O hook, with a distinct chosen-control schema.
  Reference calls and rejected supervision remain zero; full chosen input length
  is counted, not DPO's shifted-forward length. Admit the complete accumulation
  window before the first live draw; retain later failed reservations on recovery.
- Save only completed numerical state. Validate model/Adam history, masks and
  sampler/RNG cursor inside a separately reserved semantic panel before applying
  any state. Load through an independently retained snapshot receipt bound to
  both physical journals and exact payload bytes. A copied journal is not a new
  budget. Snapshot prefix recovery cannot refund later spending.
- Keep legacy unaccounted comparison execution unchanged. This API is not an
  external supervisor, physical quota, model acquisition or new GPU launcher.

## Bounded CPU verification

Use the existing isolated CPU environment, one thread, no network/install/GPU or
pretrained load. Test manual/original objective and draw parity, complete-window
cap refusal before draw/forward, partial failure poisoning, source/data/interface/
optimizer/receipt mismatches and negative payload states before state application.
For both original seeds1818/1819, compare six uninterrupted updates with a real
fresh-process continuation from update2. After saving2, deliberately fail the
second forward of update3, retain its entered/completed costs, reopen the same
journals and independently retained receipt, and resume2→6. Require exact model,
Adam, private/global RNG, numerical history and next-draw parity. Preserve every
failed verification attempt and actual child exit/timeouts. The fresh child has
a60-second ceiling. Re-run the existing matched-control tests without changing
their historical results. Report source hashes and actual CPU observations;
do not claim pretrained/CUDA/BF16, quality gain, host portability or campaign
completion from these checks.
