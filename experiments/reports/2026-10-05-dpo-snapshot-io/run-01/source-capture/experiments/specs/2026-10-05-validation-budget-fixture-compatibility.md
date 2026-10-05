# Existing CPU work fixtures with explicit validation allowances

The released838-test revision uses twelve SFT and fourteen DPO budget dimensions.
The next runner schemas add separately declared semantic-validation dimensions.
This protocol precedes changes to the two existing work-test fixture cap maps.
Historical reports and raw cap files keep their original bytes and identities.

Preserve every original training, sampling, target, position, evaluation and
generation cap, recipe, seed, objective, mask, checkpoint mode and assertion.
Add only finite allowances for the new validation dimensions:128 validation
operations,4096 history rows,2000000 tensor elements and1024 RNG states per
fixture campaign; DPO also gets4096 temporary sampler-replay draws. These are
authored tiny-CPU fixture capacities, not production approvals or estimates of
pretrained-model size. They do not enlarge the scientific training budget.

Use explicit v2 contracts. An old budgeted schema or a missing new dimension
must refuse; no automatic migration or budget refill. Old unbudgeted CPU
reference APIs can remain for original-equation comparisons. The existing
work tests must continue to assert all original caps, counts, failure paths,
cache behavior, numerical histories and later-charge retention. New focused
tests separately verify validation refusal before scans/draws/application,
malformed-state charges, actual save/load/restore call counts and fresh replay.

Runner semantic validation is not the shared reader's restricted deserialization,
generic finite/tree checks, byte hashing, serialization or a physical resource
quota. Inspect those remaining paths and keep them pending rather than claiming
that the new dimensions bound all CPU operations or hostile checkpoints.

Run only the existing isolated offline Linux ARM64 CPU environment, hidden CUDA
and one numerical thread. Retain actual exits and exclusive raw evidence; no
install, model/data acquisition, pretrained load, GPU/service/platform/Git write,
hosted CI, media production or publication. Keep13of18 complete, Day9 and all45
external actual-result dictionaries with every field null.

## Compatibility addendum before the corrected regression panel

The first expanded SFT and DPO panels retained failures in older test
expectations: the child-process `ledger-before` observation was taken after
restore, although restore now correctly charges validation. Record that
observation before restore and retain a separate `ledger-after-restore` file.
Keep the original exact before-state comparison and additionally assert that
only the new validation dimensions increased during restore. Do not refund
later failures or change any numerical assertion.

The existing metadata-only CUDA fixture constructs a CPU loop before mocking
its device. Declare its one mocked RNG tensor shape explicitly; this matches
the new construction-time layout observation without relaxing runner checks
or claiming a real GPU test. The existing joint DPO work/artifact fixture
keeps100000 in all original fourteen dimensions and declares2000000 only for
the new tensor-element dimension, with the other four new dimensions100000.
This is a compatibility allowance for tiny validation, not a scientific-cap
increase or a production-model estimate. Retain the first failing raw panels.

Before this addendum's edits, the additional affected source hashes were
`3f49a3c50ff0a39082749f8254e417665b9a5d9760aa6d3fba0cc211a5bf8f2e`
for the SFT recovery fixture and
`fbd56514f10360392a67aabf5789f553320402bdd41febaa09d9b4b33f7eca68`
for the DPO snapshot-artifact fixture. Verify these identities against the
captured shell output; a correction belongs in a new addendum, not a rewrite
of an earlier failed record. Two initially mistyped hashes in this addendum
were corrected against that output before any fixture edit or corrected panel.

The root targeted17-control run then retained four errors because the mocked
shape was injected as a plain tuple rather than the runner's observed
`torch.Size`. Correct only that fixture to `current.shape`, preserving the
runner's `.numel()` cost calculation and all original checks; rerun the same
panel. The separate diagnostic JSON preserves the actual failed exit.
