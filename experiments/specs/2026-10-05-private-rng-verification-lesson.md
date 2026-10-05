# A private sampler replay without changing live randomness

Extend the existing Day25 cache/recovery notebook, preserving all original
cells, learner content and saved previews. Pair Chapter14 questions29–30 with
an original small CPU generator microscope and a data-backed plot. This is
not a substitute for actual SFT/DPO validation-budget controls or pretrained
recovery evidence.

Use seed1818, three completed updates, accumulation2, five possible source IDs
and three future updates. Training history uses six scalar draws. Recheck it
with an independently seeded temporary generator; confirm the live RNG bytes
and next six source IDs remain unchanged. A deliberately broken variant uses
the live sampler for its check, advances its state and reports disagreement
without silently repairing it. Include zero-history/bounded-input tests and
confirm the function does not alter global Torch RNG. Separate private replay
draws from new training examples and count each actual declared draw.

Put the prediction before the runnable reference and explain the controlled
failure next to its plot. Draw values/labels from the same returned records;
label draw positions/source IDs, not FLOPs, token counts or model quality. No
new notebook route or mandatory chapter/day is added. Capture this mechanism
in the existing animation proposal; do not render media.

Run the existing offline isolated CPU environment with hidden CUDA/one numerical
thread. Execute the modified notebook in a fresh temporary-prefix kernel and
new exclusive evidence directory, separately from the unchanged five-notebook
acceptance panel. Retain sources/hashes, actual exits, executed cell/image counts
and generated figure. Do not alter the shared Spark environment, start a server,
load pretrained models, run a GPU job, mutate Git, install or publish.
# Independent accounting addendum

Read-only review after the first passing notebook run found that the returned
draw map did not include the independent expected-future comparison. Preserve
that first run at its original source revision. Add the sixth actual draw bucket
and a total, then spy on actual `torch.randint` calls to verify the complete
microscope count:36 at the frozen default, not30. This does not change any seed,
draw sequence, numerical assertion or production budget. Re-execute the same
notebook in a new exclusive directory. Its six plots remain visually identical;
do not overwrite the first run or the already extracted image.
