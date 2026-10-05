# Bounded native Base profile

The learner approved this goal-aligned next step on October5 and asked us to
continue toward the original improvement goal without repetitive routine
approval questions. Run the already declared `assistant-profile-base06` only:
Qwen3-0.6B-Base at `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, full SFT,
20 updates, microbatch1, accumulation4, length256, seed1212, learning rate2e-5,
checkpoint interval20,900-second independent ceiling and25GiB host reserve.
No download, pilot extension, data replacement or resumed invocation is implicit
in this first profile. Later agreed stages retain their own recipes and gates.

## Prediction and success criteria

Prediction: the native pinned BF16/SDPA path can complete the fixed schedule
within the existing Spark reserve and deadline. Completion requires actual
exit0, finite training losses/gradients, original data/template/tokenizer identity,
all20 updates, both complete development NLL and fixed-generation panels, and
both committed snapshots plus final export. Any cap, interface, finite-state,
memory, deadline, shutdown or logging failure remains a failed attempt.
Improved development NLL is measured, not assumed. This short profile does not
establish general assistant capability, optimal recipe or the complete goal.

## Complete declared allowances

The [tokenizer measurement](../reports/2026-10-05-base-tokenizer-sizing.json)
supplies the unchanged schedule's14 known SFT16 requirements. Use them exactly;
add conservative caps of3,000,000,000 semantic-validation tensor elements and6
RNG-state checks for two saves on oneCUDAdevice. This gives455 training plus720
development targets,1,175 total. Generation reserves16attempts/1,024new-token
slots and conservative uncached geometry; returned work is separately observed.

Each snapshot has declared payload maximum5GiB, tensor-byte maximum4GiB,
2billion tensor elements,50,000tree nodes and1MiB primitive data. I/O9 permits
two saves, no inspect/load operations; hash/serialization10GiB each, clone8GiB,
4billion element visits,100,000node visits and2MiB primitives. Each independently
owned work/I/O journal has a4MiB bound. These are conservative admitted limits,
not premeasured serialization sizes or physical aggregate quotas.

The actual35,248-byte safetensors header declares310 BF16 tensors totaling
596,049,920unique elements. Tied input/output embeddings mean native
`state_dict()` additionally contains the155,582,464-element output-head alias.
From the exact config/header and installed Adam source, projected initial/final
snapshot tensor bytes are1,503,264,768/3,887,465,688 before RNG, and two-save
semantic visits are2,695,364,918 before RNG. Actual loaded layout and committed
bytes must be observed; these projections are not measurements of fit.

Reserve12GiB free disk for the planned output (two bounded snapshots plus model
export and metadata). This is a planning allowance, not an enforced whole-output
quota. Exported model/tokenizer/log files remain outside I/O9. Reuse existing
platform environment and locks; do not install or alter shared dependencies.

## Supervision and evidence

Use the existing closed native-profile controller, independently advancing
deadline, sampled Linux MemAvailable and sanitized conflict scan. Before launch,
record actual GPU utilization and no conflicting large process; unified GPU
memory fields may be unavailable and are not replaced by invented values.
Preserve raw child logs, both numerical and I/O journals, scientific contract,
checkpoint/interface bytes, command, software/device identity, actual process
exit, sampled memory minima and owned shutdown/log acknowledgment boundaries.
Sampled memory is not continuous enforcement. No cgroup, hard filesystem quota
or hostile-process containment claim follows.

Read outcomes against the frozen original requirements. Update current campaign
evidence with actual measurements only, never backfill unexecuted rows. Retain
the frozen preparation report and all earlier failed attempts. Integrate the
result into Chapter9 and its solutions/lab; do not advance learner mastery.
