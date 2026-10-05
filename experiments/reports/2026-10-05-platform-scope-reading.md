# Platform evidence separation is not a second-machine launch gate

Independent read-only review compared DXI-17 with its preserved
[original acceptance snapshot](2026-10-05-native-base-profile/original-acceptance-before-integration.json)
and [pre-execution reproduction specification](../specs/2026-10-04-course-reproduction.md).
The original criterion3 requires separate Spark/Linux and Mac execution evidence
and prohibits presenting portable inference as a passed Mac run. It does not
explicitly require an actual Mac execution or successful Mac result. Criterion1
requires a clean supported environment, not every listed platform. Criterion5
separates CI source/local commands from hosted execution without making hosted
success compulsory.

The frozen reproduction specification explicitly distinguishes Linux actual,
Mac verification pending and CI source readiness as separate rows. The later
interpretation that criterion3 necessarily requires a Mac receipt was stronger
than the unchanged criterion. The learner challenged that interpretation before
instructing continuation and completion.

Preserve the stronger historical pending requests for an actual supported Mac
run and hosted CI. Route them as parallel portability/publication evidence
tasks, not newly added mandatory package-success gates. Original acceptance
text and dependency arrays remain unchanged. DXI-17 still needs current-source
Linux verification and its other original checks; DXI-03/18 still need actual
controlled comparisons and integrated review.

| Platform/workflow | Actual present evidence | Unobserved boundary |
|---|---|---|
| Spark/Linux ARM64 | Isolated locked CPU environment, tests/fresh kernels and source checks at retained revisions | Final current-source integration rerun is separate |
| Mac | Portable route, lock/source checks and verification packet | Not executed/unverified from this task |
| Hosted CI | Workflow source and locally exercised commands | Not executed here; publication authority separate |

Mac must not be marked passed or technically unsupported merely because this
task lacks a Mac command endpoint. Neither follows from Linux evidence. No Mac
SSH host is configured in this task's local SSH configuration; app project
registration is not an executable endpoint or receipt. The
[Mac handoff packet](../../docs/handoffs/MAC_CPU_VERIFICATION.md) remains available.

This independently reviewed original-text clarification is not a waiver to
reach18/18. Actual GPU runs, failures, recovery/interface gates and work/runtime
caps remain mandatory for their claimed scope. No package or goal is marked
complete by this report alone.
