# Original teacher data control

This fixture independently authors copy/reverse prompts and a deliberately
fallible programmatic teacher. It is not an API, pretrained model, or human
feedback source. Shared color vocabulary and task forms make held-out prompts a
combination-transfer test, not unseen-word or general instruction following.
The control split adds a polite prefix; its underlying operation/value groups
remain outside training.

All teacher attempts, including a declared error and retry, remain in the
append-only journal. Trace text is retained as provenance but the student learns
only final answers and END. Programmatic traces are not evidence about faithful
language-model reasoning. Acceptance rejects format/length/stopping and within-
prompt duplicates, **not correctness**. Wrong but well-formed answers therefore
remain available to the frozen top/random comparison.

The executable training-task verifier ranks answers; it never reads held-out
reference fields or teacher mode labels. Independent student evaluation uses
frozen held-out references and unconstrained full-vocabulary generation. Global
selection coverage is an audit, not an extra student intervention.

No upstream material is copied. The protocol's content terms preserve the
repository's undeclared redistribution-license boundary rather than asserting
an external dataset or model license.
