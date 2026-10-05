# SFT parsed work limit bytes stay fixed before model preparation

This narrow premeasurement addendum preserves the original SFT recipe and prior
development controls. The initial bounded cap-file read already supplies an
exact byte SHA to the scientific configuration, and initial identity hashing
must match it. Add explicit bounded nonblocking no-follow rechecks immediately
before tokenizer and model preparation rather than relying only on a later
provenance hash to describe the earlier parsed limits.

A CPU-only negative control mocks hardware readiness, mutates the limits file
after the first actual identity observation, and must reject before any tokenizer
or model from_pretrained call. Retain the initial/changed hashes, raw exception
and no-loader call counts. No GPU operation, pretrained bytes or acquisition is
used; positive hardware mocks are not platform evidence. Include this control
in a new exclusive reference and retain earlier raw/development outcomes.

This addresses cooperative input mutation at declared preparation boundaries.
It is not a hostile check/syscall-race guarantee across every filesystem access,
nor a new training, tokenization, masking or budget recipe. The existing SFT
cumulative-work protocol and its remaining empirical gates are unchanged.
