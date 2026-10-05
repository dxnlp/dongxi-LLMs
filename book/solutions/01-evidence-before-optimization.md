# Solutions — Chapter 1: Evidence Before Optimization

These are model answers, not scripts to memorize. A strong answer states the
evidence boundary explicitly.

## Exercise 1

**a. Observation.** The recorded third loss, 1.8, is numerically greater than the
recorded second loss, 1.7.

**b. Supported interpretation.** The logs and saved checkpoint support the
conclusion that the requested training path completed, assuming the trainer's step
semantics and output path were inspected. “Optimizer steps executed” is an
interpretation of the system evidence rather than a raw scalar.

**c. Unsupported claim.** No general-assistant evaluation or baseline comparison
was provided.

**d. Unsupported claim.** Three non-monotonic points are insufficient to establish
a convergence trend. Even a monotonic three-point sequence would be weak evidence
for long-run convergence.

## Exercise 2

Investigate high-impact differences before attributing the result to low-level GPU
nondeterminism. Suitable answers include:

1. dataset revision, split, filtering, and order;
2. batch size, sequence length, accumulation, and preprocessing configuration;
3. code and trainer commits, including local patches;
4. environment lock and installed framework versions;
5. exact model and tokenizer revisions;
6. precision and attention backend.

Hardware, driver, and kernel behavior still matter, but input or configuration
differences often explain larger discrepancies.

## Exercise 3

One acceptable hypothesis is:

> The selected model will complete three BF16 optimizer steps at microbatch size 1
> and sequence length 1024. Every reported loss will remain finite, the expected
> checkpoint files will be saved, and whole-system available memory will remain
> above 24 GiB. Any failed load, non-finite loss, missing output, nonzero exit, or
> reserve violation will fail the smoke test.

The exact configuration is less important than making every term observable and
falsifiable.

## Exercise 4

The total cannot be calculated as `5.8 + 7.1 GiB`. Cgroup accounting and
CUDA-visible allocation are different, overlapping views. Some CUDA allocations
may be charged to the cgroup, and the accounting mechanisms can include or omit
different categories. Whole-system `MemAvailable` samples provide another view of
pressure, but background activity and reclaimable cache mean that it is not a
perfect per-model allocation meter either.

Report all views separately and describe what each measures.

## Exercise 5

Successful execution and any numerical criteria may be supported, but the
memory-safety clause is falsified. If the hypothesis joins these criteria with
“and,” the overall experiment fails its precommitted success criteria. The report
should preserve both the partial success and the safety failure.

## Exercise 6

One controlled comparison is:

- **Independent variable:** microbatch size, changed from 1 to 2.
- **Dependent variables:** minimum whole-system `MemAvailable` and cgroup memory
  peak. Runtime or throughput can be an additional dependent variable.
- **Controlled variables:** model and revision, dataset examples and order,
  sequence length 1024, gradient accumulation, precision, attention backend,
  optimizer, gradient checkpointing, trainer/environment revisions, and concurrent
  services.

The result describes batch size 2 only. Continue incrementally rather than
extrapolating directly to batch size 8.

## Exercise 7

Loading checks row counts and shapes, not linguistic meanings. The new tokenizer
sends `cat` to a row trained as `dog`, and decodes the original `cat` score as
`dog`. A size check accepts a semantic interface failure. Compare the full
token-to-ID mapping, not only its length.

Identical mappings still do not establish identical encoding. Check the supported
serialized rules, including normalization and merge/pre-tokenization, plus wrapper
settings, special IDs, template and stops. Lowercasing changes how `CAT` is encoded
without changing the fixture's eight entries. Actual save/reload is the positive control.

## Exercise 8

Explicit adoption fingerprints current saved meanings/rules and records a new
declaration for downstream comparison. It cannot authenticate the upstream commit
or reconstruct an unrecorded past tokenizer. Automatic migration would hide that gap.

Timestamp and invocation ID belong in the journal, not stable recovery equality:
both change on restart. Source/data/interface/objective identity belongs in the
stable contract. Passing a fingerprint proves neither language improvement,
a CUDA kernel nor exact model-scale restart parity.

## Exercise 9

The contract requires successful fresh-process replay within the declared
deadline, not only two completed original runs. Safe sampled memory cannot
replace a failed time criterion or missing final comparison. The outer
adapter's exit1 reports its own failure; the native exit remains unknown in
this receipt, with the cleanup error retained. Preserve incurred work and the
uncommitted artifact rather than assigning a convenient native exit or calling
the case successful. A new CPU8 attempt keeps the original deadline and must
produce its own complete receipts before pilot admission.
