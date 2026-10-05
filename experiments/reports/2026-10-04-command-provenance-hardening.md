# An invocation is not Python's rewritten argument list

Independent read-only review found a command-provenance defect in DXI-01:
`[sys.executable, *sys.argv]` loses the `-m` launcher form. Python rewrites
`sys.argv[0]` to the module's file; directly executing that file can break its
relative imports. This is an identity/diagnostic correction, not a new model
experiment or a change to old numeric observations.

The shared [collector](../../src/dongxi_llms/run_identity.py) now records
sanitized original interpreter argv in `command`, separate rewritten
`python_argv`, and `command_origin`. An explicit caller invocation remains
explicit; an unavailable-original-argv fallback is labeled rather than passed
off as the original command. Credentials are redacted in both surfaces.

Two new tests independently cover mocked original/rewritten argument lists
with secret flags, and an actual `python -m timeit` child process collecting its
own identity. The real child exits0, retains `-m timeit` in the command, and
records the separate rewritten `timeit.py` application argument. The full
identity suite passes14tests. No pretrained weights, model hub fetch or GPU
operation is part of that launcher probe.

The [final locked CPU verification](2026-10-04-course-reproduction-verification.json)
binds the corrected collector/test source hashes and retains the actual
291-test command log, plus scoped independent38-test acceptance. The original
[identity report](2026-10-04-run-identity.md) and local model/export reports
retain their own historical source hashes and metadata; they are not silently
rewritten as checks of this later source. Tokenizer semantics, numeric model
state and the original recorded results are unchanged by this correction.

DXI-01 remains partial: approved pretrained/Spark checkpoint-chain validation
and actual interrupted/resumed equivalence still require their own evidence.
The new metadata does not make those unexecuted jobs complete.
