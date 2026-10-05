# Checkpoint identity and handoff verification

DXI-01's source and bounded CPU mechanisms are verified. Actual pretrained
Spark integration remains pending. The [specification](../specs/2026-10-04-run-identity.md)
preceded testing; [portable JSON](2026-10-04-run-identity.json) records source
hashes, environment, commands and the fresh notebook manifest.

## Implementation and observed checks

The original [identity module](../../src/dongxi_llms/run_identity.py) separates
actual local input bytes, supported tokenizer semantics and declared upstream
metadata. It records Git/dirty/source/input identity, Python/packages/selected
lock, command, device/driver and objective settings. Fast-tokenizer serialization,
vocabulary mapping, wrapper settings, special IDs, template and stops form the
handoff contract. Unsupported slow/custom tokenizers fail closed. Runtime backend
padding/truncation buffers are excluded, but their chosen training policies remain
configuration controls.

Twelve identity tests, eight SFT contract tests and four DPO interface tests pass.
Same-size token swaps, normalization-only changes, special/stop/template/revision
changes and altered fingerprint fields are rejected. Local HF tokenizer
save/reload passes. Explicit legacy adoption rechecks actual current saved
semantics without pretending to authenticate old ancestry. Snapshot path/revision
failures are tested; acquisition is mocked and only exposed under explicit opt-in.
Augmented adapter/tokenizer identities reproduce their canonical hashes.

A real tiny randomly initialized Qwen3-class HF model is saved and reloaded on
CPU with exact logit parity. A nonzero two-rank Q/V LoRA adapter is saved, explicitly
merged by [the local utility](../../src/dongxi_llms/checkpoint_merge.py), exported
and reloaded. Its logits agree at rtol1e-5/atol1e-6; changed recorded base bytes
are rejected before model loading. This uses local original eight-ID fixtures,
not downloaded pretrained Qwen weights or model-quality evaluation.

The actual CPU HF RLVR entry point also completes one bounded update on that
random fixture and carries common identity/interface into its report and export.
Its resource guard is mocked to100GiB for portable contract testing: this value
must not be interpreted as measured host safety. All-zero rewards would still be
a valid mechanism smoke; no learning gain is claimed.

## Runner integration and review repairs

SFT, DPO and RLVR use the shared identity. DPO verifies the actual saved parent's
tokenizer and then the proposed tokenizer before model loading/training. SFT
checks a separately pinned tokenizer against the actual tokenizer paired with
its base snapshot; different revision declarations are permitted only when
supported semantics agree. Full and merged checkpoint genealogies retain the
interface. LoRA directories do not silently substitute for full weights.

Independent review found five handoff/recovery risks during development:

1. A resume could overwrite an existing run's config before compatibility checking.
   Every SFT invocation now requires a new empty output, including resume.
2. The actual resume checkpoint was not input-hashed. It now is.
3. SFT captured base bytes only after loading. It now hashes the local snapshot
   before loading; explicit acquisition has a separate boundary and loading is local.
4. Ctrl-C could leave the invocation journal marked running. Interruption now
   finalizes the last stage before reraising. SIGKILL remains an external concern.
5. Independently pinned SFT tokenizer revisions did not ensure paired-base meaning
   compatibility. Positive equal-semantics and negative permutation/rule tests now do.

Volatile invocation IDs/timestamps/dirty-state metadata do not enter stable SFT
recovery equality. Source/data/interface/lock controls do. The CLI has no implicit
install, model download or service startup. Known credential fields are redacted
without reading credential environment values. A selected lock hash identifies
bytes, not proof of how the current packages were installed.

## Notebook, environment and verification

The new [Day1 lesson](../../notebooks/day-01/04_checkpoint_interface.ipynb)
executes five cells in a fresh CPU kernel and produces one inspected mapping
table. The complete four-notebook Day1 route passes: fourteen code cells, four
figures, zero failures. Source notebooks and learner cells remain untouched by
reference execution. Manifest: `/tmp/dongxi-course-check-s4oigs5n/manifest.json`;
its complete metadata is retained in the adjacent report JSON.

Execution: Linux/aarch64 Spark, Python3.12.14, Torch2.13.0+cu130,
Transformers5.16.1, tokenizers0.23.1, PEFT0.20.0. CUDA was hidden and HF offline
mode enabled for tests. The full repository suite passes212tests in8.616seconds;
book mathematics checks54files/1099expressions with zero issues. An installed
CUDA-capable Torch is not evidence that GPU computation occurred.

After the independently reviewed judge-parser and sampling-numerics hardening,
the integrated suite passes218tests in9.313seconds. This later observation is
stored separately in JSON; the earlier212-test measurement above is retained.
Navigation confirms15chapters/15solution guides/4appendices/65notebooks with
zero issues. Peer original-result ledgers remain historical and unchanged beside
their separate hardening reports.

```bash
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m unittest discover -s tests -q
/home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/verify_course_notebooks.py --days 1 --kernel dgx-spark-native --export-figures
python3 scripts/check_book_math.py
```

Library interfaces were checked against the installed packages and primary
[HF tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer),
[PEFT merging documentation](https://huggingface.co/docs/peft/developer_guides/model_merging)
and [HF cache API](https://huggingface.co/docs/huggingface_hub/package_reference/cache).
Implementation, prose, fixtures and figure are original.

## Remaining evidence

Actual pretrained SFT→full/merged→DPO→RLVR execution, large merge sizing,
GB10 kernel/performance behavior, and interrupted/resumed numerical equivalence
are not established. Neither source math checks nor a local fixture establishes
live GitHub rendering or actual Mac execution. DXI-01 remains in progress until
its separately approved Spark integration check is recorded; the broader goal
remains active. No model acquisition, GPU job, installation, server, publication,
animation production, commit or push occurred in this implementation pass.
