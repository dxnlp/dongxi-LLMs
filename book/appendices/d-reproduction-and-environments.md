# Appendix D — Reproduction Commands and Environments

The book has a portable CPU route and a platform-specific Spark route. The
verified environment for the 2026-10-04 material build is recorded in its
verification report. A future installation should capture its own versions;
do not infer Mac binary compatibility from a Linux execution result.

## D.1 CPU learning environment

Use a separate Python 3.12 environment for course teaching. It avoids the Mac's
system Python 3.9 and keeps installations separate from the Spark platform lock.
These are setup instructions, not commands executed by writing the chapter.

```bash
python3.12 -m venv .venv-course
.venv-course/bin/python -m pip install -r notebooks/requirements-course.txt
.venv-course/bin/python -m pip install -e . --no-deps
.venv-course/bin/python -m ipykernel install --user --name dongxi-course --display-name "Python (Dongxi course)"
```

Select a PyTorch wheel for the actual platform using the
[official installation guide](https://pytorch.org/get-started/locally/).
The requirements file states the teaching dependencies; it is not a cross-platform
binary lock. After a successful installation, freeze versions and record the
interpreter identity for that environment. The
[IPython kernel guide](https://ipython.readthedocs.io/en/9.9.0/install/kernel_install.html)
explains why a virtual environment must be registered separately with Jupyter.

Most new lessons need only Torch and Matplotlib; the foundation evidence lab also
has a standard-library-only route. No teaching reference downloads model weights.
Optional pretrained runners have additional Transformers/PEFT requirements and
must use a separately profiled training environment.

## D.2 Execute without a notebook server

From the repository root:

```bash
PYTHONPATH=src .venv-course/bin/python -m unittest discover -s tests
.venv-course/bin/python scripts/check_book_math.py
.venv-course/bin/python scripts/check_course_integrity.py
.venv-course/bin/python scripts/verify_course_notebooks.py --kernel dongxi-course
```

To execute only a focused route, add `--days 10 11`. The verifier starts a fresh
kernel for each source notebook, applies CPU/offline settings, retains executed
copies and source hashes in a new output directory, and reports failed paths.
`--export-figures` additionally regenerates reference PNG files. It preserves
source notebook cells and learner edits. The
[NBClient documentation](https://nbclient.readthedocs.io/en/latest/client.html)
describes the execution, working-directory and timeout interfaces used here.

On Spark, substitute the established interpreter
`/home/dongxi/dgx-spark-dongxi/.venv/bin/python` and kernel `dgx-spark-native`.
Do not copy that virtual environment onto Mac. A successful verifier run is
material readiness evidence; learner practice and model-scale outcomes remain
separate records.

## D.3 Existing measured GPU run

The TinyStories [pipeline guide](../../docs/TINYSTORIES_PIPELINE.md) and
[completed report](../../experiments/reports/2026-09-14-tinystories-learning-result.md)
describe the actual first learning run. It used pinned data/tokenizer revisions
and source hashes, with fresh model weights, a14,000-update horizon and four-hour
cap. Analyze its portable JSON in the Day 9 notebooks without its raw checkpoint.

Reproduction of a model-scale run starts with a resource profile, not an
unqualified command copied into a fresh terminal. Preserve25 GiB host reserve,
inspect concurrent model processes, and use the platform's bounded execution
mechanism. Actual GPU learning outcomes require their own reports.

## D.4 Post-training experiment route

Chapters9,11 and13 supply SFT, DPO and RLVR protocols. Their CPU integrations
test their objectives and graph boundaries. Optional model runners require
explicit model/checkpoint/tokenizer revisions, local data contracts, evaluation
panels and resource bounds. Read their corresponding lab guides before invoking
them. A config for an unexecuted pretrained comparison is not a measured model.

Record the parent checkpoint and objective. Full SFT and LoRA have different
trainable parameter counts; DPO needs a frozen reference; policy optimization
needs a named rollout policy and correct old/current/reference roles. Retain
CPU and CUDA RNG state, optimizer history, data position and schedule where
the declared resume contract requires them. Never trust arbitrary checkpoint
pickle files from an unverified source.

## D.5 Return evidence and machine switches

For every run, return a concise manifest: source identity, execution host,
interpreter/packages, configuration, data/model identities, actual exit status,
runtime, target/update counts, resource measurements and report path. Avoid
environment dumps that include credentials. Commit small evidence and source;
keep weights and raw dataset caches under ignored output paths.

The machine-switch phrases and [handoff](../../docs/handoffs/CURRENT.md) preserve
the learning position. Repository synchronization does not synchronize a live
kernel, running process, credential or local checkpoint directory. A final
release additionally needs a scoped publication decision and its own review.
