# Day 3 Interactive Mathematics

These notebooks turn Chapter 3 mathematics into inspectable code while retaining
the course's mechanism-first discussion style.

## Session sequence

| Session | Deep question | Primary mechanism | Evidence state |
|---|---|---|---|
| `01_logits_softmax_nll.ipynb` | Why do relative scores, rather than absolute logits, determine belief and surprise? | stable softmax, labels, NLL, $p-q$, perplexity | ready for guided use |
| `02_causal_shift_and_masks.ipynb` | How can a low loss certify the wrong task? | target alignment, causal visibility, ignored targets, valid-token mean | planned |
| `03_distribution_learning.ipynb` | How does one-hot supervision across examples become a learned distribution? | gradient accumulation, empirical frequencies, tiny next-token model | planned |

## How to use a session

1. Open the notebook in the repository's validated learning environment.
2. Stop at each **Prediction checkpoint** and write a qualitative prediction.
3. Replace only the marked `...` expressions; do not paste a complete solution.
4. Run the accompanying checks and inspect intermediate values.
5. Discuss surprising behavior before moving to the next section.
6. Finish the evidence-boundary cell in your own words.

Session 1 uses only Python's standard library. Later sessions may use PyTorch for
autograd verification and run with the registered `Python (DGX Spark Native)`
kernel. Its interpreter is
`/home/dongxi/dgx-spark-dongxi/.venv/bin/python`; the reproducible installer is
`/home/dongxi/dgx-spark-dongxi/scripts/setup_jupyter.sh`. The kernel was verified
through an actual notebook execution with PyTorch `2.13.0+cu130`, CUDA 13.0, and
the NVIDIA GB10. Do not install an arbitrary replacement PyTorch build merely to
launch a notebook.

Launch JupyterLab with the course repository as its root:

```bash
cd /home/dongxi/dongxi_ai/Dongxi_LLMs
/home/dongxi/dgx-spark-dongxi/.venv/bin/jupyter lab \
  --no-browser --ip=127.0.0.1 --port=8888
```
