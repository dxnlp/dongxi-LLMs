# Evaluation and SFT CPU verification — 2026-10-04

Scope: newly authored Chapters7–9, Days10–14 notebooks, original metric/data fixtures and bounded CPU SFT. This report contains actual local measurements, not proposed Qwen results.

Base commit before edits: c0c6c9d6988ed167267d423712e0df1c5efee09d. Interpreter: /home/dongxi/dgx-spark-dongxi/.venv/bin/python. Torch2.13.0+cu130 (actual exact string preserved in the adjacent JSON). CPUFP32, one Torch thread. No model download, GPU run or notebook server was launched.

## Measured microscopic comparison

| Fixed100-update recipe | Full SFT | Rank4 Q/V LoRA |
|---|---:|---:|
| Initial answer NLL |3.101167|3.101167|
| Final answer NLL |0.000554532|2.802457|
| Answer-plus-END sequence EM, four seen requests |1.00|0.75|
| Trainable scalar parameters |6,120|384|
| Total stored scalar parameters |6,120|6,504|
| Immediate state forward replay |equal|equal|

Both modes use seed1212, AdamW0.015, weight_decay0, clipping1, vocabulary22/width24/oneblock/threeheads, eight supervised targets per update and800 target presentations. Two measured runs completed in0.913015s after module imports. Their complete100-entry loss and gradient-norm histories and all four ID completions are in [the adjacent JSON](2026-10-04-evaluation-and-sft-course.json).

Interpretation: full tuning learned the four seen requests at this recipe. The rank-constrained update reduced NLL and generated three exact sequences. This is a mechanism demonstration, not evidence of a universal method ranking, unseen transfer or a real base-to-assistant transition.

## Correctness checks

14 targeted unit tests passed in0.780s after the visual-review fixture correction:

- pass@k product versus exhaustive small combinatorial cases;
- interval edge cases and fixed exact-match normalization;
- paired scores and planted cross-split leakage;
- aggregate improvement with a genuine rare-slice regression;
- role/label alignment, END preservation, isolated visibility and mixtures;
- zero direct ignored-logit gradients;
- joint versus token-weighted accumulated gradients;
- initial LoRA equivalence, first-step A/B gradients and merged-matrix equality;
- bounded loss improvement and immediate forward replay;
- optional runner multi-turn spans, prefix incompatibility/length rejection;
- original English source-group split separation.

Command:

~~~bash
PYTHONPATH=src OMP_NUM_THREADS=1 /home/dongxi/dgx-spark-dongxi/.venv/bin/python \
  -m unittest discover -s tests -p 'test_*sft*.py' -v
~~~

The explicit Jinja template was compiled with installed Transformers' template utility. Its generated assistant prefix matched the completed conversation prefix exactly. This verifies the authored template structure without loading Qwen; the runner repeats the check on real token IDs before training.

The original English fixture generator was executed into a new temporary directory. It produced240/60/120 examples in80/20/40 disjoint value groups. [The saved data card](../data/instruction-interface-v1-data-card.json) preserves each exact JSONL hash; the generated data remain temporary until a named training run prepares its own output directory.

A local math-source check passed after the new chapters and solutions were added. Its repository-wide counts change as other course chapters are authored, so the final integrated report owns those counts.

Visual review exposed an incorrectly constructed evaluation fixture: the first version improved both common and rare slices despite its intended hidden-regression lesson. The corrected deterministic fixture gives A common20/30, rare5/10, total25/40; B common26/30, rare2/10, total28/40. Overall B improves from0.625 to0.700 while rare-format accuracy drops from0.50 to0.20. Its paired difference is0.075 with percentile interval[-0.125,0.275]. The adjacent JSON retains the original fixture's historical difference and records the correction. The SFT measurements use a separate symbolic dataset and are unchanged.

## Sources at this verification boundary

| Path | SHA-256 |
|---|---|
|src/dongxi_llms/evaluation_lab.py|75ffea74bc308180f94a180a6e90afd3bfe69095c20b64fd8ed002a6f137dddd|
|src/dongxi_llms/instruction_data_lab.py|7a1b4d174c61f0cba8377b8e5f2f5708b04b519226194d8c0c90b4ee78b4e4ed|
|src/dongxi_llms/sft_lab.py|0016e5795828b386e6e86571505545b68c46095d8d8e001916f1e3d778ae372e|
|tests/test_evaluation_sft_labs.py|9fa0083df39e030ea0a53bb81f88dbed8f19a1d08a196be623b14158ab78c486|
|tests/test_spark_sft_contract.py|10408724bdcd4ae118e380b0c5ca58d536ea6325372a2ef0dcf943ecaf4508f5|
|scripts/run_chapter09_spark_sft.py|a9b85b8cd03a43561f9990eb68cdb35879bac5e170d86317f344ff8a0a3c31f3|
|scripts/prepare_chapter09_instruction_fixture.py|02b2096b2ecbd59011df3da5fe16d0c6c8bfc7051df37a87aa191e9b38494674|
|experiments/data/instruction_interface_v1.jinja|f26fd6284d6bc05d057a4b5800e2a268a85df6546edf484d979e23a51ed7886d|

## Notebook and experiment boundaries

Ten visual notebooks cover all Days10–14. The course verifier executes each source in a fresh CPU/offline kernel, exports data-derived reference PNGs and records source hashes; its final integrated manifest/report supplies execution evidence. An initial Matplotlib failure revealed a tiny negative upper error length from floating-point roundoff at all-success probability; the plot now clamps error lengths at zero. The underlying Wilson calculation was preserved.

The real-model runner is implemented and CPU contract-tested, with exact base snapshots, full/adapter checkpoint exports, runtime/memory bounds and raw baseline/candidate grid outputs. It has not been GPU-integrated or profiled. No real Qwen improvement, Spark throughput, memory peak or publication-suite result is claimed.

Primary references: [code-evaluation pass@k](https://arxiv.org/abs/2107.03374), [LoRA](https://arxiv.org/abs/2106.09685), [Transformers templates](https://huggingface.co/docs/transformers/chat_templating), [TRL SFT documentation](https://huggingface.co/docs/trl/sft_trainer), and the official pinned Qwen base cards linked in the lab.
