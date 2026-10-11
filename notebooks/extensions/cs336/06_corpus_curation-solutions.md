# Corpus curation: worked reasoning and negative evidence

Use [the notebook](06_corpus_curation.ipynb) for predictions, controlled changes
and plot regeneration. The original fixture is
[`corpus.json`](../../../experiments/fixtures/cs336/corpus.json); reusable logic
is in [`corpus_curation_lab.py`](../../../src/dongxi_llms/corpus_curation_lab.py).

1. Extraction removes only explicitly named HTML containers. Matching removes
   accents and punctuation for lexical comparison; training text retains them.
   Empty extraction `t11` remains in the ledger. Fictitious contact text remains
   visible, demonstrating that this pipeline provides no privacy sanitization.
2. Eleven inputs yield ten raw, nine exact and five curated records. The fixed
   eight-word rule removes useful short and code examples. The declared near
   duplicate grouping also removes an altered copy. Changing the threshold
   changes the eligible distribution; authored usefulness labels are a teaching
   rubric, not a general quality benchmark.
3. Exact matching catches the identical garden story but misses its changed
   adjective. Seeded MinHash/LSH retrieves candidates; exact trigram Jaccard
   verifies them. A single long band misses some oracle-positive pairs. More
   bands admit more candidates, which still require verification. The separate
   F/G control shares one of five union trigrams (Jaccard 0.2), is retrieved by
   one-hash bands, and fails the 0.7 verification rule. Its precision is zero;
   neither control text enters the learning fixture. Empty sets
   have no candidates and Jaccard zero under the declared convention.
4. The original A/B/C sets give AB=BC=0.8 and AC=0.6. Connected-component
   grouping joins all three without claiming AC exceeds 0.75. Crossing a split
   through C exposes leakage. Declared source ancestry is checked separately
   because lexical overlap cannot detect every shared origin.
5. Half of 100 token presentations uses a ten-token source five times and a
   thousand-token source 0.05 times. A two-epoch cap fails for the scarce source.
   Increasing its capacity to fifty makes the cap feasible while objective
   shares stay fixed. Shares, capacities and presentations must use the same
   supervised-token unit.
6. The bounded learning reference uses three seeds, three arms and 24 updates
   per arm, with two complete 16-target windows per update. Each arm receives
   768 targets. Initial parameter hashes agree within each seed. The eligible
   pools differ, so repeated exposure differs too; omitted tail targets are
   counted (raw 91, exact 79, curated 45).

Actual final held-out NLL, in nats per supervised byte/EOS target:

| Seed | Raw | Exact | Curated |
|---|---:|---:|---:|
| 1701 | 3.625225 | 3.681660 | 3.712504 |
| 1702 | 3.723926 | 3.631579 | 3.650342 |
| 1703 | 3.613868 | 3.671728 | 3.727350 |

Curated loss exceeds exact loss on every seed. Greedy continuations collapse to
spaces or repeated `e`, reach the 32-token cap, and never emit natural EOS.
Preserve those strings, all slice metrics and all seeds rather than selecting a
favorable arm. The frozen test texts do not choose filters, seeds or checkpoints.
This tiny authored corpus cannot establish general data-quality or story gains.
The [complete reference](../../../experiments/reports/2026-10-11-cs336-upgrade/run-01/data/learning-reference.json)
retains configurations, histories, source/window exposure and literal outputs.
