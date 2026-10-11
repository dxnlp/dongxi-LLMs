"""Regenerate original CS33602/03 notebook sources; does not execute them."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[5]
DEST = ROOT / 'notebooks/extensions/cs336'
DEST.mkdir(parents=True, exist_ok=True)


def make(name, title, intro, setup, sections):
    cells = []
    def md(text, ident):
        cell = nbf.v4.new_markdown_cell(text); cell['id'] = ident; cells.append(cell)
    def code(text, ident):
        cell = nbf.v4.new_code_cell(text); cell['id'] = ident; cells.append(cell)
    md('# '+title+'\n\n'+intro, name+'-intro')
    code(setup, name+'-setup')
    for number, (prompt, reference, explanation) in enumerate(sections, 1):
        md(prompt, f'{name}-prompt-{number}')
        md('### Reference solution\n\nRun after writing the prediction. This cell executes the named reference and controlled variant.', f'{name}-reference-{number}')
        code(reference, f'{name}-code-{number}')
        md('### Reference explanation\n\n'+explanation, f'{name}-explanation-{number}')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name':'python3','display_name':'Python 3','language':'python'},
        'language_info': {'name':'python','version':'3.12'}})
    nbf.write(notebook, DEST/(name+'.ipynb'))

setup = '''from pathlib import Path
import sys
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(root / "src"))
import math, json
import matplotlib.pyplot as plt
from dongxi_llms.tokenizer_contracts import (
    TokenizerContracts, ByteScore, ByteScoringContract, aggregate_byte_scores)
'''

make('05_tokenizer_contracts', 'One text, several tokenizer contracts',
'''A tokenizer can preserve the decoded text while changing the model's IDs and cost. Which boundaries must be frozen before a token count or likelihood comparison means anything?

This original CPU microscope wraps the existing `ByteBPE`. It is not a GPT-2/Qwen implementation. The BPE algorithm is reused; segmentation, control admission and record framing are the changed mechanisms. No model download is needed. Predicted results are followed immediately by executable references and explanations. Companion reasoning: [worked notes](05_tokenizer_contracts-solutions.md).''', setup, [
('''## Prediction — which boundaries survive a merge?

Train on repeated `ab cd`. Predict whether whole-record merges can produce one token spanning the space, and whether a word/space segmentation permits that same token.

## Checkpoint — isolate one policy

Write the predicted token counts before execution. Keep text and merge budget fixed; change only segmentation. Decode both results and explain why identical text does not imply identical ID streams.''',
'''whole = TokenizerContracts.train(["ab cd"] * 10, 12)
segmented = TokenizerContracts.train(["ab cd"] * 10, 12, segmentation="words")
variants = {"whole record": whole, "words/spaces": segmented}
for label, tok in variants.items():
    ids = tok.encode("ab cd")
    print(label, ids, repr(tok.decode(ids)), tok.identity[:12])
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(list(variants), [len(t.encode("ab cd")) for t in variants.values()])
ax.set(ylabel="tokens for the same five-character text", title="Measured teaching-tokenizer outputs")
plt.show()''',
'''The whole-record reference produces one token; segmentation produces three. This is a boundary intervention, not a claim that fewer tokens improve language understanding. Read the bar height as sequence positions; the input text is the same. The configured `words` regex splits words, whitespace and punctuation and is explicitly independent of production tokenizer rules. The displayed cell regenerates the figure.'''),
('''## Prediction — can a control marker become ordinary merge data?

Two configured markers overlap: `<STOP>` and `<STOP><STOP>`. Predict which ID the doubled marker receives under longest-match recognition.

## Checkpoint — admission and isolation

Compare recognized controls with rejected reserved text. State whether a user literally typing a control string should be recognized, rejected or treated as ordinary text; that choice belongs to the input contract.''',
'''control = TokenizerContracts.train(["ab<STOP>cd"] * 6, 12,
    special_tokens=("<STOP>", "<STOP><STOP>"))
text = "ab<STOP><STOP>cd<STOP>"
ids = control.encode(text)
print("IDs:", ids, "special IDs:", control.special_ids)
print("Decoded:", control.decode(ids))
assert ids.count(control.special_ids["<STOP><STOP>"]) == 1
assert not any(b"STOP" in p for p in control.model.pieces)
reject = TokenizerContracts(control.model, special_tokens=("<STOP>",), special_policy="reject")
try:
    reject.encode("a user literally typed <STOP>")
except ValueError as error:
    print("Declared rejected-input variant:", error)''',
'''Special spans are separated before BPE training and encoding. The doubled marker wins by the declared longest-match policy, and control IDs cannot be produced by merging ordinary segments. Recognition is not always the appropriate user-input policy: the rejected variant enforces a different trust boundary. No Qwen marker identity is inferred from these original strings.'''),
('''## Prediction — do chunks mean separate records?

A trained token represents `garden`. Predict the ID streams for `['gar', 'den']` when they are independent records versus chunks of one record.

## Checkpoint — a runnable broken variant

Compare both streams with whole-text encoding. Explain why decoding alone misses the broken equivalence check. Then deliberately exceed the chunk buffer limit.''',
'''framed = TokenizerContracts.train(["garden"] * 6, 10)
whole_ids = framed.encode("garden")
record_ids = list(framed.encode_iterable(["gar", "den"], framing="records"))
chunk_ids = list(framed.encode_iterable(["gar", "den"], framing="chunks"))
print("whole", whole_ids, "records", record_ids, "chunks", chunk_ids)
assert chunk_ids == whole_ids and record_ids != whole_ids
assert framed.decode(record_ids) == "garden"
try:
    list(framed.encode_iterable(["gar", "den"], framing="chunks", max_record_chars=5))
except ValueError as error:
    print("Bounded record guard:", error)
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(["whole", "independent records", "one chunked record"], [len(whole_ids), len(record_ids), len(chunk_ids)])
ax.set(ylabel="emitted IDs", title="Same decoded text; different framing contract")
plt.show()''',
'''Independent encoding breaks a cross-boundary merge but still decodes to the same text. The record-framed iterator yields IDs lazily per record and inserts no EOS. Chunk mode buffers the entire single record before encoding, so its memory is proportional to that record; it does not promise constant memory for an arbitrary stream. The positive size guard fails before buffering beyond its declared character budget. The plot compares actual ID counts.'''),
('''## Prediction — what does a round trip preserve?

Predict whether NFC-normalizing `cafe` plus a combining acute preserves the original code-point sequence. Also predict whether a lone UTF-8 byte can be decoded strictly.

## Checkpoint — bind the saved rules

Serialize/reload the tokenizer, then corrupt a policy field. Distinguish preserved normalized text, preserved IDs, and rejected serialized identity.''',
'''normalized = TokenizerContracts.train(["café"], 6, normalization="NFC")
reloaded = TokenizerContracts.from_json(normalized.to_json())
source = "cafe\\u0301"
print("source", repr(source), "decoded", repr(reloaded.decode(reloaded.encode(source))))
assert reloaded.identity == normalized.identity
broken = json.loads(normalized.to_json())
broken["tokenizer"]["normalization"] = "none"
try:
    TokenizerContracts.from_json(json.dumps(broken))
except ValueError as error:
    print("Corruption rejected:", error)
try:
    normalized.decode([230])
except UnicodeDecodeError:
    print("A partial UTF-8 sequence fails strict decoding")
replacement = TokenizerContracts(normalized.model, decode_errors="replace")
print("Explicit replacement policy:", repr(replacement.decode([230])))''',
'''NFC preserves normalized text, not the original decomposed source. Serialization binds both learned pieces and policies; the corrupted field fails the saved content identity. A valid token ID can represent only part of a UTF-8 character, so strict decoding and replacement decoding make different claims. Neither mode invents an original character from malformed bytes.'''),
('''## Prediction — can perplexity change while text probability stays fixed?

One tokenizer assigns probability 0.25 to a one-token text; another assigns 0.5 to each of two tokens for the same text. Predict perplexities and bits per raw byte.

## Checkpoint — aggregate the corpus

Use `a` and a three-byte character as separate scored texts. Compare total-NLL/total-bytes with the misleading mean of document rates. Then change only the EOS convention and require rejection.''',
'''one = -math.log(.25)
two = -math.log(.5) - math.log(.5)
print("token perplexities", math.exp(one), math.exp(two/2))
print("same text bits/byte", aggregate_byte_scores([ByteScore(one, "a")])["bits_per_byte"],
      aggregate_byte_scores([ByteScore(two, "a")])["bits_per_byte"])
records = [ByteScore(math.log(4), "a"), ByteScore(math.log(16), "數")]
corpus = aggregate_byte_scores(records)
wrong_mean = (2 + 4/3)/2
print("corpus", corpus, "wrong document mean", wrong_mean)
try:
    aggregate_byte_scores([records[0], ByteScore(1., "b", ByteScoringContract(eos="included"))])
except ValueError as error:
    print("Incompatible score contract:", error)
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(["corpus sum / bytes", "mean document rates"], [corpus["bits_per_byte"], wrong_mean])
ax.set(ylabel="bits per scored UTF-8 byte", title="Unequal document lengths expose wrong aggregation")
plt.show()''',
'''The same text probability gives perplexities 4 and 2 because token counts differ. Its bits per byte remains 2 under the fixed raw-text contract. The corpus example totals six bits over four bytes: 1.5 bits/byte, rather than averaging document rates. `scored_text` must be the exact raw span corresponding to the summed NLL. Normalization, context, masks and EOS conventions must agree; these toy probabilities do not establish equivalent learned models. Read the plot's height as an aggregated rate, not a capability score.''')])

corpus_setup = '''from pathlib import Path
import sys
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(root / "src"))
import copy, json
import numpy as np
import matplotlib.pyplot as plt
from dongxi_llms.corpus_curation_lab import (
    load_fixture, extract_text, matching_text, prepare_corpus, duplicate_audit,
    split_audit, source_exposure, learning_reference, shingles, jaccard)
fixture = load_fixture()
train = [r for r in fixture if r["split"] == "train"]
'''
make('06_corpus_curation', 'A cleaned corpus can teach the wrong lesson',
'''Which useful examples disappear when a corpus looks cleaner, and how would we discover that loss before training a larger model?

This original authored HTML fixture contains eleven training records and four frozen held-out records. Extraction, matching normalization and training text remain separate. The rules are fixed demonstrations, not trained quality classifiers. The reference uses the existing 258-ID byte decoder and `TrainingSession`: three seeds, three arms, 24 updates per arm, 32 valid targets per update. All negative outcomes are retained. No public crawl, pretrained model, external classifier or GPU is used. [Worked notes](06_corpus_curation-solutions.md).''', corpus_setup, [
('''## Prediction — is extracted text the same as matching text?

Predict what happens to navigation, scripts, accented words, code and visible contact text. Does removing accents for duplicate matching justify removing them from model inputs?

## Checkpoint — trace an actual record

Inspect raw HTML, extracted text and matching text separately. Find an empty extraction. Keep its ID and failure reason in the ledger rather than silently losing it.''',
'''html = '<head><title>hidden</title></head><nav>links</nav><p>Café &amp; code</p><script>noise</script><pre>x &lt; 3</pre>'
visible = extract_text(html)
print("raw HTML:", html, "\\ntraining text:", visible, "\\nmatching text:", matching_text(visible))
raw = prepare_corpus(train, "raw")
for row in raw["ledger"]:
    if not row["accepted"] or row["slice"] in ("multilingual", "contact"):
        print(row["id"], row["slice"], repr(row["training_text"]), row["reasons"])
assert "Café" in visible and "cafe" in matching_text(visible)''',
'''The standard-library HTML parser removes explicitly named containers; this simple policy is not a production main-content extractor. Matching uses casefold/NFD/punctuation removal while training retains visible accents and punctuation. Contact strings are fictitious; retained contact text shows this pipeline is not a privacy sanitizer. Empty extraction `t11` remains identifiable with its reason code.'''),
('''## Prediction — which valuable text will a shortness rule remove?

Predict raw, exact-deduplicated and curated retained counts. Identify useful short/code examples likely to fail the eight-word rule.

## Checkpoint — retain the adverse cases

Print removals and compare retention across authored slices. Change only `min_words` from 8 to 3; report which removals persist. This is a rule intervention, not a trained quality estimate.''',
'''arms = {mode: prepare_corpus(train, mode) for mode in ("raw", "exact", "curated")}
for mode, result in arms.items():
    print(mode, result["counts"], "useful removals:", result["false_removals"])
    assert result["counts"]["input"] == result["counts"]["retained"] + result["counts"]["removed"]
for row in arms["curated"]["ledger"]:
    if not row["accepted"]: print(row["id"], row["slice"], row["reasons"], repr(row["training_text"]))
relaxed = prepare_corpus(train, "curated", min_words=3)
print("Relaxed rule:", relaxed["counts"], relaxed["false_removals"])
fig, axes = plt.subplots(1, 2, figsize=(11, 3))
counts = [len(train), *[r["counts"]["retained"] for r in arms.values()]]
drops = [counts[i] - counts[i+1] for i in range(3)]
stages = ["input", "empty", "exact copies", "rules + near", "retained"]
axes[0].bar(stages, [counts[0], *drops, counts[-1]],
            bottom=[0, *counts[1:], 0], color=["steelblue", "salmon", "salmon", "salmon", "steelblue"])
for index, label in enumerate([str(counts[0]), *["-" + str(d) for d in drops], str(counts[-1])]):
    axes[0].text(index, [counts[0], *counts[:-1], counts[-1]][index]+.15, label, ha="center")
axes[0].tick_params(axis="x", rotation=25)
axes[0].set(ylabel="training documents remaining", title="Retention waterfall", ylim=(0, 12.5))
slices = sorted({r["slice"] for r in train})
rates = [sum(r["accepted"] for r in arms["curated"]["ledger"] if r["slice"] == s)/sum(r["slice"] == s for r in train) for s in slices]
axes[1].bar(slices, rates); axes[1].tick_params(axis="x", rotation=35)
axes[1].set(ylabel="curated retained fraction", ylim=(0, 1.1), title="Authored content slices")
plt.tight_layout(); plt.show()''',
'''The pipeline retains 10 raw, 9 exact and 5 curated records. Six removals reconcile with eleven inputs. Useful short text and code are rejected by the fixed rule; the altered garden copy is removed by near-duplicate grouping. These authored usefulness labels expose a tradeoff and are not a general human quality benchmark. In the waterfall, floating red bars show removals from the remaining count; the right plot shows within-slice fractions. A lower threshold changes the eligible distribution; it does not by itself demonstrate better student behavior.'''),
('''## Prediction — will exact hashing catch an altered garden story?

Predict whether one changed adjective survives exact matching. Compare a permissive LSH configuration with one band containing the whole signature.

## Checkpoint — candidates are not verified duplicates

Use the all-pairs exact oracle on this small fixture. Print missed pairs and false candidates. Then retrieve two original short texts sharing just one trigram and verify why their collision must not become an edge.''',
'''texts = {r["id"]: extract_text(r["html"]) for r in train}
normal = duplicate_audit(texts)
sparse = duplicate_audit(texts, bands=1)
broad = duplicate_audit(texts, bands=64)
for label, audit in [("default", normal), ("one band", sparse), ("many bands", broad)]:
    print(label, "recall", audit["candidate_recall"], "precision", audit["candidate_precision"],
          "missed", audit["missed_edges"], "false candidates", audit["false_candidates"])
collision = duplicate_audit({"F":"red blue green white black", "G":"red blue green orange purple"}, bands=64)
print("Controlled false candidate:", collision["candidates"], collision["similarities"],
      "verified", collision["verified_edges"], "precision", collision["candidate_precision"])
assert collision["false_candidates"] == [("F", "G")] and not collision["verified_edges"]
names = sorted(texts)
matrix = np.diag([jaccard(shingles(texts[name]), shingles(texts[name])) for name in names])
for row in normal["similarities"]:
    a, b = [names.index(i) for i in row["pair"]]; matrix[a,b] = matrix[b,a] = row["jaccard"]
fig, ax = plt.subplots(figsize=(6, 5))
image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(len(names)), names, rotation=90); ax.set_yticks(range(len(names)), names)
ax.set(title="Exact word-trigram Jaccard, after matching normalization")
fig.colorbar(image, ax=ax, label="intersection / union"); plt.tight_layout(); plt.show()''',
'''The altered copy `t03` is close to `t01/t02` yet not normalized-exact. With one long band, a signature must match in all positions to become a candidate; that loses the altered copy even though the exact oracle detects it. The F/G control shares one of five union trigrams: Jaccard 0.2. One-hash bands retrieve it, but the 0.7 verification rule rejects it; its candidate precision is zero. No F/G text enters the training fixture. Empty texts are excluded from retrieval, and short nonempty texts have a defined fallback shingle. The heatmap reports actual normalized-trigram overlap; it is not semantic similarity.'''),
('''## Prediction — must every member of a duplicate group be pairwise similar?

Sets A and C share three of five union words. B contains all five. Predict which edges exceed 0.75 and whether all three are grouped.

## Checkpoint — split by ancestry and verified components

Put C in a held-out split to make the leak visible. Restore all three to training and verify the fixture's actual independent held-out sources. Explain why a component is a policy, not a transitive similarity theorem.''',
'''chain = {"A": "red blue green white", "B": "red blue green white black", "C": "red blue green black"}
chain_audit = duplicate_audit(chain, n=1, num_hashes=16, bands=16, threshold=.75)
print("verified edges", chain_audit["verified_edges"], "components", chain_audit["components"])
records = [{"id":k, "source_group":k, "split":"test" if k == "C" else "train"} for k in chain]
print("Planted cross-split issue:", split_audit(records, chain_audit["components"]))
for row in records: row["split"] = "train"
assert not split_audit(records, chain_audit["components"])
fixture_audit = duplicate_audit({r["id"]: extract_text(r["html"]) for r in fixture})
assert not split_audit(fixture, fixture_audit["components"])
positions = {"A": (0, 0), "B": (1, 1), "C": (2, 0)}
fig, ax = plt.subplots(figsize=(6, 3))
for a,b in chain_audit["verified_edges"]:
    ax.plot([positions[a][0], positions[b][0]], [positions[a][1], positions[b][1]], color="steelblue")
for name,(x,y) in positions.items():
    ax.scatter(x,y,s=300); ax.text(x,y+.12,name,ha="center")
ax.set(xlabel="schematic document position (no metric units)", ylabel="schematic position", title="Verified edges AB and BC; no AC edge")
ax.set_ylim(-.1, 1.3)
plt.show()''',
'''AB and BC each have Jaccard 0.8, while AC has 0.6. Grouping the connected component protects against splitting a source family through an intermediate copy; it does not assert AC passed the threshold. The plot is a graph schematic with verified measured edge values, not a distance embedding. Declared source genealogy is also audited because lexical matching cannot detect every shared origin.'''),
('''## Prediction — equal shares or equal distinct coverage?

Two sources receive half of 100 supervised-token presentations. One has ten available tokens and one has a thousand. Predict repetitions and whether a two-epoch cap is feasible.

## Checkpoint — bind the units

Show objective shares beside source reuse. Change only rare-source capacity. State why a smaller retained pool can change repetitions even when training presents the same number of targets.''',
'''exposure = source_exposure({"rare":.5, "large":.5}, {"rare":10, "large":1000}, 100, epoch_cap=2)
print(exposure)
changed = source_exposure({"rare":.5, "large":.5}, {"rare":50, "large":1000}, 100, epoch_cap=2)
print("Capacity intervention:", changed)
fig, axes = plt.subplots(1,2,figsize=(9,3))
axes[0].bar(list(exposure["sources"]), [r["share"] for r in exposure["sources"].values()]); axes[0].set(ylabel="supervised-token objective share", ylim=(0,1))
axes[1].bar(list(exposure["sources"]), [r["effective_epochs"] for r in exposure["sources"].values()]); axes[1].axhline(2,color="red",linestyle="--",label="declared two-epoch cap")
axes[1].set(ylabel="presentations / available source tokens"); axes[1].legend(); plt.tight_layout(); plt.show()''',
'''The scarce source repeats five times, versus 0.05 effective epochs for the large source; the cap is infeasible. The correct cap is presentations ≤ cap × capacity in the same token units. Increasing capacity alone changes repetitions while objective shares remain 0.5/0.5. This extends the existing mixture lesson, which already distinguishes example and token shares.'''),
('''## Prediction — will curated data beat raw data on every slice?

Write a prediction before running all nine bounded fits. Freeze three seeds, the final-checkpoint selection rule, the held-out texts and filter rules. Do not tune rules against the held-out outputs.

## Checkpoint — compare matched updates and targets

Reuse `TrainingSession` for 24 updates per arm, two full 16-target windows per update. Check identical parent hashes within each seed, 768 target presentations per arm and retained omitted-tail counts. Show all seeds, held-out slices and literal capped generations, even if cleaning harms results.''',
'''result = learning_reference()
for row in result["arms"]:
    print(row["seed"], row["arm"], "targets", row["target_presentations"],
          "NLL", round(row["held_out_nll_after"], 6), "tail targets omitted", row["omitted_tail_targets"],
          "pool epochs", round(row["effective_pool_epochs"], 3),
          "generation", repr(row["generation"]["text"]), "EOS", row["generation"]["natural_eos"])
    print("  slices", row["held_out_slice_nll"])
assert {r["target_presentations"] for r in result["arms"]} == {768}
for seed in result["seeds"]:
    assert len({r["initial_parameters_sha256"] for r in result["arms"] if r["seed"] == seed}) == 1
fig, axes = plt.subplots(1,2,figsize=(11,4))
colors = {"raw":"steelblue", "exact":"darkorange", "curated":"seagreen"}
for row in result["arms"]:
    axes[0].plot([h["tokens"] for h in row["history"]], [h["loss"] for h in row["history"]], color=colors[row["arm"]], alpha=.5, label=row["arm"] if row["seed"] == result["seeds"][0] else None)
    axes[1].scatter(row["arm"], row["held_out_nll_after"], color=colors[row["arm"]])
axes[0].set(xlabel="valid-target presentations",ylabel="training-batch NLL (nats/target)",title="All three seeds; changing batch populations"); axes[0].legend()
axes[1].set(ylabel="fixed held-out NLL (nats/target)",title="Final checkpoints; all seeds shown")
plt.tight_layout(); plt.show()''',
'''Every arm sees 768 targets through 24 identical update geometries and schedules, but each training pool contains different documents and repeated exposure. Complete-only windows omit counted tails, including short-document tails; that limitation belongs to the comparison. The observed fixture does not show a universal curated-data benefit, and its greedy continuations can collapse to spaces or repeated letters without EOS. Training NLL curves grade changing batches and should not be used alone to rank arms. The right panel uses common frozen held-out texts; three seeds are too small for a broad method ranking. This cell regenerates actual bounded CPU learning evidence, not model-scale story quality.''')])
