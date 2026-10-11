"""Original bounded corpus-curation microscope; no web crawl or classifier.

Matching normalization never replaces training text. LSH retrieves candidates;
exact Jaccard verifies edges. Connected components are a declared grouping rule.
The learning reference reuses TrainingSession, its decoder, AdamW and schedule.
"""
from collections import Counter
from html.parser import HTMLParser
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import time
import unicodedata

DEFAULT_FIXTURE = Path(__file__).resolve().parents[2]/"experiments/fixtures/cs336/corpus.json"


class _TextParser(HTMLParser):
    """Keep visible authored content; ignore these explicitly chosen containers."""
    ignored = {"script", "style", "head", "nav", "footer"}
    blocks = {"p", "div", "section", "article", "pre", "li", "h1", "h2", "br"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth, self.parts = 0, []

    def handle_starttag(self, tag, attrs):
        if tag in self.ignored:
            self.depth += 1
        if not self.depth and tag in self.blocks:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.ignored and self.depth:
            self.depth -= 1
        if not self.depth and tag in self.blocks:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.depth:
            self.parts.append(data)


def extract_text(html):
    if not isinstance(html, str):
        raise TypeError("HTML fixture must be text")
    parser = _TextParser(); parser.feed(html); parser.close()
    return "\n".join(" ".join(line.split()) for line in "".join(parser.parts).splitlines() if line.strip())


def matching_text(text):
    """Casefold, NFD, remove marks/punctuation, collapse whitespace FOR MATCHING."""
    text = unicodedata.normalize("NFD", text.casefold())
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text if not unicodedata.combining(c))
    return " ".join(text.split())


def shingles(text, n=3):
    if not isinstance(n, int) or n < 1:
        raise ValueError("Positive integer shingle width required")
    words = matching_text(text).split()
    # Short nonempty texts retain identity rather than all sharing an empty set.
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b)/len(a | b) if a or b else 0.


def minhash_signature(items, num_hashes=64, seed=336):
    if num_hashes < 1 or seed < 0:
        raise ValueError("Positive signature size and nonnegative seed required")
    if not items:
        return None
    encoded = [json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode() for item in sorted(items)]
    return tuple(min(int.from_bytes(hashlib.blake2b(str(seed+i).encode()+b"\0"+item, digest_size=8).digest(), "big") for item in encoded) for i in range(num_hashes))


def lsh_candidates(signatures, bands=16):
    lengths = {len(v) for v in signatures.values() if v is not None}
    if not lengths:
        if bands < 1:
            raise ValueError("Positive number of bands required")
        return set()
    if len(lengths) != 1 or bands < 1 or next(iter(lengths)) % bands:
        raise ValueError("Aligned signature size must be divisible by positive bands")
    width = next(iter(lengths))//bands
    buckets = {}; pairs = set()
    for name, signature in sorted(signatures.items()):
        if signature is None:
            continue
        for band in range(bands):
            key = (band, signature[band*width:(band+1)*width])
            bucket = buckets.setdefault(key, [])
            pairs.update(tuple(sorted((previous, name))) for previous in bucket)
            bucket.append(name)
    return pairs


def duplicate_audit(texts, *, n=3, num_hashes=64, bands=16, threshold=.7, seed=336):
    if not 0 < threshold <= 1:
        raise ValueError("Similarity threshold must be in (0,1]")
    sets = {name: shingles(text, n) for name, text in texts.items()}
    signatures = {name: minhash_signature(items, num_hashes, seed) for name, items in sets.items()}
    candidates = lsh_candidates(signatures, bands)
    scores = {pair: jaccard(sets[pair[0]], sets[pair[1]]) for pair in itertools.combinations(sorted(sets), 2)}
    oracle = {pair for pair, score in scores.items() if score >= threshold}
    edges = {pair for pair in candidates if scores[pair] >= threshold}
    parent = {name: name for name in texts}
    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]; name = parent[name]
        return name
    for a, b in sorted(edges):
        aa, bb = find(a), find(b); parent[max(aa, bb)] = min(aa, bb)
    groups = {}
    for name in sorted(texts):
        groups.setdefault(find(name), []).append(name)
    return {"candidates": sorted(candidates), "verified_edges": sorted(edges),
            "oracle_edges": sorted(oracle), "missed_edges": sorted(oracle-edges),
            "false_candidates": sorted(candidates-oracle), "components": list(groups.values()),
            "candidate_recall": len(candidates & oracle)/len(oracle) if oracle else None,
            "candidate_precision": len(candidates & oracle)/len(candidates) if candidates else None,
            "similarities": [{"pair": list(pair), "jaccard": score} for pair, score in scores.items()],
            "parameters": {"n": n, "num_hashes": num_hashes, "bands": bands, "threshold": threshold, "seed": seed}}


def split_audit(records, components):
    """Audit both declared source ancestry and verified duplicate components."""
    lookup = {r["id"]: r for r in records}
    if len(lookup) != len(records):
        raise ValueError("Unique document IDs required")
    source_groups = {}
    for record in records:
        source_groups.setdefault(record["source_group"], []).append(record["id"])
    issues = []
    for kind, groups in (("source", source_groups.values()), ("verified_duplicate", components)):
        for group in groups:
            if any(name not in lookup for name in group):
                raise ValueError("Unknown component member")
            if len({lookup[name]["split"] for name in group}) > 1:
                issues.append({"kind": kind, "ids": sorted(group), "splits": sorted({lookup[name]["split"] for name in group})})
    return issues


def source_exposure(shares, capacities, presentations, *, unit="supervised tokens", epoch_cap=None):
    if set(shares) != set(capacities) or not shares or not math.isfinite(presentations) or presentations <= 0:
        raise ValueError("Aligned nonempty sources and positive finite presentations required")
    if any(not math.isfinite(v) or v < 0 for v in shares.values()) or not math.isclose(sum(shares.values()), 1.):
        raise ValueError("Finite source shares must sum to one")
    if any(not math.isfinite(v) or v <= 0 for v in capacities.values()) or not unit:
        raise ValueError("Positive capacities measured in the declared unit required")
    if epoch_cap is not None and (not math.isfinite(epoch_cap) or epoch_cap <= 0):
        raise ValueError("Positive finite epoch cap required")
    rows = {s: {"share": shares[s], "capacity": capacities[s], "presentations": shares[s]*presentations,
                "effective_epochs": shares[s]*presentations/capacities[s]} for s in shares}
    return {"unit": unit, "sources": rows, "cap_feasible": epoch_cap is None or all(r["effective_epochs"] <= epoch_cap for r in rows.values())}


def load_fixture(path=DEFAULT_FIXTURE):
    return json.loads(Path(path).read_text())


def prepare_corpus(records, mode="curated", *, min_words=8, max_repetition=.45):
    if mode not in ("raw", "exact", "curated"):
        raise ValueError("Unknown curation arm")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Unique document IDs required")
    ledger, accepted, seen = [], [], set()
    for record in records:
        raw = record["html"]; text = extract_text(raw); match = matching_text(text)
        words = match.split(); counts = Counter(words)
        repetition = max(counts.values())/len(words) if words else 0.
        reasons = []
        if not text:
            reasons.append("no_extracted_content")
        if mode != "raw" and match in seen:
            reasons.append("normalized_exact_duplicate")
        if mode == "curated" and len(words) < min_words:
            reasons.append("short_rule")
        if mode == "curated" and repetition > max_repetition:
            reasons.append("repetition_rule")
        if not reasons:
            accepted.append({**record, "training_text": text, "matching_text": match})
            seen.add(match)
        ledger.append({"id": record["id"], "source_group": record["source_group"], "split": record["split"],
                       "slice": record["slice"], "raw_html": raw, "extracted_text": text, "matching_text": match,
                       "training_text": text, "accepted": not reasons, "reasons": reasons,
                       "word_count": len(words), "repetition": repetition, "expected_useful": record["expected_useful"]})
    if mode == "curated" and accepted:
        audit = duplicate_audit({r["id"]: r["training_text"] for r in accepted})
        removed = {name for group in audit["components"] for name in group[1:]}
        for row in ledger:
            if row["id"] in removed:
                row["accepted"] = False; row["reasons"].append("verified_near_duplicate")
        accepted = [r for r in accepted if r["id"] not in removed]
    else:
        audit = duplicate_audit({r["id"]: r["training_text"] for r in accepted})
    return {"mode": mode, "records": accepted, "ledger": ledger, "duplicate_audit": audit,
            "counts": {"input": len(records), "retained": len(accepted), "removed": len(records)-len(accepted)},
            "false_removals": [r["id"] for r in ledger if r["expected_useful"] and not r["accepted"]]}


def _fixed_generation(model, prefix="The ", maximum=32):
    import torch
    from .pretraining_lab import BOS, EOS
    ids, generated, stopped = [BOS, *prefix.encode()], [], False
    was_training = model.training; model.eval()
    with torch.no_grad():
        for _ in range(maximum):
            token = int(model(torch.tensor([ids[-64:]]))[0, -1].argmax())
            generated.append(token); ids.append(token)
            if token == EOS:
                stopped = True; break
    model.train(was_training)
    content = bytes(i for i in generated if i < 256).decode("utf-8", errors="replace")
    return {"prefix": prefix, "raw_ids": generated, "text": content, "natural_eos": stopped,
            "maximum_new_tokens": maximum, "bos_predictions": generated.count(BOS)}


def learning_reference(*, seeds=(1701, 1702, 1703), updates=24, fixture_path=DEFAULT_FIXTURE):
    """Three matched arms; full windows give exactly 32 valid targets per update.

    Filtering changes the eligible training pool. All omitted tails are counted.
    The frozen test text is never used to choose filters, seeds or checkpoints.
    """
    import torch
    from .pretraining_lab import TrainingSession, evaluate, make_windows
    if not 4 <= updates <= 24 or len(set(seeds)) != len(seeds) or len(seeds) != 3:
        raise ValueError("Exactly three distinct seeds and 4..24 updates required")
    fixture = load_fixture(fixture_path)
    train = [r for r in fixture if r["split"] == "train"]
    held_out = [r for r in fixture if r["split"] == "test"]
    valid_documents = [(r["id"], extract_text(r["html"])) for r in held_out]
    rows = []; started = time.monotonic()
    for seed in seeds:
        initial_hash = None
        for mode in ("raw", "exact", "curated"):
            prepared = prepare_corpus(train, mode)
            documents = [(r["id"], r["training_text"]) for r in prepared["records"]]
            session = TrainingSession(total=updates, train_documents=documents, valid_documents=valid_documents,
                                      seed=seed, full_windows_only=True)
            digest = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in session.model.parameters())).hexdigest()
            if initial_hash is not None and digest != initial_hash:
                raise AssertionError("Within-seed parent initialization differs")
            initial_hash = digest
            baseline = evaluate(session.model, session.valid)
            history = [session.update() for _ in range(updates)]
            if [r["tokens"] for r in history] != [32*(i+1) for i in range(updates)]:
                raise AssertionError("Matched valid-target exposure failed")
            owner_counts = Counter(session.train["documents"][i] for r in history for i in r["indices"])
            slice_nll = {}
            for slice_name in sorted({r["slice"] for r in held_out}):
                docs = [(r["id"], extract_text(r["html"])) for r in held_out if r["slice"] == slice_name]
                slice_nll[slice_name] = evaluate(session.model, make_windows(docs))
            rows.append({"seed": seed, "arm": mode, "initial_parameters_sha256": digest,
                         "contract": session.contract, "retention_counts": prepared["counts"],
                         "false_removals": prepared["false_removals"], "history": history,
                         "held_out_nll_before": baseline, "held_out_nll_after": evaluate(session.model, session.valid),
                         "held_out_slice_nll": slice_nll, "generation": _fixed_generation(session.model),
                         "pool_windows": len(session.train["x"]), "omitted_tail_targets": session.omitted_train_targets,
                         "target_presentations": session.tokens, "effective_pool_epochs": session.tokens/(16*len(session.train["x"])),
                         "distinct_windows_presented": len({i for h in history for i in h["indices"]}),
                         "window_presentations_by_document": dict(owner_counts)})
    return {"scope": "CPU fixture learning; original local text; no pretrained model or crawl",
            "seeds": list(seeds), "updates_per_arm": updates, "arms": rows,
            "runtime_seconds": time.monotonic()-started, "torch": torch.__version__,
            "fixture_sha256": hashlib.sha256(Path(fixture_path).read_bytes()).hexdigest(),
            "selection_rule": "final checkpoint only; fixed filters, seeds, test slices and greedy prefix",
            "limitations": ["Small fixture cannot establish general data-quality gains",
                            "Full-window matching omits counted tails and changes training-text coverage",
                            "Rules remove useful short/code text; discarded material remains in the ledger"]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(learning_reference(), indent=2, allow_nan=False)+"\n")
