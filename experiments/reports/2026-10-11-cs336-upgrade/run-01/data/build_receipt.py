"""Bind final CS33602/03 sources and actual bounded verification evidence."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
BASE = "78640952e7e6ffb35c57f4ab48569e6a660e93b6"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


files = ["src/dongxi_llms/tokenizer_contracts.py", "src/dongxi_llms/corpus_curation_lab.py",
         "src/dongxi_llms/pretraining_lab.py", "tests/test_tokenizer_contracts.py",
         "tests/test_corpus_curation_lab.py", "experiments/fixtures/cs336/corpus.json",
         "notebooks/extensions/cs336/05_tokenizer_contracts.ipynb",
         "notebooks/extensions/cs336/06_corpus_curation.ipynb",
         "notebooks/extensions/cs336/05_tokenizer_contracts-solutions.md",
         "notebooks/extensions/cs336/06_corpus_curation-solutions.md"]
for name in files:
    destination = OUT / "source-final" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((ROOT / name).read_bytes())

baseline = json.loads((OUT.parent / "baseline.json").read_text())
preserved = []
for name in baseline["notebook_cells"]:
    original = subprocess.check_output(["git", "show", BASE + ":" + name], cwd=ROOT)
    assert original == (ROOT / name).read_bytes(), name
    preserved.append({"path": name, "sha256": sha(ROOT / name)})

executions = json.loads((OUT / "notebook-execution.json").read_text())
for row in executions:
    assert sha(ROOT / row["source"]) == row["source_sha256"]
reference = json.loads((OUT / "learning-reference.json").read_text())
summary = [{"seed": r["seed"], "arm": r["arm"], "held_out_nll": r["held_out_nll_after"],
            "targets": r["target_presentations"], "retained": r["retention_counts"]["retained"],
            "pool_windows": r["pool_windows"], "pool_epochs": r["effective_pool_epochs"],
            "omitted_tail_targets": r["omitted_tail_targets"], "generation": r["generation"]}
           for r in reference["arms"]]
from dongxi_llms.corpus_curation_lab import load_fixture, extract_text, prepare_corpus, duplicate_audit, split_audit
fixture = load_fixture(); train = [r for r in fixture if r["split"] == "train"]
arms = {m: prepare_corpus(train, m) for m in ("raw", "exact", "curated")}
audits = {"default": duplicate_audit({r["id"]:extract_text(r["html"]) for r in train}),
          "one_band": duplicate_audit({r["id"]:extract_text(r["html"]) for r in train}, bands=1),
          "false_candidate_control": duplicate_audit({"F":"red blue green white black", "G":"red blue green orange purple"}, bands=64)}
fixture_audit = duplicate_audit({r["id"]:extract_text(r["html"]) for r in fixture})
issues = split_audit(fixture, fixture_audit["components"])
assert not issues

result = {
    "scope": "CS33602+03 author implementation receipt; no learner completion or model-scale claims",
    "baseline_commit": BASE,
    "sources_sha256": {name: sha(ROOT / name) for name in files},
    "source_snapshot": str((OUT / "source-final").relative_to(ROOT)),
    "environment": {"python_executable": sys.executable, "python": platform.python_version(),
                    "torch": torch.__version__, "platform": platform.platform(),
                    "cuda_available": torch.cuda.is_available(), "threads": torch.get_num_threads(),
                    "uv_lock_sha256": sha(ROOT / "uv.lock"), "kernel": "dongxi-cs336"},
    "api": {
        "tokenizer": "TokenizerContracts.train(documents, merge_count=16, **policies), encode/decode, segments, encode_iterable(framing=records|chunks,max_record_chars), to_json/from_json, identity; ByteScore/ByteScoringContract + aggregate_byte_scores",
        "corpus": "extract_text, matching_text, shingles, jaccard, minhash_signature, lsh_candidates, duplicate_audit, split_audit, source_exposure, load_fixture, prepare_corpus, learning_reference",
        "minimal_training_injection": "TrainingSession(total=24, *, train_documents=None, valid_documents=None, seed=808, length=16, full_windows_only=False); defaults preserved; full-window tails counted in explicit contract"},
    "reuse": ["Existing course_foundations.ByteBPE merges/encode; no second BPE algorithm",
              "Existing byte decoder, document-local windows, split check, AdamW, clipping, warmup/cosine schedule and recovery in pretraining_lab",
              "Standard-library HTMLParser; original fixtures, no imported upstream code/prose/tests"],
    "teaching_routes": [{"source": executions[0]["source"], "day": 2, "chapter": 2, "pairs": 5,
                         "controlled_changes": ["Whole versus word/space segmentation", "Recognized versus rejected controls; overlap longest-match", "Incorrect independent-record framing versus complete chunks", "Saved-policy corruption; normalization and malformed-byte decoding", "Wrong mean of document byte rates and incompatible EOS"]},
                        {"source": executions[1]["source"], "day": 8, "chapter": 6, "pairs": 6,
                         "controlled_changes": ["Matching normalization versus preserved training text", "Eight-word versus three-word rule; adverse useful removals", "One long versus many short LSH bands; exact false-candidate control", "Planted transitive-component split leak", "Finite source-capacity intervention", "Three fixed corpus arms, matched initialization/targets/update schedule"]}],
    "notebook_contract": "Visible Prediction+Checkpoint before each reference marker/code; Reference explanation immediately after; stable cell IDs; labeled regeneration plots adjacent to references",
    "preserved_original_notebook_bytes": {"count": len(preserved), "files": preserved},
    "focused_tests": {"command": "python -m unittest tests.test_tokenizer_contracts tests.test_corpus_curation_lab tests.test_pretraining_lab -v", "count": 25, "exit_code": 0,
                      "log": str((OUT / "focused-tests.log").relative_to(ROOT)), "log_sha256": sha(OUT / "focused-tests.log"),
                      "independent_oracles": ["Existing/default BPE merge/ID equality and unseen UTF-8", "Hand-computed weighted byte rates and incompatible contracts", "Original word-set Jaccard 0.8/0.8/0.6; transitive grouping", "Shared one-of-five trigrams gives Jaccard0.2 and verified false retrieval", "Finite-capacity token arithmetic", "Restored continuation and actual full-window target counts"]},
    "baseline_default_compatibility": {"receipt": str((OUT / "default-compatibility.json").relative_to(ROOT)),
                                       "sha256": sha(OUT / "default-compatibility.json"), "original_source_loaded_from_git_read_only": True,
                                       "result": "Contracts, every four-update record, and every parameter tensor match bitwise at each step"},
    "notebook_execution": executions,
    "visual_review": {"images": 8, "status": "accepted after inspecting every extracted PNG", "observations": ["Labels/units readable and unclipped", "Actual tokenizer bars 1/3 and 1/2/1; weighted byte rate1.5", "Waterfall11→10→9→5; slice adverse removals visible", "Empty t11 Jaccard diagonal zero", "Verified AB/BC graph with schematic axes; label B clears title", "Equal token shares versus epochs5/0.05; all nine curves/final points preserved"]},
    "corpus_audit": {"retention": {m:r["counts"] for m,r in arms.items()}, "ledger": {m:r["ledger"] for m,r in arms.items()},
                     "useful_removals": arms["curated"]["false_removals"], "retrieval_controls": audits,
                     "split_issues": issues, "held_out_ids": [r["id"] for r in fixture if r["split"] == "test"],
                     "training_never_uses_test": True},
    "learning_reference": {"path": str((OUT / "learning-reference.json").relative_to(ROOT)), "sha256": sha(OUT / "learning-reference.json"),
                           "seeds": reference["seeds"], "updates_per_arm": 24, "fits": 9, "rows": summary,
                           "runtime_seconds": reference["runtime_seconds"], "selection_rule": reference["selection_rule"],
                           "negative_outcomes": "Curated fixed held-out NLL exceeds exact for all three seeds; all greedy outputs reach cap32 with no natural EOS"},
    "failures": [{"receipt": str((OUT / "failed-attempts/attempt-1.json").relative_to(ROOT)),
                  "issue": "Notebook plot passed dictionary category input to matplotlib", "resolution": "Explicit label lists; final fresh kernels pass"}],
    "limitations": ["Teaching segmentation is original, not GPT-2/Qwen compatibility", "Chunks buffer a complete single record; not constant-memory arbitrary streaming",
                    "Scored byte spans and context/mask/EOS conventions remain explicit caller responsibility; toy probabilities are not learned-model comparison",
                    "HTML extraction has named-container rules, not production main-content extraction or privacy removal", "Lexical normalization may conflate meaningful punctuation/accents; only used for matching",
                    "All-pairs duplicate oracle is for the bounded fixture; connected components are a policy rather than pairwise similarity",
                    "Complete-only training windows omit counted tails; matched total target exposure does not match distinct data coverage/repetitions",
                    "Tiny authored corpus, fixed three seeds and24updates cannot establish general corpus quality or coherent story ability"]}
(OUT / "data-implementation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print("Bound author receipt:", len(files), "sources;", len(preserved), "unchanged old notebooks;", len(summary), "CPU fits")
