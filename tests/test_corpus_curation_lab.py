"""Small independent oracles for curation and unchanged training contracts."""
from copy import deepcopy
import unittest

import torch

from dongxi_llms.corpus_curation_lab import (extract_text, matching_text, shingles,
    jaccard, duplicate_audit, lsh_candidates, split_audit, source_exposure,
    load_fixture, prepare_corpus)
from dongxi_llms.pretraining_lab import TrainingSession, TRAIN, VALID, IGNORE


class CorpusCurationTests(unittest.TestCase):
    def test_html_visibility_and_matching_preserve_training_text(self):
        html = '<head><title>hidden</title></head><nav>links</nav><p>Café &amp; code</p><script>noise</script><pre>x &lt; 3</pre>'
        text = extract_text(html)
        self.assertEqual(text, "Café & code\nx < 3")
        self.assertEqual(matching_text(text), "cafe code x 3")
        self.assertIn("Café", text)

    def test_small_exact_oracle_and_transitive_graph(self):
        # Sets have sizes 4, 5, 4. AB=BC=4/5, AC=3/5.
        texts = {"A":"red blue green white", "B":"red blue green white black", "C":"red blue green black", "empty":""}
        audit = duplicate_audit(texts, n=1, num_hashes=16, bands=16, threshold=.75)
        self.assertEqual(audit["oracle_edges"], [("A", "B"), ("B", "C")])
        self.assertEqual(audit["verified_edges"], audit["oracle_edges"])
        self.assertIn(["A", "B", "C"], audit["components"])
        self.assertEqual(jaccard(shingles(texts["A"], 1), shingles(texts["C"], 1)), .6)
        self.assertNotIn(("A", "C"), audit["verified_edges"])
        records = [{"id":k,"source_group":k,"split":"test" if k == "C" else "train"} for k in texts]
        self.assertTrue(split_audit(records, audit["components"]))

    def test_candidates_are_not_similarity_edges(self):
        # Deliberately colliding signatures retrieve two completely unrelated docs.
        self.assertEqual(lsh_candidates({"a":(4,5), "b":(4,9)}, bands=2), {("a","b")})
        audit = duplicate_audit({"a":"north", "b":"south"}, n=3)
        self.assertEqual(audit["verified_edges"], [])
        # Original three-gram sets share one of five union items: J=1/5.
        # Permissive one-hash bands retrieve the shared item; verification rejects.
        collision = duplicate_audit({"F":"red blue green white black", "G":"red blue green orange purple"}, bands=64)
        self.assertEqual(collision["similarities"][0]["jaccard"], .2)
        self.assertEqual(collision["false_candidates"], [("F", "G")])
        self.assertEqual(collision["verified_edges"], [])
        self.assertEqual(collision["candidate_precision"], 0.)
        self.assertEqual(jaccard(set(), set()), 0.)
        self.assertEqual(shingles("tiny", 3), {("tiny",)})
        with self.assertRaises(ValueError): lsh_candidates({"x":(1,2,3)}, bands=2)

    def test_retention_counts_and_false_removal_remain_visible(self):
        train = [r for r in load_fixture() if r["split"] == "train"]
        raw, exact, curated = [prepare_corpus(train, mode) for mode in ("raw", "exact", "curated")]
        self.assertEqual(raw["counts"], {"input":11,"retained":10,"removed":1})
        self.assertEqual(exact["counts"]["retained"], 9)
        self.assertIn("t06", curated["false_removals"])
        self.assertIn("t07", curated["false_removals"])
        for result in (raw, exact, curated):
            self.assertEqual(result["counts"]["input"], result["counts"]["retained"]+result["counts"]["removed"])
            self.assertEqual(sum(row["accepted"] for row in result["ledger"]), len(result["records"]))
        fixture = load_fixture(); audit = duplicate_audit({r["id"]:extract_text(r["html"]) for r in fixture})
        self.assertEqual(split_audit(fixture, audit["components"]), [])

    def test_source_cap_has_token_units(self):
        result = source_exposure({"rare":.5,"large":.5}, {"rare":10,"large":1000}, 100, epoch_cap=2)
        self.assertEqual(result["sources"]["rare"]["effective_epochs"], 5.)
        self.assertEqual(result["sources"]["large"]["effective_epochs"], .05)
        self.assertFalse(result["cap_feasible"])

    def test_default_injection_reproduces_existing_update_and_recovery(self):
        old = TrainingSession(total=4)
        explicit = TrainingSession(total=4, train_documents=TRAIN, valid_documents=VALID, seed=808, length=16)
        self.assertEqual(old.contract, explicit.contract)
        self.assertEqual(old.update(), explicit.update())
        for p,q in zip(old.model.parameters(), explicit.model.parameters()): self.assertTrue(torch.equal(p,q))
        state = deepcopy(old.checkpoint()); expected = old.update()
        restored = TrainingSession(total=4); restored.restore(state)
        self.assertEqual(expected, restored.update())

    def test_full_window_geometry_matches_actual_target_count(self):
        train = (("a", "A complete window has enough bytes to fill several positions."),)
        valid = (("b", "Independent held out content."),)
        s = TrainingSession(total=4, train_documents=train, valid_documents=valid, seed=1, full_windows_only=True)
        self.assertTrue(bool(((s.train["y"] != IGNORE).sum(1) == 16).all()))
        self.assertGreater(s.omitted_train_targets, 0)
        self.assertEqual(s.update()["tokens"], 32)


if __name__ == "__main__": unittest.main()
