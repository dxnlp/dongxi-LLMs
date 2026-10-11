"""Boundary examples are independently authored, not upstream test copies."""
import json
import math
import unittest

from dongxi_llms.course_foundations import ByteBPE
from dongxi_llms.tokenizer_contracts import (TokenizerContracts, ByteScore,
                                           ByteScoringContract, aggregate_byte_scores)


class TokenizerContractTests(unittest.TestCase):
    def test_existing_byte_model_is_reused_without_changed_defaults(self):
        corpus = ["garden gate garden gate", "å 数 café"]
        old = ByteBPE.train(corpus, 12)
        wrapped = TokenizerContracts.train(corpus, 12)
        self.assertEqual(old.merges, wrapped.model.merges)
        self.assertEqual(old.counts, wrapped.model.counts)
        for text in ["", "garden gate", "未见文字", "e\u0301"]:
            self.assertEqual(old.encode(text), wrapped.encode(text))
            self.assertEqual(wrapped.decode(wrapped.encode(text)), text)

    def test_boundaries_change_ids_without_changing_text(self):
        whole = TokenizerContracts.train(["ab cd"]*10, 12)
        words = TokenizerContracts.train(["ab cd"]*10, 12, segmentation="words")
        self.assertEqual(len(whole.encode("ab cd")), 1)
        self.assertEqual(len(words.encode("ab cd")), 3)
        self.assertEqual(words.decode(words.encode("ab cd")), "ab cd")

    def test_reserved_markers_are_not_merge_training_data(self):
        obj = TokenizerContracts.train(["aa<STOP><STOP_LONG>bb"]*4, 30,
                                      special_tokens=("<STOP>", "<STOP_LONG>"))
        ids = obj.encode("aa<STOP_LONG>bb")
        self.assertEqual(ids[1], obj.special_ids["<STOP_LONG>"])
        self.assertNotIn(b"<STOP>", obj.model.pieces)
        self.assertEqual(obj.decode(ids), "aa<STOP_LONG>bb")
        reject = TokenizerContracts(obj.model, special_tokens=("<STOP>",), special_policy="reject")
        with self.assertRaises(ValueError): reject.encode("user said <STOP>")
        overlapping = TokenizerContracts(obj.model, special_tokens=("<S>", "<S><S>"))
        self.assertEqual(overlapping.encode("<S><S>"), [overlapping.special_ids["<S><S>"]])

    def test_framing_is_an_actual_semantic_choice(self):
        obj = TokenizerContracts.train(["garden"]*5, 10)
        self.assertEqual(len(obj.encode("garden")), 1)
        self.assertEqual(list(obj.encode_iterable(["gar", "den"], framing="chunks")), obj.encode("garden"))
        self.assertNotEqual(list(obj.encode_iterable(["gar", "den"], framing="records")), obj.encode("garden"))
        self.assertEqual(obj.decode(list(obj.encode_iterable(["gar", "den"]))), "garden")
        with self.assertRaises(ValueError): list(obj.encode_iterable(["gar", "den"], framing="chunks", max_record_chars=5))

    def test_serialization_and_normalization_have_distinct_claims(self):
        obj = TokenizerContracts.train(["café"], 6, normalization="NFC", special_tokens=("<E>",))
        reload = TokenizerContracts.from_json(obj.to_json())
        self.assertEqual(obj.identity, reload.identity)
        self.assertEqual(reload.decode(reload.encode("cafe\u0301<E>")), "café<E>")
        broken = json.loads(obj.to_json()); broken["tokenizer"]["normalization"] = "none"
        with self.assertRaises(ValueError): TokenizerContracts.from_json(json.dumps(broken))
        with self.assertRaises(UnicodeDecodeError): obj.decode([230])
        replace = TokenizerContracts(obj.model, decode_errors="replace")
        self.assertEqual(replace.decode([230]), "\ufffd")

    def test_byte_score_uses_a_corpus_sum_not_mean_document_rates(self):
        result = aggregate_byte_scores([ByteScore(math.log(4), "a"), ByteScore(math.log(16), "數")])
        self.assertEqual(result["scored_bytes"], 4)
        self.assertAlmostEqual(result["bits_per_byte"], 1.5)
        equivalent = aggregate_byte_scores([ByteScore(-math.log(.5)-math.log(.5), "a")])
        self.assertAlmostEqual(equivalent["bits_per_byte"], 2.)
        with self.assertRaises(ValueError): aggregate_byte_scores([ByteScore(1., "a"), ByteScore(1., "b", ByteScoringContract(eos="included"))])
        for bad in [ByteScore(float("nan"), "x"), ByteScore(-1., "x"), ByteScore(1., "")]:
            with self.assertRaises(ValueError): aggregate_byte_scores([bad])


if __name__ == "__main__": unittest.main()
