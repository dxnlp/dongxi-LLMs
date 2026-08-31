"""Unit tests for Unicode and tokenizer-mechanics measurements."""

import unicodedata
import unittest

from dongxi_llms.tokenizer_mechanics_lab import analyze_unicode


class TokenizerMechanicsLabTest(unittest.TestCase):
    def test_nfc_and_nfd_cafe_have_same_grapheme_count(self) -> None:
        nfc = analyze_unicode("café")
        nfd = analyze_unicode(unicodedata.normalize("NFD", "café"))

        self.assertEqual(nfc["code_point_count"], 4)
        self.assertEqual(nfd["code_point_count"], 5)
        self.assertEqual(nfc["grapheme_cluster_count"], 4)
        self.assertEqual(nfd["grapheme_cluster_count"], 4)
        self.assertEqual(nfc["nfc"], nfd["nfc"])

    def test_family_emoji_is_one_grapheme_with_seven_code_points(self) -> None:
        result = analyze_unicode("👨‍👩‍👧‍👦")

        self.assertEqual(result["code_point_count"], 7)
        self.assertEqual(result["utf8_byte_count"], 25)
        self.assertEqual(result["grapheme_cluster_count"], 1)


if __name__ == "__main__":
    unittest.main()
