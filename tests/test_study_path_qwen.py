import unittest

from dongxi_llms.study_path import qwen as Q


class StudyPathQwenDataTest(unittest.TestCase):
    def test_parse_and_pair(self):
        row = {"chosen": "\n\nHuman: Hi?\n\nAssistant: Hello.\n\nHuman: Garlic?\n\nAssistant: Plant cloves.",
               "rejected": "\n\nHuman: Hi?\n\nAssistant: Hello.\n\nHuman: Garlic?\n\nAssistant: No idea."}
        pair = Q.to_pair(row)
        self.assertEqual(len(pair["prompt"]), 3)
        self.assertEqual(pair["chosen"], "Plant cloves.")
        self.assertEqual(pair["rejected"], "No idea.")
        self.assertIsNone(Q.to_pair({"chosen": row["chosen"], "rejected": row["chosen"]}))
        self.assertEqual(Q.first_question(row), "Hi?")

    def test_render_ends_on_im_end(self):
        try:
            tok = Q.load_tokenizer()
        except Exception as error:  # tokenizer cache absent on this machine
            self.skipTest(f"Qwen3 tokenizer not cached: {error}")
        text = Q.render(tok, [{"role": "user", "content": "Hi?"}], "Hello.")
        self.assertTrue(text.endswith("<|im_end|>"))
        self.assertTrue(Q.chat_prompt(tok, "Hi?").endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertEqual(Q.gib(2**30), 1.0)


if __name__ == "__main__":
    unittest.main()
