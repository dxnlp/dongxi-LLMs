"""The workbook checks must pass correct answers, catch typical mistakes and never crash."""
import contextlib
import io
import unittest

import torch
import torch.nn.functional as F

from dongxi_llms import sft_workbook as wb


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = fn(*args, **kwargs)
    return result, out.getvalue()


LEVELS = torch.linspace(-1, 1, 16)


class WorkbookCheckTests(unittest.TestCase):
    def test_not_attempted_never_raises(self):
        for check in (wb.check_shifted_loss, wb.check_rank_one, wb.check_top_r, wb.check_energy,
                      wb.check_lora_param_count, wb.check_lora_linear, wb.check_trainable_count, wb.check_memory):
            ok, out = quiet(check, None)
            self.assertFalse(ok)
            self.assertIn("not attempted", out)

    def test_placeholder_code_reports_instead_of_crashing(self):
        def unfinished(logits, labels):
            return F.cross_entropy(..., labels)
        ok, out = quiet(wb.check_shifted_loss, unfinished)
        self.assertFalse(ok)
        self.assertIn("raised", out)
        with self.assertRaises(AssertionError):
            quiet(wb.check_shifted_loss, unfinished, strict=True)

    def test_shifted_loss_detects_missing_shift(self):
        ok, _ = quiet(wb.check_shifted_loss,
                      lambda lg, lb: F.cross_entropy(lg[:-1], lb[1:], ignore_index=-100), strict=True)
        self.assertTrue(ok)
        ok, out = quiet(wb.check_shifted_loss, lambda lg, lb: F.cross_entropy(lg, lb, ignore_index=-100))
        self.assertFalse(ok)
        self.assertIn("copying-task", out)

    def test_rank_one_and_top_r_and_energy(self):
        u = torch.tensor([[1.], [2.], [3.]])
        self.assertTrue(quiet(wb.check_rank_one, u @ u.T, strict=True)[0])
        self.assertFalse(quiet(wb.check_rank_one, torch.eye(3))[0])
        self.assertTrue(quiet(wb.check_top_r, wb.ref_top_r, strict=True)[0])
        self.assertFalse(quiet(wb.check_top_r, lambda M, r: M)[0])
        good = lambda M, r: (torch.linalg.svdvals(M)[:r] ** 2).sum() / (torch.linalg.svdvals(M) ** 2).sum()
        self.assertTrue(quiet(wb.check_energy, good, strict=True)[0])
        unsquared = lambda M, r: torch.linalg.svdvals(M)[:r].sum() / torch.linalg.svdvals(M).sum()
        self.assertFalse(quiet(wb.check_energy, unsquared)[0])

    def test_lora_count_and_layer(self):
        self.assertTrue(quiet(wb.check_lora_param_count, lambda o, i, r: r * i + o * r, strict=True)[0])
        self.assertFalse(quiet(wb.check_lora_param_count, lambda o, i, r: r * i)[0])
        self.assertTrue(quiet(wb.check_lora_linear, wb.RefLoRALinear, strict=True)[0])

        class BothZero(wb.RefLoRALinear):
            def __init__(self, base, r=8, alpha=16):
                super().__init__(base, r, alpha)
                torch.nn.init.zeros_(self.A)
        ok, out = quiet(wb.check_lora_linear, BothZero)
        self.assertFalse(ok)
        self.assertIn("A starts random", out)

        class Unfrozen(wb.RefLoRALinear):
            def __init__(self, base, r=8, alpha=16):
                super().__init__(base, r, alpha)
                self.base.weight.requires_grad_(True)
        self.assertFalse(quiet(wb.check_lora_linear, Unfrozen)[0])
        self.assertTrue(quiet(wb.check_trainable_count, 1_146_880, strict=True)[0])

    def test_train_step(self):
        def step(model, batch, opt):
            loss = model(**batch).loss
            loss.backward(); opt.step(); opt.zero_grad()
            return float(loss)
        self.assertTrue(quiet(wb.check_train_step, step, strict=True)[0])
        def no_zero_grad(model, batch, opt):
            loss = model(**batch).loss
            loss.backward(); opt.step()
            return float(loss)
        ok, out = quiet(wb.check_train_step, no_zero_grad)
        self.assertFalse(ok)
        self.assertIn("cleared", out)

    def test_quantize_pack_and_memory(self):
        self.assertTrue(quiet(wb.check_quantize, wb.ref_quantize, wb.ref_dequantize, LEVELS, strict=True)[0])
        def global_scale(w, levels, block=64):
            codes, _ = wb.ref_quantize(w, levels, block)
            return codes, torch.full((w.numel() // block,), float(w.abs().max()))
        self.assertFalse(quiet(wb.check_quantize, global_scale, wb.ref_dequantize, LEVELS)[0])
        pack = lambda c: (c[:, 0::2] << 4) | c[:, 1::2]
        unpack = lambda p: torch.stack([p >> 4, p & 15], -1).reshape(p.shape[0], -1)
        self.assertTrue(quiet(wb.check_pack, pack, unpack, strict=True)[0])
        swapped = lambda c: (c[:, 1::2] << 4) | c[:, 0::2]
        self.assertFalse(quiet(wb.check_pack, swapped, unpack)[0])
        bytes_per = {"full": 16, "lora": 2, "qlora": 0.5}
        self.assertTrue(quiet(wb.check_memory, lambda n, m: n * bytes_per[m], strict=True)[0])
        self.assertFalse(quiet(wb.check_memory, lambda n, m: n * 4)[0])


class TokenizerCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from transformers import AutoTokenizer
            cls.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
        except Exception as error:  # pragma: no cover - depends on the local cache
            raise unittest.SkipTest(f"cached Qwen3-0.6B tokenizer unavailable: {error}")

    def test_build_labels_and_span(self):
        tok = self.tok
        header = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
        think = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        ref = lambda ids: wb.ref_build_labels(ids, header, think, im_end)
        self.assertTrue(quiet(wb.check_build_labels, ref, tok, strict=True)[0])
        def includes_newline(ids):
            labels = ref(ids)
            labels[-1] = ids[-1]
            return labels
        self.assertFalse(quiet(wb.check_build_labels, includes_newline, tok)[0])

        def span(ids, i):
            start = i + len(header)
            if ids[start:start + len(think)] == think:
                start += len(think)
            return start, ids.index(im_end, start)
        self.assertTrue(quiet(wb.check_answer_span, span, tok, strict=True)[0])
        self.assertFalse(quiet(wb.check_answer_span, lambda ids, i: (i, i + 3), tok)[0])

        messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}]
        ids = list(tok.apply_chat_template(messages, return_dict=True)["input_ids"])
        self.assertTrue(quiet(wb.check_render, ids, tok, messages, strict=True)[0])
        self.assertFalse(quiet(wb.check_render, ids[:-1], tok, messages)[0])


if __name__ == "__main__":
    unittest.main()
