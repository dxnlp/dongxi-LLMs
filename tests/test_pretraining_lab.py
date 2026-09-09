import tempfile
import unittest
from pathlib import Path

import torch

from dongxi_llms import pretraining_lab as lab


class PretrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_split_contamination(self):
        with self.assertRaises(ValueError):
            lab.check_splits((("a", "A CAT"),), (("b", " a  cat "),))
        with self.assertRaises(ValueError):
            lab.check_splits((("a", "cat"),), (("a", "dog"),))

    def test_window_alignment_and_budget(self):
        windows = lab.make_windows((("a", "abc"),), 3)
        self.assertEqual(windows["x"].tolist(), [[lab.BOS, 97, 98], [99, lab.EOS, lab.EOS]])
        self.assertEqual(windows["y"].tolist(), [[97, 98, 99], [lab.EOS, -100, -100]])
        train, valid = lab.fixture()
        self.assertEqual(int((train["y"] != -100).sum()), sum(len(t.encode())+1 for _, t in lab.TRAIN))
        self.assertFalse(set(train["documents"]) & set(valid["documents"]))

    def test_unicode_byte_labels(self):
        w = lab.make_windows((("zh", "数"),), 8)
        self.assertEqual(w["y"][0, :4].tolist(), [230, 149, 176, lab.EOS])

    def test_accumulation(self):
        audit = lab.accumulation_audit()
        self.assertLess(audit["correct_error"], 1e-10)
        self.assertGreater(audit["wrong_error"], 1e-4)

    def test_adamw_multiple_steps(self):
        parameter = torch.nn.Parameter(torch.tensor([.4, -.2], dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=.01, weight_decay=.01, foreach=False)
        expected, m, v = parameter.detach().clone(), torch.zeros_like(parameter), torch.zeros_like(parameter)
        for i, g in enumerate(([1., -3.], [-2., .5], [.1, .2]), 1):
            parameter.grad = torch.tensor(g, dtype=torch.float64)
            expected, m, v = lab.adamw_reference(expected, parameter.grad, m, v, i, .01)
            optimizer.step()
            torch.testing.assert_close(parameter, expected, atol=1e-14, rtol=1e-14)
        self.assertLess(lab.optimizer_audit()["error"], 1e-12)

    def test_schedule(self):
        rates = [lab.learning_rate(s) for s in range(24)]
        self.assertEqual(rates[2], .01)
        self.assertEqual(rates[-1], .001)
        self.assertTrue(all(a > b for a, b in zip(rates[2:], rates[3:])))
        with self.assertRaises(ValueError):
            lab.learning_rate(24)

    def test_clipping(self):
        audit = lab.clipping_audit()
        self.assertGreater(audit["before"], 1.)
        self.assertLessEqual(audit["after"], 1.000001)
        self.assertAlmostEqual(audit["cosine"], 1., places=5)
        with self.assertRaises(ValueError):
            lab.clipping_audit(0)

    def test_stream_resume_across_epochs(self):
        stream = lab.WindowStream(3)
        stream.take(5)
        state = stream.state_dict()
        expected = stream.take(12)
        restored = lab.WindowStream(3)
        restored.load_state_dict(state)
        self.assertEqual(restored.take(12), expected)

    def test_validation_weighting_and_mode(self):
        model = lab.make_model(dtype=torch.float64)
        _, valid = lab.fixture()
        a, b = lab.evaluate(model, valid, 1), lab.evaluate(model, valid, 3)
        self.assertAlmostEqual(a, b, places=12)
        self.assertTrue(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        model.eval()
        lab.evaluate(model, valid)
        self.assertFalse(model.training)

    def test_checkpoint_contract_and_rng(self):
        session = lab.TrainingSession()
        state = session.checkpoint()
        expected = torch.rand(5)
        session.restore(state)
        torch.testing.assert_close(torch.rand(5), expected, atol=0, rtol=0)
        state["contract"]["tokenizer"] = "different"
        with self.assertRaises(ValueError):
            session.restore(state)

    def test_recovery_and_negative_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = lab.recovery_audit(Path(directory)/"checkpoint.pt")
        self.assertEqual(audit["complete"]["parameter_error"], 0.)
        self.assertTrue(audit["complete"]["same_history"])
        self.assertTrue(audit["no_optimizer"]["same_batches"])
        self.assertFalse(audit["no_cursor"]["same_batches"])
        for name in ("no_optimizer", "no_cursor"):
            self.assertGreater(audit[name]["parameter_error"], 1e-5)

    def test_figures_preserve_tensors_and_rng(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from dongxi_llms import pretraining_visuals as viz
        train, _ = lab.fixture()
        before_x, before_y = train["x"].clone(), train["y"].clone()
        audit = lab.optimizer_audit()
        before_gradient = audit["gradient"].clone()
        rng = torch.get_rng_state().clone()
        for fig in (viz.process_map([0, 1]), viz.window_map(train), viz.optimizer_plot(audit),
                    viz.schedule_plot([lab.learning_rate(i) for i in range(24)])):
            fig.canvas.draw()
            plt.close(fig)
        torch.testing.assert_close(train["x"], before_x)
        torch.testing.assert_close(train["y"], before_y)
        torch.testing.assert_close(audit["gradient"], before_gradient)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
