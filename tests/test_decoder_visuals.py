"""Check displayed data and side-effect boundaries, not only figure creation."""
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from dongxi_llms import decoder_visuals as viz
from dongxi_llms import decoder_architecture as architecture


class DecoderVisualTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_lookup_renders_selected_rows_without_changing_data_or_rng(self):
        table = torch.arange(32., dtype=torch.float64).reshape(8,4).requires_grad_()
        ids = torch.tensor([2,5,2,7])
        original = table.detach().clone()
        rng = torch.get_rng_state().clone()
        figure = viz.embedding_lookup(table, ids)
        np.testing.assert_array_equal(figure.axes[1].images[0].get_array(), original[ids].numpy())
        torch.testing.assert_close(table, original)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIsNone(table.grad)
        figure.canvas.draw()

    def test_attention_maps_mask_and_scale(self):
        weights = torch.ones(4,6,6, dtype=torch.float64).tril()
        weights /= weights.sum(-1,keepdim=True)
        figure = viz.attention_maps(weights)
        for head, axis in enumerate(figure.axes[:4]):
            data = axis.images[0].get_array()
            np.testing.assert_array_equal(data.mask, np.triu(np.ones((6,6),bool),1))
            np.testing.assert_allclose(data.data, weights[head].numpy())
            self.assertEqual(axis.images[0].get_clim(), (0.,1.))
        figure.canvas.draw()

    def test_mixture_displays_actual_signed_contributions(self):
        weights = torch.tensor([.25,.75])
        values = torch.tensor([[2.,-4.],[-2.,8.]])
        figure = viz.value_mixture(weights,values)
        np.testing.assert_allclose(figure.axes[1].images[0].get_array(), (weights[:,None]*values).numpy())
        np.testing.assert_allclose(figure.axes[2].images[0].get_array(), (weights@values)[None].numpy())
        figure.canvas.draw()

    def test_comparison_heatmaps_share_scale(self):
        figure=viz.matrices([np.ones((3,4)),np.full((3,4),-5)],['First','Second'],'Compare')
        self.assertEqual(figure.axes[0].images[0].get_clim(),(-5.,5.))
        self.assertEqual(figure.axes[1].images[0].get_clim(),(-5.,5.))

    def test_learning_curve_alignment_and_probabilities(self):
        figure=viz.learning([2.,1.],.5,torch.zeros(2,3),torch.ones(2,3),torch.tensor([1,2]))
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(),[0,1,2])
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(),[2.,1.,.5])
        np.testing.assert_allclose(figure.axes[1].images[0].get_array(), np.full((2,3),1/3))
        figure.canvas.draw()

    def test_count_figures_use_supplied_values(self):
        rows=[dict(kv_heads=h,parameters=100*h,logical_kv_bytes=20*h,
                   dense_forward_matmul_flops=50*h) for h in (4,2,1)]
        figure=viz.costs(rows)
        self.assertEqual([b.get_height() for b in figure.axes[1].patches],[80,40,20])
        figure.canvas.draw()

    def test_architecture_maps_distinguish_position_mechanisms(self):
        baseline=architecture.model_map('embeddings')
        modern=architecture.model_map('positions',modern=True)
        baseline_text=[t.get_text() for t in baseline.axes[0].texts]
        modern_text=[t.get_text() for t in modern.axes[0].texts]
        self.assertIn('Token + position embeddings',baseline_text)
        self.assertIn('Token embedding',modern_text)
        self.assertIn('Final RMSNorm',modern_text)
        self.assertNotIn('Token + position embeddings',modern_text)
        baseline.canvas.draw(); modern.canvas.draw()

    def test_parameter_budget_uses_actual_ledger(self):
        figure = viz.parameter_budget({'Embeddings': 256, 'MLP': 3072})
        self.assertEqual([p.get_width() for p in figure.axes[0].patches], [256, 3072])
        figure.canvas.draw()

    def test_audit_tiles_match_report_values(self):
        figure = viz.audit_outcomes({'good': {'finite': True, 'causal': True, 'cache': True},
                                     'bad': {'finite': True, 'causal': False, 'cache': False}})
        np.testing.assert_array_equal(figure.axes[0].images[0].get_array(), [[1,1,1],[1,0,0]])
        self.assertEqual([t.get_text() for t in figure.axes[0].texts].count('FAIL'), 2)
        figure.canvas.draw()

    def test_all_architecture_focuses_render(self):
        for focus in ('embeddings','attention','residual','norm','mlp','assembly',
                      'training','modern','positions','gqa','recurrence'):
            figure=architecture.model_map(focus)
            figure.canvas.draw()
            plt.close(figure)
        with self.assertRaises(ValueError):
            architecture.model_map('unknown')

    def test_component_schematics_render_without_changing_rng(self):
        rng=torch.get_rng_state().clone()
        figures=[architecture.embedding_detail(),architecture.attention_detail(),
                 architecture.attention_detail(grouped=True),architecture.block_detail(),
                 architecture.block_detail(focus='norm',modern=True),architecture.mlp_detail(),
                 architecture.mlp_detail(gated=True),architecture.rope_detail(),
                 architecture.training_detail(),architecture.recurrence_detail()]
        for figure in figures:
            figure.canvas.draw()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))


if __name__=='__main__':
    unittest.main()
