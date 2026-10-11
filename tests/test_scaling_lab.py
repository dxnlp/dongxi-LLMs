import copy
import math
import unittest

from dongxi_llms.scaling_lab import (
    ScalingFitError, bootstrap_predictions, evaluate_holdout, fit_isoflops,
    freeze_prediction, simulated_rows,
)


class ScalingTests(unittest.TestCase):
    def setUp(self):
        self.calibration = simulated_rows(budgets=(6e6, 24e6, 96e6))

    def test_known_optimum_and_exponents_from_independent_symmetry(self):
        law = fit_isoflops(self.calibration)
        # At fixed C, equal positive N and D minimize 1/sqrt(N)+1/sqrt(D).
        self.assertAlmostEqual(law.parameter_exponent, .5, places=12)
        self.assertAlmostEqual(law.excess_loss_exponent, -.25, places=12)
        prediction = freeze_prediction(law, 384e6)
        expected_n = math.sqrt(384e6/6)
        self.assertAlmostEqual(prediction['prediction']['parameters'], expected_n, places=8)
        self.assertAlmostEqual(prediction['prediction']['loss'], 2+40/math.sqrt(expected_n), places=12)
        result = evaluate_holdout(prediction, simulated_rows(budgets=(384e6,)))
        self.assertEqual(len(result['entire_budget_rows']), 15)
        self.assertAlmostEqual(result['parameter_relative_error'], 0., places=12)
        self.assertAlmostEqual(result['optimal_loss_error'], 0., places=12)

    def test_sparse_budget_and_boundary_minimum_failures(self):
        with self.assertRaisesRegex(ScalingFitError, 'three distinct'):
            fit_isoflops(simulated_rows(budgets=(6e6, 24e6)))
        boundary = [r for r in self.calibration if r['parameters'] <= math.sqrt(r['compute_budget']/6)]
        with self.assertRaisesRegex(ScalingFitError, 'boundary') as caught:
            fit_isoflops(boundary)
        self.assertTrue(caught.exception.diagnostics)

    def test_invalid_log_inputs_and_floor(self):
        for value in [0, -1, float('nan'), float('inf')]:
            rows = copy.deepcopy(self.calibration); rows[0]['loss'] = value
            with self.assertRaises(ScalingFitError):
                fit_isoflops(rows)
        with self.assertRaises(ScalingFitError):
            fit_isoflops(self.calibration, assumed_loss_floor=10.)

    def test_refuses_architecture_or_data_policy_pooling(self):
        for key in ['architecture', 'data_policy', 'mode']:
            rows = copy.deepcopy(self.calibration); rows[-1][key] = 'changed'
            with self.assertRaisesRegex(ScalingFitError, 'pool'):
                fit_isoflops(rows)

    def test_prediction_freeze_identity_and_budget_isolation(self):
        law = fit_isoflops(self.calibration)
        with self.assertRaisesRegex(ScalingFitError, 'calibration'):
            freeze_prediction(law, 96e6)
        frozen = freeze_prediction(law, 384e6)
        changed = copy.deepcopy(frozen); changed['prediction']['parameters'] += 1
        with self.assertRaisesRegex(ScalingFitError, 'identity'):
            evaluate_holdout(changed, simulated_rows(budgets=(384e6,)))
        with self.assertRaisesRegex(ScalingFitError, 'entire'):
            evaluate_holdout(frozen, self.calibration)
        rows = simulated_rows(budgets=(384e6,)); rows[-1]['data_policy'] = 'repeated'
        with self.assertRaisesRegex(ScalingFitError, 'differs'):
            evaluate_holdout(frozen, rows)

    def test_bootstrap_reproducible_and_retains_failed_fits(self):
        noisy = simulated_rows(budgets=(6e6, 24e6, 96e6), noise=.2)
        first = bootstrap_predictions(noisy, 384e6, draws=20)
        self.assertEqual(first, bootstrap_predictions(noisy, 384e6, draws=20))
        self.assertEqual(len(first['predictions'])+len(first['failed_fits']), 20)
        self.assertTrue(first['failed_fits'])


if __name__ == '__main__':
    unittest.main()
