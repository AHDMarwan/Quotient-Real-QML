import unittest

import numpy as np

from automatic_pruning import (
    MixedGaugeQuotientVQC,
    aggregate_projective_qfim,
    deterministic_orthogonal_matrix,
    ideal_parameter_shift_variance,
    project_parameters_to_active_subspace,
    projective_fidelity,
    projector_distance,
    qfim_active_subspace,
    shifted_circuit_shots_per_estimate,
)


class TestAutomaticQFIMPruning(unittest.TestCase):
    def test_mixing_matrix_is_orthogonal(self):
        matrix = deterministic_orthogonal_matrix(11, seed=17)
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(11), atol=1e-12)

    def test_full_qfim_recovers_hidden_quotient_nullspace(self):
        rng = np.random.default_rng(123)
        model = MixedGaugeQuotientVQC(3, 1, mixing_seed=29)
        theta = model.initial_parameters(seed=31)
        x = rng.normal(size=(6, 3))

        qfim = aggregate_projective_qfim(model, x, theta)
        values, active, null = qfim_active_subspace(qfim, rtol=1e-7, atol=1e-8)

        self.assertEqual(active.shape[1], model.n_physical_parameters)
        self.assertEqual(null.shape[1], model.n_gauge_parameters)
        self.assertGreater(float(np.min(values)), 1e-4)
        self.assertLess(projector_distance(active, model.true_active_basis), 2e-6)
        self.assertLess(projector_distance(null, model.true_null_basis), 2e-6)

        # Redundancy is deliberately not aligned with individual exposed coordinates.
        self.assertGreater(float(np.min(np.diag(qfim))), 1e-3)

    def test_automatic_projection_preserves_observables_and_projective_state(self):
        rng = np.random.default_rng(77)
        model = MixedGaugeQuotientVQC(3, 1, mixing_seed=13)
        theta = model.initial_parameters(seed=9)
        geometry_inputs = rng.normal(size=(6, 3))
        qfim = aggregate_projective_qfim(model, geometry_inputs, theta)
        _, active, _ = qfim_active_subspace(qfim)
        pruned = project_parameters_to_active_subspace(theta, active)

        held_out = rng.normal(size=(8, 3))
        np.testing.assert_allclose(
            model.probabilities(held_out, theta),
            model.probabilities(held_out, pruned),
            atol=3e-8,
            rtol=3e-8,
        )
        for row in held_out[:3]:
            fidelity = projective_fidelity(
                model.complex_state(row, theta), model.complex_state(row, pruned)
            )
            self.assertGreater(fidelity, 1.0 - 2e-9)

    def test_fixed_shift_budget_reduces_gradient_variance_after_pruning(self):
        total_shots = 24000
        augmented_parameters = 9
        physical_parameters = 6
        s_aug = shifted_circuit_shots_per_estimate(total_shots, augmented_parameters)
        s_fixed = shifted_circuit_shots_per_estimate(total_shots, physical_parameters)

        variance_aug = ideal_parameter_shift_variance(0.72, 0.31, s_aug)
        variance_fixed = ideal_parameter_shift_variance(0.72, 0.31, s_fixed)

        self.assertGreater(s_fixed, s_aug)
        self.assertAlmostEqual(
            variance_aug / variance_fixed, s_fixed / s_aug, places=12
        )
        self.assertGreater(variance_aug / variance_fixed, 1.49)


if __name__ == "__main__":
    unittest.main()
