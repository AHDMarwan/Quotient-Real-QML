import unittest

import numpy as np

from quotient_gauge import (
    GaugeAugmentedQuotientVQC,
    apply_flag_gauge,
    finite_difference_state_jacobian,
    numerical_rank,
    pure_state_qfim,
    quotient_state_to_complex,
)
from quotient_real import QuotientRealVQC, canonical_quotient_state


class TestQuotientGauge(unittest.TestCase):
    def test_local_flag_phases_collapse_to_total_phase(self):
        rng = np.random.default_rng(13)
        for n in (1, 2, 3, 4):
            psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
            psi /= np.linalg.norm(psi)
            gauge = rng.normal(size=2 * n)
            state = canonical_quotient_state(psi, n)
            rotated = apply_flag_gauge(state, gauge, n)
            expected = canonical_quotient_state(np.exp(1j * np.sum(gauge)) * psi, n)
            np.testing.assert_allclose(rotated, expected, atol=2e-12, rtol=2e-12)

    def test_redistributed_gauge_with_same_sum_is_identical(self):
        rng = np.random.default_rng(17)
        n = 4
        psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
        psi /= np.linalg.norm(psi)
        alpha = rng.normal(size=n)
        beta = rng.normal(size=n)
        beta[-1] += np.sum(alpha) - np.sum(beta)
        state = canonical_quotient_state(psi, n)
        left = apply_flag_gauge(state, alpha, n)
        right = apply_flag_gauge(state, beta, n)
        np.testing.assert_allclose(left, right, atol=2e-12, rtol=2e-12)

    def test_gauge_augmented_predictions_equal_gauge_fixed_model(self):
        rng = np.random.default_rng(19)
        for n in (2, 3, 4):
            layers = 2
            augmented = GaugeAugmentedQuotientVQC(n, layers)
            fixed = QuotientRealVQC(n, layers)
            theta = augmented.initial_parameters(seed=100 + n)
            physical, _ = augmented.split_parameters(theta)
            x = rng.normal(size=(5, n))
            np.testing.assert_allclose(
                augmented.probabilities(x, theta),
                fixed.probabilities(x, physical),
                atol=3e-12,
                rtol=3e-12,
            )

    def test_probability_jacobian_has_zero_gauge_columns(self):
        rng = np.random.default_rng(23)
        model = GaugeAugmentedQuotientVQC(3, 2)
        theta = model.initial_parameters(seed=5)
        x = rng.normal(size=(4, 3))
        jac = model.parameter_shift_probability_jacobian(x, theta)
        self.assertLess(float(np.max(np.abs(jac[:, model.gauge_parameter_indices]))), 2e-12)

    def test_gauge_tangent_rank_one_but_projective_qfi_rank_zero(self):
        rng = np.random.default_rng(29)
        model = GaugeAugmentedQuotientVQC(3, 2)
        theta = model.initial_parameters(seed=8)
        x = rng.normal(size=3)
        derivs = finite_difference_state_jacobian(
            model, x, theta, model.gauge_parameter_indices, eps=1e-6
        )
        gram = np.real(derivs @ derivs.conj().T)
        self.assertEqual(numerical_rank(gram, atol=2e-7), 1)

        psi = quotient_state_to_complex(model.state(x, theta), model.n_qubits)
        qfim = pure_state_qfim(psi, derivs)
        self.assertEqual(numerical_rank(qfim, atol=2e-6), 0)
        self.assertLess(float(np.max(np.abs(qfim))), 2e-6)

    def test_pruning_saves_one_third_of_parameter_shift_evaluations(self):
        for n in (2, 3, 5):
            for layers in (1, 2, 3):
                model = GaugeAugmentedQuotientVQC(n, layers)
                augmented = 2 * model.n_parameters
                fixed = 2 * model.n_physical_parameters
                self.assertAlmostEqual(augmented / fixed, 1.5, places=12)
                self.assertAlmostEqual((augmented - fixed) / augmented, 1.0 / 3.0, places=12)


if __name__ == "__main__":
    unittest.main()
