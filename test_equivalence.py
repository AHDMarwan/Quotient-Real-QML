import unittest
import numpy as np

from qreal_vqc import QRealVQC


class TestRealification(unittest.TestCase):
    def test_probabilities_match_complex_backend(self):
        rng = np.random.default_rng(7)
        for n_qubits in (1, 2, 3, 4):
            complex_model = QRealVQC(n_qubits, n_layers=2, backend="complex")
            real_model = QRealVQC(n_qubits, n_layers=2, backend="structured_real")
            theta = rng.normal(size=complex_model.n_parameters)
            x = rng.normal(size=(8, n_qubits))
            np.testing.assert_allclose(
                complex_model.probabilities(x, theta),
                real_model.probabilities(x, theta),
                atol=1e-12,
                rtol=1e-12,
            )

    def test_norms_match(self):
        rng = np.random.default_rng(11)
        model_c = QRealVQC(3, n_layers=2, backend="complex")
        model_r = QRealVQC(3, n_layers=2, backend="structured_real")
        theta = rng.normal(size=model_c.n_parameters)
        x = rng.normal(size=3)
        psi = model_c.state(x, theta)
        phi = model_r.state(x, theta)
        dim = psi.size
        self.assertAlmostEqual(float(np.vdot(psi, psi).real), 1.0, places=12)
        self.assertAlmostEqual(float(np.dot(phi, phi)), 1.0, places=12)
        np.testing.assert_allclose(phi[:dim], psi.real, atol=1e-12)
        np.testing.assert_allclose(phi[dim:], psi.imag, atol=1e-12)

    def test_single_sample_api(self):
        model = QRealVQC(3, backend="complex")
        theta = model.initial_parameters(1)
        out = model.probabilities(np.zeros(3), theta)
        self.assertEqual(out.shape, (1,))

    def test_parameter_shift_jacobian_matches_finite_difference(self):
        rng = np.random.default_rng(3)
        model = QRealVQC(2, n_layers=1, backend="complex")
        theta = rng.normal(size=model.n_parameters)
        x = rng.normal(size=(3, 2))
        jac = model.parameter_shift_probability_jacobian(x, theta)
        eps = 1e-6
        fd = np.empty_like(jac)
        for j in range(theta.size):
            plus = theta.copy(); plus[j] += eps
            minus = theta.copy(); minus[j] -= eps
            fd[:, j] = (model.probabilities(x, plus) - model.probabilities(x, minus)) / (2 * eps)
        np.testing.assert_allclose(jac, fd, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
