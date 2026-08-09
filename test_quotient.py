import unittest

import numpy as np

from qreal_vqc import QRealVQC
from quotient_real import (
    J_FLAG,
    QuotientRealVQC,
    apply_local_complex_operator,
    canonical_quotient_state,
    compress_quotient_state,
    expand_quotient_coordinates,
    project_raw_product_to_quotient,
    quotient_flag_basis,
    quotient_flag_projector,
    raw_product_realification,
    rz,
)


class TestQuotientConstruction(unittest.TestCase):
    def test_flag_basis_is_orthonormal_and_j_is_shared(self):
        for n in range(1, 5):
            even, odd = quotient_flag_basis(n)
            self.assertAlmostEqual(np.dot(even, even), 1.0, places=12)
            self.assertAlmostEqual(np.dot(odd, odd), 1.0, places=12)
            self.assertAlmostEqual(np.dot(even, odd), 0.0, places=12)

            for q in range(n):
                e = even.reshape((2,) * n)
                o = odd.reshape((2,) * n)

                def apply_local_j(v):
                    moved = np.moveaxis(v, q, 0)
                    moved = np.tensordot(J_FLAG, moved, axes=([1], [0]))
                    return np.moveaxis(moved, 0, q).reshape(-1)

                np.testing.assert_allclose(apply_local_j(e), odd, atol=1e-12)
                np.testing.assert_allclose(apply_local_j(o), -even, atol=1e-12)

    def test_compress_expand_is_exact_on_canonical_states(self):
        rng = np.random.default_rng(4)
        for n in (1, 2, 3, 4):
            psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
            psi /= np.linalg.norm(psi)
            explicit = canonical_quotient_state(psi, n)
            coords = compress_quotient_state(explicit, n)
            np.testing.assert_allclose(coords[: 1 << n], psi.real, atol=1e-12)
            np.testing.assert_allclose(coords[1 << n :], psi.imag, atol=1e-12)
            np.testing.assert_allclose(
                expand_quotient_coordinates(coords, n), explicit, atol=1e-12
            )

    def test_phase_redistribution_is_identified_by_quotient(self):
        zero = np.array([1, 0], dtype=np.complex128)
        one = np.array([0, 1], dtype=np.complex128)

        ref = raw_product_realification([zero, one])
        shifted = raw_product_realification([1j * zero, -1j * one])
        self.assertGreater(np.linalg.norm(ref - shifted), 1.0)

        qref = project_raw_product_to_quotient(ref, [2, 2])
        qshift = project_raw_product_to_quotient(shifted, [2, 2])
        np.testing.assert_allclose(qref, qshift, atol=1e-12)

        target = canonical_quotient_state(np.kron(zero, one), 2)
        np.testing.assert_allclose(qref, target, atol=1e-12)

    def test_local_complex_actions_preserve_canonical_subspace(self):
        rng = np.random.default_rng(5)
        n = 3
        psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
        psi /= np.linalg.norm(psi)
        state = canonical_quotient_state(psi, n)

        for q in range(n):
            theta = 0.3 + 0.2 * q
            out = apply_local_complex_operator(state, rz(theta), q, n)
            coords = compress_quotient_state(out, n)

            ref = psi.reshape((2,) * n)
            ref = np.moveaxis(ref, q, 0)
            ref = np.tensordot(rz(theta), ref, axes=([1], [0]))
            ref = np.moveaxis(ref, 0, q).reshape(-1)
            np.testing.assert_allclose(coords[: 1 << n], ref.real, atol=1e-12)
            np.testing.assert_allclose(coords[1 << n :], ref.imag, atol=1e-12)

            projector = quotient_flag_projector(n)
            matrix = out.reshape(1 << n, 1 << n)
            np.testing.assert_allclose(matrix @ projector, matrix, atol=1e-12)

    def test_vqc_matches_complex_probabilities_and_gradients(self):
        rng = np.random.default_rng(8)
        for n in (1, 2, 3, 4):
            complex_model = QRealVQC(n, 2, "complex")
            quotient_model = QuotientRealVQC(n, 2)
            theta = rng.normal(scale=0.4, size=complex_model.n_parameters)
            x = rng.normal(size=(4, n))
            np.testing.assert_allclose(
                quotient_model.probabilities(x, theta),
                complex_model.probabilities(x, theta),
                atol=2e-12,
                rtol=2e-12,
            )

        n = 3
        complex_model = QRealVQC(n, 1, "complex")
        quotient_model = QuotientRealVQC(n, 1)
        theta = rng.normal(size=complex_model.n_parameters)
        x = rng.normal(size=(2, n))
        np.testing.assert_allclose(
            quotient_model.parameter_shift_probability_jacobian(x, theta),
            complex_model.parameter_shift_probability_jacobian(x, theta),
            atol=2e-12,
            rtol=2e-12,
        )


if __name__ == "__main__":
    unittest.main()
