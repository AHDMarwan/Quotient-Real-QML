from __future__ import annotations

import unittest

import numpy as np

from realification_general import apply_complex, apply_realified_3m, apply_realified_4m, random_unitary
from realification_simulator import (
    as_complex,
    parameter_shift_jacobian,
    probabilities,
    simulate_complex,
    simulate_realified,
    state_bytes,
)


class TestRealificationScaling(unittest.TestCase):
    def test_specialized_realification_matches_complex(self):
        rng = np.random.default_rng(7)
        for n in range(1, 6):
            angles = rng.normal(size=n)
            theta = rng.normal(size=2 * n * 2)
            complex_state = simulate_complex(angles, theta, n, 2)
            real_state = simulate_realified(angles, theta, n, 2)
            np.testing.assert_allclose(as_complex(real_state), complex_state, atol=2e-12, rtol=2e-12)
            self.assertEqual(state_bytes(complex_state), state_bytes(real_state))

    def test_float32_realification_matches_complex64(self):
        rng = np.random.default_rng(8)
        for n in (2, 4, 6):
            angles = rng.normal(size=n)
            theta = rng.normal(size=2 * n)
            c = simulate_complex(angles, theta, n, dtype=np.complex64)
            r = simulate_realified(angles, theta, n, dtype=np.float32)
            np.testing.assert_allclose(as_complex(r, np.complex64), c, atol=3e-6, rtol=3e-6)
            self.assertEqual(state_bytes(c), state_bytes(r))

    def test_three_m_and_four_m_match_complex_gate(self):
        rng = np.random.default_rng(9)
        n = 5
        dim = 1 << n
        psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        psi = np.asarray(psi / np.linalg.norm(psi), dtype=np.complex128)
        state = (psi.real.copy(), psi.imag.copy())
        gate = random_unitary(12)
        target = apply_complex(psi, gate, 2, n)
        r4 = apply_realified_4m(state, gate, 2, n)
        r3 = apply_realified_3m(state, gate, 2, n)
        np.testing.assert_allclose(as_complex(r4), target, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(as_complex(r3), target, atol=2e-12, rtol=2e-12)

    def test_probabilities_and_parameter_shift_match(self):
        rng = np.random.default_rng(10)
        n = 4
        xs = rng.normal(size=(3, n))
        theta = rng.normal(scale=0.2, size=2 * n)
        pc = probabilities("complex", xs, theta, n)
        pr = probabilities("realified", xs, theta, n)
        np.testing.assert_allclose(pr, pc, atol=2e-12, rtol=2e-12)
        jc = parameter_shift_jacobian("complex", xs, theta, n)
        jr = parameter_shift_jacobian("realified", xs, theta, n)
        np.testing.assert_allclose(jr, jc, atol=3e-12, rtol=3e-12)


if __name__ == "__main__":
    unittest.main()
