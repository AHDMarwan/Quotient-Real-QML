"""General complex-gate realification kernels, including Gauss 3M."""
from __future__ import annotations

import numpy as np

from realification_ops import complex_matrix, real_matrix


def apply_complex(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    return complex_matrix(state, np.asarray(gate, dtype=state.dtype), qubit, n_qubits)


def apply_realified_4m(state, gate: np.ndarray, qubit: int, n_qubits: int):
    x, y = state
    a = np.asarray(np.real(gate), dtype=x.dtype)
    b = np.asarray(np.imag(gate), dtype=x.dtype)
    ax = real_matrix(x, a, qubit, n_qubits)
    by = real_matrix(y, b, qubit, n_qubits)
    bx = real_matrix(x, b, qubit, n_qubits)
    ay = real_matrix(y, a, qubit, n_qubits)
    return ax - by, bx + ay


def apply_realified_3m(state, gate: np.ndarray, qubit: int, n_qubits: int):
    """Compute (A+iB)(x+iy) with three real matrix applications."""
    x, y = state
    a = np.asarray(np.real(gate), dtype=x.dtype)
    b = np.asarray(np.imag(gate), dtype=x.dtype)
    p = real_matrix(x, a, qubit, n_qubits)
    q = real_matrix(y, b, qubit, n_qubits)
    r = real_matrix(x + y, a + b, qubit, n_qubits)
    return p - q, r - p - q


def random_unitary(seed: int, dtype=np.complex128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, r = np.linalg.qr(z)
    phases = np.diag(r)
    phases = phases / np.where(np.abs(phases) == 0, 1.0, np.abs(phases))
    q = q * phases.conj()
    return np.asarray(q, dtype=dtype)
