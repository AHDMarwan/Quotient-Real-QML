"""Fair complex and realified VQC simulators using matched specialized kernels."""
from __future__ import annotations

import numpy as np

from realification_ops import (
    cnot_realified,
    cnot_vector,
    ry_complex,
    ry_realified,
    rz_complex,
    rz_realified,
    zero_complex,
    zero_realified,
)


def _validate(angles, parameters, n_qubits: int, n_layers: int):
    angles = np.asarray(angles, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    if angles.shape != (n_qubits,):
        raise ValueError(f"angles must have shape ({n_qubits},)")
    if parameters.shape != (2 * n_qubits * n_layers,):
        raise ValueError("wrong parameter shape")
    return angles, parameters


def simulate_complex(angles, parameters, n_qubits: int, n_layers: int = 1, dtype=np.complex128):
    angles, parameters = _validate(angles, parameters, n_qubits, n_layers)
    state = zero_complex(n_qubits, dtype)
    for q, angle in enumerate(angles):
        state = ry_complex(state, float(angle), q, n_qubits)
    k = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            state = ry_complex(state, float(parameters[k]), q, n_qubits)
            k += 1
            state = rz_complex(state, float(parameters[k]), q, n_qubits)
            k += 1
        for q in range(n_qubits - 1):
            state = cnot_vector(state, q, q + 1, n_qubits)
        if n_qubits > 2:
            state = cnot_vector(state, n_qubits - 1, 0, n_qubits)
    return state


def simulate_realified(angles, parameters, n_qubits: int, n_layers: int = 1, dtype=np.float64):
    angles, parameters = _validate(angles, parameters, n_qubits, n_layers)
    state = zero_realified(n_qubits, dtype)
    for q, angle in enumerate(angles):
        state = ry_realified(state, float(angle), q, n_qubits)
    k = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            state = ry_realified(state, float(parameters[k]), q, n_qubits)
            k += 1
            state = rz_realified(state, float(parameters[k]), q, n_qubits)
            k += 1
        for q in range(n_qubits - 1):
            state = cnot_realified(state, q, q + 1, n_qubits)
        if n_qubits > 2:
            state = cnot_realified(state, n_qubits - 1, 0, n_qubits)
    return state


def as_complex(state, dtype=None):
    x, y = state
    if dtype is None:
        dtype = np.complex64 if x.dtype == np.float32 else np.complex128
    return x.astype(dtype) + np.asarray(1j, dtype=dtype) * y.astype(dtype)


def probability_one_complex(state: np.ndarray, n_qubits: int, measured_qubit: int = 0) -> float:
    tensor = state.reshape((2,) * n_qubits)
    index = [slice(None)] * n_qubits
    index[measured_qubit] = 1
    return float(np.sum(np.abs(tensor[tuple(index)]) ** 2, dtype=np.float64))


def probability_one_realified(state, n_qubits: int, measured_qubit: int = 0) -> float:
    x, y = state
    tx, ty = x.reshape((2,) * n_qubits), y.reshape((2,) * n_qubits)
    index = [slice(None)] * n_qubits
    index[measured_qubit] = 1
    a, b = tx[tuple(index)], ty[tuple(index)]
    return float(np.sum(a * a + b * b, dtype=np.float64))


def state_bytes(state) -> int:
    if isinstance(state, tuple):
        return int(state[0].nbytes + state[1].nbytes)
    return int(state.nbytes)


def probabilities(backend: str, xs, parameters, n_qubits: int, n_layers: int = 1, precision: str = "64"):
    xs = np.atleast_2d(np.asarray(xs, dtype=np.float64))
    out = []
    for row in xs:
        if backend == "complex":
            dtype = np.complex64 if precision == "32" else np.complex128
            s = simulate_complex(row, parameters, n_qubits, n_layers, dtype=dtype)
            out.append(probability_one_complex(s, n_qubits))
        elif backend == "realified":
            dtype = np.float32 if precision == "32" else np.float64
            s = simulate_realified(row, parameters, n_qubits, n_layers, dtype=dtype)
            out.append(probability_one_realified(s, n_qubits))
        else:
            raise ValueError("backend must be complex or realified")
    return np.asarray(out, dtype=np.float64)


def parameter_shift_jacobian(backend: str, xs, parameters, n_qubits: int, n_layers: int = 1, precision: str = "64"):
    theta = np.asarray(parameters, dtype=np.float64)
    jac = np.empty((np.atleast_2d(xs).shape[0], theta.size), dtype=np.float64)
    shift = np.pi / 2.0
    for j in range(theta.size):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += shift
        minus[j] -= shift
        jac[:, j] = 0.5 * (
            probabilities(backend, xs, plus, n_qubits, n_layers, precision)
            - probabilities(backend, xs, minus, n_qubits, n_layers, precision)
        )
    return jac
