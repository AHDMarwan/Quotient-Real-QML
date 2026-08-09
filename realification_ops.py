"""Specialized state-vector operations for fair complex/realified benchmarks."""
from __future__ import annotations

import numpy as np


def zero_complex(n_qubits: int, dtype=np.complex128) -> np.ndarray:
    state = np.zeros(1 << n_qubits, dtype=dtype)
    state[0] = 1.0
    return state


def zero_realified(n_qubits: int, dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(1 << n_qubits, dtype=dtype)
    y = np.zeros_like(x)
    x[0] = 1.0
    return x, y


def _ry_vector(state: np.ndarray, theta: float, qubit: int, n_qubits: int) -> np.ndarray:
    dtype = np.dtype(state.dtype)
    c = dtype.type(np.cos(theta / 2.0))
    s = dtype.type(np.sin(theta / 2.0))
    tensor = np.moveaxis(state.reshape((2,) * n_qubits), qubit, 0)
    out = np.empty_like(tensor)
    out[0] = c * tensor[0] - s * tensor[1]
    out[1] = s * tensor[0] + c * tensor[1]
    return np.moveaxis(out, 0, qubit).reshape(-1)


def ry_complex(state: np.ndarray, theta: float, qubit: int, n_qubits: int) -> np.ndarray:
    return _ry_vector(state, theta, qubit, n_qubits)


def ry_realified(state, theta: float, qubit: int, n_qubits: int):
    x, y = state
    return _ry_vector(x, theta, qubit, n_qubits), _ry_vector(y, theta, qubit, n_qubits)


def rz_complex(state: np.ndarray, theta: float, qubit: int, n_qubits: int) -> np.ndarray:
    dtype = np.dtype(state.dtype)
    p0 = dtype.type(np.exp(-0.5j * theta))
    p1 = dtype.type(np.exp(0.5j * theta))
    tensor = np.moveaxis(state.reshape((2,) * n_qubits), qubit, 0)
    out = np.empty_like(tensor)
    out[0] = p0 * tensor[0]
    out[1] = p1 * tensor[1]
    return np.moveaxis(out, 0, qubit).reshape(-1)


def rz_realified(state, theta: float, qubit: int, n_qubits: int):
    x, y = state
    dtype = np.dtype(x.dtype)
    c = dtype.type(np.cos(theta / 2.0))
    s = dtype.type(np.sin(theta / 2.0))
    tx = np.moveaxis(x.reshape((2,) * n_qubits), qubit, 0)
    ty = np.moveaxis(y.reshape((2,) * n_qubits), qubit, 0)
    ox, oy = np.empty_like(tx), np.empty_like(ty)
    ox[0] = c * tx[0] + s * ty[0]
    oy[0] = c * ty[0] - s * tx[0]
    ox[1] = c * tx[1] - s * ty[1]
    oy[1] = c * ty[1] + s * tx[1]
    return np.moveaxis(ox, 0, qubit).reshape(-1), np.moveaxis(oy, 0, qubit).reshape(-1)


def cnot_vector(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    source = state.reshape((2,) * n_qubits)
    out = state.copy()
    dest = out.reshape((2,) * n_qubits)
    a = [slice(None)] * n_qubits
    b = [slice(None)] * n_qubits
    a[control] = b[control] = 1
    a[target], b[target] = 0, 1
    dest[tuple(a)] = source[tuple(b)]
    dest[tuple(b)] = source[tuple(a)]
    return out


def cnot_realified(state, control: int, target: int, n_qubits: int):
    x, y = state
    return cnot_vector(x, control, target, n_qubits), cnot_vector(y, control, target, n_qubits)


def real_matrix(state: np.ndarray, matrix: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    tensor = np.moveaxis(state.reshape((2,) * n_qubits), qubit, 0)
    out = np.tensordot(matrix, tensor, axes=([1], [0]))
    return np.moveaxis(out, 0, qubit).reshape(-1)


def complex_matrix(state: np.ndarray, matrix: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    tensor = np.moveaxis(state.reshape((2,) * n_qubits), qubit, 0)
    out = np.tensordot(matrix, tensor, axes=([1], [0]))
    return np.moveaxis(out, 0, qubit).reshape(-1)
