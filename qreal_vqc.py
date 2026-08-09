"""Small state-vector VQC simulator for complex and realified quantum models.

The structured_real backend stores a complex state psi=x+i y as the real vector
[x, y] and applies each gate U=A+iB through

    [A -B] [x]
    [B  A] [y]

without ever storing complex amplitudes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Backend = Literal["complex", "structured_real", "naive_real"]


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=np.complex128,
    )


def _apply_single_complex(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    tensor = state.reshape((2,) * n_qubits)
    tensor = np.moveaxis(tensor, qubit, 0)
    tensor = np.tensordot(gate, tensor, axes=([1], [0]))
    tensor = np.moveaxis(tensor, 0, qubit)
    return tensor.reshape(-1)


def _apply_single_realified(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    dim = 1 << n_qubits
    x, y = state[:dim], state[dim:]
    a, b = np.asarray(gate.real, dtype=np.float64), np.asarray(gate.imag, dtype=np.float64)
    ax = _apply_single_complex(x.astype(np.complex128), a, qubit, n_qubits).real
    by = _apply_single_complex(y.astype(np.complex128), b, qubit, n_qubits).real
    bx = _apply_single_complex(x.astype(np.complex128), b, qubit, n_qubits).real
    ay = _apply_single_complex(y.astype(np.complex128), a, qubit, n_qubits).real
    return np.concatenate([ax - by, bx + ay])


def _apply_cnot_vector(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    out = np.empty_like(state)
    for idx in range(state.size):
        bits = [(idx >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        j = idx
        if bits[control]:
            j ^= 1 << (n_qubits - 1 - target)
        out[j] = state[idx]
    return out


def _apply_cnot(state: np.ndarray, control: int, target: int, n_qubits: int, backend: Backend) -> np.ndarray:
    if backend == "structured_real":
        dim = 1 << n_qubits
        return np.concatenate(
            [
                _apply_cnot_vector(state[:dim], control, target, n_qubits),
                _apply_cnot_vector(state[dim:], control, target, n_qubits),
            ]
        )
    return _apply_cnot_vector(state, control, target, n_qubits)


def _p_one(state: np.ndarray, n_qubits: int, backend: Backend, measured_qubit: int = 0) -> float:
    dim = 1 << n_qubits
    mask = 1 << (n_qubits - 1 - measured_qubit)
    indices = np.arange(dim)
    select = (indices & mask) != 0
    if backend == "structured_real":
        x, y = state[:dim], state[dim:]
        return float(np.sum(x[select] ** 2 + y[select] ** 2))
    return float(np.sum(np.abs(state[select]) ** 2))


@dataclass
class QRealVQC:
    n_qubits: int
    n_layers: int = 1
    backend: Backend = "complex"

    def __post_init__(self) -> None:
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.backend not in {"complex", "structured_real", "naive_real"}:
            raise ValueError(f"unknown backend: {self.backend}")

    @property
    def n_parameters(self) -> int:
        return 2 * self.n_qubits * self.n_layers

    def initial_parameters(self, seed: int = 0, scale: float = 0.2) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, scale, size=self.n_parameters)

    def _zero_state(self) -> np.ndarray:
        dim = 1 << self.n_qubits
        if self.backend == "structured_real":
            state = np.zeros(2 * dim, dtype=np.float64)
        else:
            dtype = np.float64 if self.backend == "naive_real" else np.complex128
            state = np.zeros(dim, dtype=dtype)
        state[0] = 1.0
        return state

    def _single(self, state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
        if self.backend == "structured_real":
            return _apply_single_realified(state, gate, qubit, self.n_qubits)
        if self.backend == "naive_real":
            return _apply_single_complex(state.astype(np.complex128), np.asarray(gate.real), qubit, self.n_qubits).real
        return _apply_single_complex(state, gate, qubit, self.n_qubits)

    def state(self, angles: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        angles = np.asarray(angles, dtype=np.float64)
        parameters = np.asarray(parameters, dtype=np.float64)
        if angles.shape != (self.n_qubits,):
            raise ValueError(f"angles must have shape ({self.n_qubits},), got {angles.shape}")
        if parameters.shape != (self.n_parameters,):
            raise ValueError(f"parameters must have shape ({self.n_parameters},), got {parameters.shape}")

        state = self._zero_state()
        for q, angle in enumerate(angles):
            state = self._single(state, ry(float(angle)), q)

        k = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                state = self._single(state, ry(float(parameters[k])), q)
                k += 1
                gate = ry(float(parameters[k])) if self.backend == "naive_real" else rz(float(parameters[k]))
                state = self._single(state, gate, q)
                k += 1
            for q in range(self.n_qubits - 1):
                state = _apply_cnot(state, q, q + 1, self.n_qubits, self.backend)
            if self.n_qubits > 2:
                state = _apply_cnot(state, self.n_qubits - 1, 0, self.n_qubits, self.backend)
        return state

    def probability(self, angles: np.ndarray, parameters: np.ndarray) -> float:
        return _p_one(self.state(angles, parameters), self.n_qubits, self.backend)

    def probabilities(self, angles: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        angles = np.asarray(angles, dtype=np.float64)
        if angles.ndim == 1:
            return np.array([self.probability(angles, parameters)])
        if angles.ndim != 2 or angles.shape[1] != self.n_qubits:
            raise ValueError(f"angles must have shape (samples, {self.n_qubits})")
        return np.array([self.probability(row, parameters) for row in angles], dtype=np.float64)

    def parameter_shift_probability_jacobian(self, angles: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(angles, dtype=np.float64))
        theta = np.asarray(parameters, dtype=np.float64)
        jac = np.empty((x.shape[0], theta.size), dtype=np.float64)
        shift = np.pi / 2.0
        for j in range(theta.size):
            plus = theta.copy(); plus[j] += shift
            minus = theta.copy(); minus[j] -= shift
            jac[:, j] = 0.5 * (self.probabilities(x, plus) - self.probabilities(x, minus))
        return jac

    def fit(
        self,
        angles: np.ndarray,
        labels: np.ndarray,
        *,
        parameters: np.ndarray | None = None,
        epochs: int = 10,
        learning_rate: float = 0.1,
        seed: int = 0,
    ) -> tuple[np.ndarray, list[float]]:
        x = np.atleast_2d(np.asarray(angles, dtype=np.float64))
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if x.shape != (y.size, self.n_qubits):
            raise ValueError("angles/labels shapes are inconsistent")
        theta = self.initial_parameters(seed) if parameters is None else np.asarray(parameters, dtype=np.float64).copy()
        history: list[float] = []
        eps = 1e-9
        for _ in range(epochs):
            p = np.clip(self.probabilities(x, theta), eps, 1.0 - eps)
            loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
            history.append(loss)
            jac = self.parameter_shift_probability_jacobian(x, theta)
            dloss_dp = (p - y) / (p * (1.0 - p)) / y.size
            grad = jac.T @ dloss_dp
            theta -= learning_rate * grad
        return theta, history
