r"""Explicit quotient-space real quantum mechanics for small multipartite VQCs.

This module follows Barrios Hita et al., Phys. Rev. Lett. 136, 240202 (2026),
arXiv:2503.17307v2. A complex N-partite state is represented by its canonical
real quotient representative

    R(psi) = Re(psi) \otimes |even_N> + Im(psi) \otimes |odd_N>,

where |even_N> and |odd_N> span the orthogonal complement of the kernel of the
tensored inverse single-system realification map (Supplement Eq. S38).

The explicit representation stores all N local two-dimensional flags. It is
therefore intended as a correctness/research implementation, not as an
optimized simulator. The quotient itself has only two physical flag classes;
compressing to that two-dimensional quotient basis recovers the usual global
realification coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

J_FLAG = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
I_FLAG = np.eye(2, dtype=np.float64)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=np.complex128,
    )


def quotient_flag_basis(n_subsystems: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized canonical |even_N>, |odd_N> flag representatives (S37)."""
    if n_subsystems < 1:
        raise ValueError("n_subsystems must be >= 1")
    dim = 1 << n_subsystems
    even = np.zeros(dim, dtype=np.float64)
    odd = np.zeros(dim, dtype=np.float64)
    norm = np.sqrt(2.0 ** (n_subsystems - 1))
    for idx in range(dim):
        weight = idx.bit_count()
        if weight % 2 == 0:
            even[idx] = (-1.0) ** (weight // 2) / norm
        else:
            odd[idx] = (-1.0) ** ((weight - 1) // 2) / norm
    return even, odd


def quotient_flag_projector(n_subsystems: int) -> np.ndarray:
    """Projector onto ker[(S^-1)^tensor N]^perp in the local-flag tensor space (S36)."""
    even, odd = quotient_flag_basis(n_subsystems)
    return np.outer(even, even) + np.outer(odd, odd)


def canonical_quotient_state(psi: np.ndarray, n_subsystems: int) -> np.ndarray:
    """Map a complex multipartite state to its explicit normalized canonical quotient representative."""
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    physical_dim = 1 << n_subsystems
    if psi.size != physical_dim:
        raise ValueError(f"psi must have size {physical_dim}, got {psi.size}")
    even, odd = quotient_flag_basis(n_subsystems)
    return np.kron(psi.real, even) + np.kron(psi.imag, odd)


def compress_quotient_state(state: np.ndarray, n_subsystems: int) -> np.ndarray:
    """Coordinates of a canonical explicit state in the {|even_N>, |odd_N>} quotient basis."""
    physical_dim = 1 << n_subsystems
    flag_dim = 1 << n_subsystems
    state = np.asarray(state, dtype=np.float64).reshape(physical_dim, flag_dim)
    even, odd = quotient_flag_basis(n_subsystems)
    return np.concatenate([state @ even, state @ odd])


def expand_quotient_coordinates(coords: np.ndarray, n_subsystems: int) -> np.ndarray:
    """Expand [Re psi, Im psi] quotient coordinates into the explicit canonical representative."""
    physical_dim = 1 << n_subsystems
    coords = np.asarray(coords, dtype=np.float64).reshape(-1)
    if coords.size != 2 * physical_dim:
        raise ValueError(f"coords must have size {2 * physical_dim}")
    even, odd = quotient_flag_basis(n_subsystems)
    return np.kron(coords[:physical_dim], even) + np.kron(coords[physical_dim:], odd)


def local_realification(psi: np.ndarray) -> np.ndarray:
    """Single-system S map as a (physical, flag) array."""
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    out = np.zeros((psi.size, 2), dtype=np.float64)
    out[:, 0] = psi.real
    out[:, 1] = psi.imag
    return out


def raw_product_realification(local_states: list[np.ndarray]) -> np.ndarray:
    """Tensor local S maps without quotienting, returned with physical axes before flag axes."""
    if not local_states:
        raise ValueError("local_states must be non-empty")
    n = len(local_states)
    tensor = local_realification(local_states[0])
    for psi in local_states[1:]:
        tensor = np.tensordot(tensor, local_realification(psi), axes=0)
    perm = list(range(0, 2 * n, 2)) + list(range(1, 2 * n, 2))
    return np.transpose(tensor, perm).reshape(-1)


def project_raw_product_to_quotient(raw_state: np.ndarray, local_dims: list[int]) -> np.ndarray:
    """Project a raw tensor product of local realifications to the canonical quotient subspace and normalize."""
    n = len(local_dims)
    if n < 1:
        raise ValueError("local_dims must be non-empty")
    physical_dim = int(np.prod(local_dims))
    flag_dim = 1 << n
    raw = np.asarray(raw_state, dtype=np.float64).reshape(physical_dim, flag_dim)
    projected = raw @ quotient_flag_projector(n)
    flat = projected.reshape(-1)
    norm = np.linalg.norm(flat)
    if norm == 0:
        raise ValueError("projection has zero norm")
    return flat / norm


def _apply_axis(tensor: np.ndarray, matrix: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(tensor, axis, 0)
    moved = np.tensordot(matrix, moved, axes=([1], [0]))
    return np.moveaxis(moved, 0, axis)


def apply_local_complex_operator(
    state: np.ndarray, operator: np.ndarray, subsystem: int, n_subsystems: int
) -> np.ndarray:
    """Apply T_q(A)=Re(A) on physical q + Im(A) coupled to local J_flag,q."""
    if not 0 <= subsystem < n_subsystems:
        raise ValueError("invalid subsystem")
    op = np.asarray(operator, dtype=np.complex128)
    if op.shape != (2, 2):
        raise ValueError("this prototype currently supports qubit-local 2x2 operators")
    tensor = np.asarray(state, dtype=np.float64).reshape((2,) * (2 * n_subsystems))
    real_term = _apply_axis(tensor, op.real, subsystem)
    imag_term = _apply_axis(tensor, op.imag, subsystem)
    imag_term = _apply_axis(imag_term, J_FLAG, n_subsystems + subsystem)
    return (real_term + imag_term).reshape(-1)


def apply_real_cnot(state: np.ndarray, control: int, target: int, n_subsystems: int) -> np.ndarray:
    """Apply a real CNOT to physical subsystems; local flags are untouched."""
    if control == target:
        raise ValueError("control and target must differ")
    physical_dim = 1 << n_subsystems
    flag_dim = 1 << n_subsystems
    matrix = np.asarray(state, dtype=np.float64).reshape(physical_dim, flag_dim)
    out = np.empty_like(matrix)
    for idx in range(physical_dim):
        j = idx
        if (idx >> (n_subsystems - 1 - control)) & 1:
            j ^= 1 << (n_subsystems - 1 - target)
        out[j, :] = matrix[idx, :]
    return out.reshape(-1)


def quotient_probability_one(state: np.ndarray, n_subsystems: int, measured_qubit: int = 0) -> float:
    physical_dim = 1 << n_subsystems
    flag_dim = 1 << n_subsystems
    matrix = np.asarray(state, dtype=np.float64).reshape(physical_dim, flag_dim)
    mask = 1 << (n_subsystems - 1 - measured_qubit)
    select = (np.arange(physical_dim) & mask) != 0
    return float(np.sum(matrix[select, :] ** 2))


@dataclass
class QuotientRealVQC:
    """Small explicit canonical-representative VQC using N local quotient flags."""

    n_qubits: int
    n_layers: int = 1

    def __post_init__(self) -> None:
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")

    @property
    def n_parameters(self) -> int:
        return 2 * self.n_qubits * self.n_layers

    @property
    def explicit_state_size(self) -> int:
        return 1 << (2 * self.n_qubits)

    @property
    def quotient_dimension(self) -> int:
        return 2 * (1 << self.n_qubits)

    def initial_parameters(self, seed: int = 0, scale: float = 0.2) -> np.ndarray:
        return np.random.default_rng(seed).normal(0.0, scale, size=self.n_parameters)

    def _zero_state(self) -> np.ndarray:
        psi = np.zeros(1 << self.n_qubits, dtype=np.complex128)
        psi[0] = 1.0
        return canonical_quotient_state(psi, self.n_qubits)

    def state(self, angles: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        angles = np.asarray(angles, dtype=np.float64)
        parameters = np.asarray(parameters, dtype=np.float64)
        if angles.shape != (self.n_qubits,):
            raise ValueError(f"angles must have shape ({self.n_qubits},), got {angles.shape}")
        if parameters.shape != (self.n_parameters,):
            raise ValueError(f"parameters must have shape ({self.n_parameters},), got {parameters.shape}")

        state = self._zero_state()
        for q, angle in enumerate(angles):
            state = apply_local_complex_operator(state, ry(float(angle)), q, self.n_qubits)

        k = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                state = apply_local_complex_operator(state, ry(float(parameters[k])), q, self.n_qubits)
                k += 1
                state = apply_local_complex_operator(state, rz(float(parameters[k])), q, self.n_qubits)
                k += 1
            for q in range(self.n_qubits - 1):
                state = apply_real_cnot(state, q, q + 1, self.n_qubits)
            if self.n_qubits > 2:
                state = apply_real_cnot(state, self.n_qubits - 1, 0, self.n_qubits)
        return state

    def probability(self, angles: np.ndarray, parameters: np.ndarray) -> float:
        return quotient_probability_one(self.state(angles, parameters), self.n_qubits)

    def probabilities(self, angles: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        x = np.asarray(angles, dtype=np.float64)
        if x.ndim == 1:
            return np.array([self.probability(x, parameters)])
        if x.ndim != 2 or x.shape[1] != self.n_qubits:
            raise ValueError(f"angles must have shape (samples, {self.n_qubits})")
        return np.array([self.probability(row, parameters) for row in x], dtype=np.float64)

    def parameter_shift_probability_jacobian(self, angles: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(angles, dtype=np.float64))
        theta = np.asarray(parameters, dtype=np.float64)
        jac = np.empty((x.shape[0], theta.size), dtype=np.float64)
        shift = np.pi / 2.0
        for j in range(theta.size):
            plus = theta.copy()
            plus[j] += shift
            minus = theta.copy()
            minus[j] -= shift
            jac[:, j] = 0.5 * (self.probabilities(x, plus) - self.probabilities(x, minus))
        return jac

    def fit(
        self,
        angles: np.ndarray,
        labels: np.ndarray,
        *,
        parameters: np.ndarray | None = None,
        epochs: int = 5,
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
            theta -= learning_rate * (jac.T @ dloss_dp)
        return theta, history
