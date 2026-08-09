"""Gauge-aware parameterizations for the multipartite quotient-real VQC.

The quotient construction gives every subsystem a two-dimensional real flag.
On the canonical quotient subspace, the complex-structure generator acting on
any local flag has the same action. Consequently, local flag rotations
R(alpha_q)=exp(alpha_q J) collapse to one total complex phase
exp(i sum_q alpha_q). Parameters that only redistribute that phase between
local flags are redundant in the quotient representation; for ordinary
phase-insensitive observables, the remaining total phase is also unobservable.

This module makes those redundant directions explicit so they can be measured
and then pruned rather than being mistaken for trainable QML capacity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quotient_real import (
    J_FLAG,
    QuotientRealVQC,
    compress_quotient_state,
    quotient_probability_one,
)


def flag_rotation(angle: float) -> np.ndarray:
    """Real rotation exp(angle * J_FLAG) representing multiplication by exp(i angle)."""
    c, s = np.cos(angle), np.sin(angle)
    return c * np.eye(2, dtype=np.float64) + s * J_FLAG


def _apply_axis(tensor: np.ndarray, matrix: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(tensor, axis, 0)
    moved = np.tensordot(matrix, moved, axes=([1], [0]))
    return np.moveaxis(moved, 0, axis)


def apply_local_flag_rotation(
    state: np.ndarray, angle: float, subsystem: int, n_subsystems: int
) -> np.ndarray:
    """Rotate one local quotient flag while leaving physical axes untouched."""
    if not 0 <= subsystem < n_subsystems:
        raise ValueError("invalid subsystem")
    tensor = np.asarray(state, dtype=np.float64).reshape((2,) * (2 * n_subsystems))
    tensor = _apply_axis(tensor, flag_rotation(float(angle)), n_subsystems + subsystem)
    return tensor.reshape(-1)


def apply_flag_gauge(
    state: np.ndarray, gauge_angles: np.ndarray, n_subsystems: int
) -> np.ndarray:
    """Apply a sequence of local flag rotations.

    Gauge angles may contain any multiple of n_subsystems entries. Entries are
    assigned cyclically to local flags, which is convenient for one gauge angle
    per subsystem per variational layer.
    """
    angles = np.asarray(gauge_angles, dtype=np.float64).reshape(-1)
    out = np.asarray(state, dtype=np.float64).reshape(-1).copy()
    for j, angle in enumerate(angles):
        out = apply_local_flag_rotation(out, float(angle), j % n_subsystems, n_subsystems)
    return out


def quotient_state_to_complex(state: np.ndarray, n_subsystems: int) -> np.ndarray:
    """Recover complex coordinates from a canonical quotient representative."""
    coords = compress_quotient_state(state, n_subsystems)
    dim = 1 << n_subsystems
    return coords[:dim] + 1j * coords[dim:]


def finite_difference_state_jacobian(
    model: "GaugeAugmentedQuotientVQC",
    angles: np.ndarray,
    parameters: np.ndarray,
    indices: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Complex state Jacobian for selected parameters using central differences."""
    theta = np.asarray(parameters, dtype=np.float64)
    rows = []
    for j in np.asarray(indices, dtype=int).reshape(-1):
        plus = theta.copy(); plus[j] += eps
        minus = theta.copy(); minus[j] -= eps
        psi_plus = quotient_state_to_complex(model.state(angles, plus), model.n_qubits)
        psi_minus = quotient_state_to_complex(model.state(angles, minus), model.n_qubits)
        rows.append((psi_plus - psi_minus) / (2.0 * eps))
    return np.asarray(rows, dtype=np.complex128)


def pure_state_qfim(state: np.ndarray, derivatives: np.ndarray) -> np.ndarray:
    """Pure-state quantum Fisher information matrix for supplied state derivatives."""
    psi = np.asarray(state, dtype=np.complex128).reshape(-1)
    derivs = np.asarray(derivatives, dtype=np.complex128)
    qfim = np.empty((derivs.shape[0], derivs.shape[0]), dtype=np.float64)
    for i, di in enumerate(derivs):
        for j, dj in enumerate(derivs):
            qfim[i, j] = 4.0 * np.real(
                np.vdot(di, dj) - np.vdot(di, psi) * np.vdot(psi, dj)
            )
    return qfim


def numerical_rank(matrix: np.ndarray, *, atol: float = 1e-8) -> int:
    """Numerical matrix rank using an absolute singular-value threshold."""
    singular = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    return int(np.sum(singular > atol))


@dataclass
class GaugeAugmentedQuotientVQC:
    """Quotient-real VQC augmented with deliberately redundant local flag phases.

    The first 2*n_qubits*n_layers parameters are the physical RY/RZ ansatz
    parameters used by QuotientRealVQC. The final n_qubits*n_layers parameters
    are one local flag phase per subsystem per layer. They expose the quotient
    gauge directions explicitly; they do not add observable model capacity.
    """

    n_qubits: int
    n_layers: int = 1

    def __post_init__(self) -> None:
        self.base = QuotientRealVQC(self.n_qubits, self.n_layers)

    @property
    def n_physical_parameters(self) -> int:
        return self.base.n_parameters

    @property
    def n_gauge_parameters(self) -> int:
        return self.n_qubits * self.n_layers

    @property
    def n_parameters(self) -> int:
        return self.n_physical_parameters + self.n_gauge_parameters

    @property
    def gauge_parameter_indices(self) -> np.ndarray:
        return np.arange(self.n_physical_parameters, self.n_parameters, dtype=int)

    @property
    def physical_parameter_indices(self) -> np.ndarray:
        return np.arange(self.n_physical_parameters, dtype=int)

    def initial_parameters(self, seed: int = 0, scale: float = 0.2) -> np.ndarray:
        rng = np.random.default_rng(seed)
        physical = rng.normal(0.0, scale, size=self.n_physical_parameters)
        gauge = rng.normal(0.0, scale, size=self.n_gauge_parameters)
        return np.concatenate([physical, gauge])

    def split_parameters(self, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.asarray(parameters, dtype=np.float64).reshape(-1)
        if theta.size != self.n_parameters:
            raise ValueError(f"parameters must have size {self.n_parameters}")
        return theta[: self.n_physical_parameters], theta[self.n_physical_parameters :]

    def state(self, angles: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        physical, gauge = self.split_parameters(parameters)
        state = self.base.state(angles, physical)
        return apply_flag_gauge(state, gauge, self.n_qubits)

    def probability(self, angles: np.ndarray, parameters: np.ndarray) -> float:
        return quotient_probability_one(self.state(angles, parameters), self.n_qubits)

    def probabilities(self, angles: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        x = np.asarray(angles, dtype=np.float64)
        if x.ndim == 1:
            return np.array([self.probability(x, parameters)])
        if x.ndim != 2 or x.shape[1] != self.n_qubits:
            raise ValueError(f"angles must have shape (samples, {self.n_qubits})")
        return np.array([self.probability(row, parameters) for row in x], dtype=np.float64)

    def parameter_shift_probability_jacobian(
        self, angles: np.ndarray, parameters: np.ndarray
    ) -> np.ndarray:
        x = np.atleast_2d(np.asarray(angles, dtype=np.float64))
        theta = np.asarray(parameters, dtype=np.float64)
        jac = np.empty((x.shape[0], theta.size), dtype=np.float64)
        shift = np.pi / 2.0
        for j in range(theta.size):
            plus = theta.copy(); plus[j] += shift
            minus = theta.copy(); minus[j] -= shift
            jac[:, j] = 0.5 * (
                self.probabilities(x, plus) - self.probabilities(x, minus)
            )
        return jac

    def gauge_fixed_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """Drop all unobservable local flag phases, retaining only physical parameters."""
        physical, _ = self.split_parameters(parameters)
        return physical.copy()
