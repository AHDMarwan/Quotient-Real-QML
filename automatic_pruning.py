"""Automatic quotient-aware redundancy detection from projective quantum geometry.

This module deliberately hides the quotient gauge directions by applying a fixed
orthogonal mixing to physical and local-flag coordinates.  In the exposed
parameterization, no coordinate needs to be individually redundant.  The
aggregated pure-state QFIM is then used to recover the active physical subspace
and its null (gauge) complement automatically.

The construction is useful as a controlled research test: the true physical and
gauge subspaces are known internally, so QFIM-based recovery can be validated
quantitatively before applying the same machinery to less structured ansatzes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quotient_gauge import (
    GaugeAugmentedQuotientVQC,
    pure_state_qfim,
    quotient_state_to_complex,
)


def deterministic_orthogonal_matrix(dimension: int, seed: int = 0) -> np.ndarray:
    """Return a deterministic Haar-like orthogonal matrix from a seeded QR factorization."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.where(np.diag(r) >= 0.0, 1.0, -1.0)
    return q @ np.diag(signs)


@dataclass
class MixedGaugeQuotientVQC:
    """Gauge-augmented quotient VQC with dense exposed parameter coordinates.

    Let z = M w, where M is orthogonal.  The latent vector z consists of the
    ordinary physical VQC parameters followed by quotient-local flag phases.
    The exposed vector w therefore mixes physical and gauge directions densely.
    Observable redundancy is no longer coordinate-aligned and must be recovered
    as a subspace.
    """

    n_qubits: int
    n_layers: int = 1
    mixing_seed: int = 0

    def __post_init__(self) -> None:
        self.latent_model = GaugeAugmentedQuotientVQC(self.n_qubits, self.n_layers)
        self.mixing_matrix = deterministic_orthogonal_matrix(
            self.latent_model.n_parameters, self.mixing_seed
        )

    @property
    def n_parameters(self) -> int:
        return self.latent_model.n_parameters

    @property
    def n_physical_parameters(self) -> int:
        return self.latent_model.n_physical_parameters

    @property
    def n_gauge_parameters(self) -> int:
        return self.latent_model.n_gauge_parameters

    @property
    def true_active_basis(self) -> np.ndarray:
        """Exact exposed-coordinate basis that changes physical latent parameters."""
        return self.mixing_matrix.T[:, : self.n_physical_parameters]

    @property
    def true_null_basis(self) -> np.ndarray:
        """Exact exposed-coordinate quotient gauge nullspace."""
        return self.mixing_matrix.T[:, self.n_physical_parameters :]

    def latent_parameters(self, parameters: np.ndarray) -> np.ndarray:
        theta = np.asarray(parameters, dtype=np.float64).reshape(-1)
        if theta.size != self.n_parameters:
            raise ValueError(f"parameters must have size {self.n_parameters}")
        return self.mixing_matrix @ theta

    def exposed_parameters(self, latent_parameters: np.ndarray) -> np.ndarray:
        latent = np.asarray(latent_parameters, dtype=np.float64).reshape(-1)
        if latent.size != self.n_parameters:
            raise ValueError(f"latent_parameters must have size {self.n_parameters}")
        return self.mixing_matrix.T @ latent

    def initial_parameters(self, seed: int = 0, scale: float = 0.2) -> np.ndarray:
        latent = self.latent_model.initial_parameters(seed=seed, scale=scale)
        return self.exposed_parameters(latent)

    def state(self, angles: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        return self.latent_model.state(angles, self.latent_parameters(parameters))

    def complex_state(self, angles: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        return quotient_state_to_complex(
            self.state(angles, parameters), self.n_qubits
        )

    def probability(self, angles: np.ndarray, parameters: np.ndarray) -> float:
        return self.latent_model.probability(
            angles, self.latent_parameters(parameters)
        )

    def probabilities(self, angles: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        return self.latent_model.probabilities(
            angles, self.latent_parameters(parameters)
        )


def finite_difference_exposed_state_jacobian(
    model: MixedGaugeQuotientVQC,
    angles: np.ndarray,
    parameters: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Complex-state Jacobian with respect to all exposed mixed coordinates."""
    theta = np.asarray(parameters, dtype=np.float64).reshape(-1)
    derivatives = []
    for j in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += eps
        minus[j] -= eps
        derivatives.append(
            (model.complex_state(angles, plus) - model.complex_state(angles, minus))
            / (2.0 * eps)
        )
    return np.asarray(derivatives, dtype=np.complex128)


def aggregate_projective_qfim(
    model: MixedGaugeQuotientVQC,
    angles: np.ndarray,
    parameters: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Average the pure-state QFIM over several encoded inputs.

    Aggregating multiple inputs avoids mistaking an input-specific inactive
    direction for a structural parameter redundancy.
    """
    x = np.atleast_2d(np.asarray(angles, dtype=np.float64))
    theta = np.asarray(parameters, dtype=np.float64).reshape(-1)
    qfim = np.zeros((theta.size, theta.size), dtype=np.float64)
    for row in x:
        psi = model.complex_state(row, theta)
        derivatives = finite_difference_exposed_state_jacobian(
            model, row, theta, eps=eps
        )
        qfim += pure_state_qfim(psi, derivatives)
    qfim /= x.shape[0]
    return 0.5 * (qfim + qfim.T)


def qfim_active_subspace(
    qfim: np.ndarray,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return active eigenvalues, active basis, and null basis of a QFIM."""
    matrix = 0.5 * (
        np.asarray(qfim, dtype=np.float64) + np.asarray(qfim, dtype=np.float64).T
    )
    values, vectors = np.linalg.eigh(matrix)
    max_value = max(float(np.max(values)), 0.0)
    threshold = max(atol, rtol * max_value)
    active = values > threshold
    return values[active], vectors[:, active], vectors[:, ~active]


def project_parameters_to_active_subspace(
    parameters: np.ndarray, active_basis: np.ndarray
) -> np.ndarray:
    """Orthogonally remove the QFIM-null component of a parameter vector."""
    theta = np.asarray(parameters, dtype=np.float64).reshape(-1)
    basis = np.asarray(active_basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != theta.size:
        raise ValueError("active_basis has incompatible shape")
    return basis @ (basis.T @ theta)


def projector_distance(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Spectral distance between two orthogonal-subspace projectors."""
    a = np.asarray(basis_a, dtype=np.float64)
    b = np.asarray(basis_b, dtype=np.float64)
    pa = a @ a.T
    pb = b @ b.T
    return float(np.linalg.norm(pa - pb, ord=2))


def projective_fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Pure-state fidelity, insensitive to the global phase removed by the quotient."""
    a = np.asarray(state_a, dtype=np.complex128).reshape(-1)
    b = np.asarray(state_b, dtype=np.complex128).reshape(-1)
    return float(np.abs(np.vdot(a, b)) ** 2)


def shifted_circuit_shots_per_estimate(
    total_shift_shots: int, n_parameters: int
) -> int:
    """Shots per +/- parameter-shift circuit under a fixed total shift budget."""
    if total_shift_shots < 1 or n_parameters < 1:
        raise ValueError("shot budget and parameter count must be positive")
    return max(1, total_shift_shots // (2 * n_parameters))


def ideal_parameter_shift_variance(
    p_plus: float, p_minus: float, shots_per_circuit: int
) -> float:
    """Bernoulli-shot variance of 0.5*(p_plus_hat-p_minus_hat)."""
    s = int(shots_per_circuit)
    if s < 1:
        raise ValueError("shots_per_circuit must be positive")
    return float(
        0.25
        * (p_plus * (1.0 - p_plus) + p_minus * (1.0 - p_minus))
        / s
    )
