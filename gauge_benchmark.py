from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from quotient_gauge import (
    GaugeAugmentedQuotientVQC,
    finite_difference_state_jacobian,
    numerical_rank,
    pure_state_qfim,
    quotient_state_to_complex,
)
from quotient_real import QuotientRealVQC


def benchmark_configuration(n_qubits: int, n_layers: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    augmented = GaugeAugmentedQuotientVQC(n_qubits, n_layers)
    fixed = QuotientRealVQC(n_qubits, n_layers)
    theta = augmented.initial_parameters(seed=seed)
    physical, gauge = augmented.split_parameters(theta)
    x = rng.normal(size=(4, n_qubits))

    p_aug = augmented.probabilities(x, theta)
    p_fixed = fixed.probabilities(x, physical)

    # Observable gradients: gauge columns should vanish.
    t0 = time.perf_counter()
    jac_aug = augmented.parameter_shift_probability_jacobian(x, theta)
    augmented_jacobian_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    jac_fixed = fixed.parameter_shift_probability_jacobian(x, physical)
    fixed_jacobian_seconds = time.perf_counter() - t0

    gauge_cols = jac_aug[:, augmented.gauge_parameter_indices]
    physical_cols = jac_aug[:, augmented.physical_parameter_indices]

    # State-vector geometry: all local flag phases collapse to one total phase,
    # while projective quantum geometry removes that last global-phase direction.
    state_derivs = finite_difference_state_jacobian(
        augmented,
        x[0],
        theta,
        augmented.gauge_parameter_indices,
        eps=1e-6,
    )
    euclidean_gram = np.real(state_derivs @ state_derivs.conj().T)
    psi = quotient_state_to_complex(augmented.state(x[0], theta), n_qubits)
    qfim = pure_state_qfim(psi, state_derivs)

    p = augmented.n_physical_parameters
    g = augmented.n_gauge_parameters
    total = augmented.n_parameters
    augmented_shift_evals = 2 * total
    fixed_shift_evals = 2 * p

    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "physical_parameters": p,
        "local_gauge_parameters": g,
        "augmented_parameters": total,
        "vector_space_gauge_rank": numerical_rank(euclidean_gram, atol=2e-7),
        "projective_qfi_gauge_rank": numerical_rank(qfim, atol=2e-6),
        "max_projective_qfi_entry": float(np.max(np.abs(qfim))),
        "max_prediction_error_after_pruning": float(np.max(np.abs(p_aug - p_fixed))),
        "max_gauge_probability_gradient": float(np.max(np.abs(gauge_cols))),
        "max_physical_jacobian_error": float(np.max(np.abs(physical_cols - jac_fixed))),
        "augmented_parameter_shift_evaluations_per_sample": augmented_shift_evals,
        "gauge_fixed_parameter_shift_evaluations_per_sample": fixed_shift_evals,
        "evaluation_ratio_augmented_over_fixed": augmented_shift_evals / fixed_shift_evals,
        "evaluation_fraction_saved_by_pruning": (augmented_shift_evals - fixed_shift_evals) / augmented_shift_evals,
        "augmented_jacobian_seconds": float(augmented_jacobian_seconds),
        "gauge_fixed_jacobian_seconds": float(fixed_jacobian_seconds),
        "wall_time_ratio_augmented_over_fixed": float(augmented_jacobian_seconds / fixed_jacobian_seconds),
        "gauge_angle_sum": float(np.sum(gauge)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/gauge")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    seed = 41
    for n_qubits in (2, 3, 4):
        for n_layers in (1, 2):
            rows.append(benchmark_configuration(n_qubits, n_layers, seed))
            seed += 1

    with (out / "gauge_benchmark.json").open("w") as f:
        json.dump(rows, f, indent=2)
    with (out / "gauge_benchmark.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Quotient gauge-pruning summary")
    for row in rows:
        print(
            f"n={row['n_qubits']} L={row['n_layers']}: "
            f"params {row['augmented_parameters']} -> {row['physical_parameters']} "
            f"({100 * row['evaluation_fraction_saved_by_pruning']:.1f}% PS evals saved), "
            f"vector gauge rank={row['vector_space_gauge_rank']}, "
            f"projective QFI rank={row['projective_qfi_gauge_rank']}, "
            f"prediction error={row['max_prediction_error_after_pruning']:.3e}, "
            f"gauge grad={row['max_gauge_probability_gradient']:.3e}, "
            f"wall ratio={row['wall_time_ratio_augmented_over_fixed']:.2f}x"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
