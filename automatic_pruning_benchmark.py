from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from automatic_pruning import (
    MixedGaugeQuotientVQC,
    aggregate_projective_qfim,
    ideal_parameter_shift_variance,
    project_parameters_to_active_subspace,
    projective_fidelity,
    projector_distance,
    qfim_active_subspace,
    shifted_circuit_shots_per_estimate,
)
from qreal_vqc import QRealVQC
from quotient_gauge import GaugeAugmentedQuotientVQC
from quotient_real import QuotientRealVQC


def geometry_benchmark(n_qubits: int, n_layers: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    model = MixedGaugeQuotientVQC(n_qubits, n_layers, mixing_seed=1000 + seed)
    theta = model.initial_parameters(seed=2000 + seed)
    geometry_inputs = rng.normal(size=(6, n_qubits))

    t0 = time.perf_counter()
    qfim = aggregate_projective_qfim(model, geometry_inputs, theta)
    qfim_seconds = time.perf_counter() - t0
    values, active, null = qfim_active_subspace(qfim)

    pruned = project_parameters_to_active_subspace(theta, active)
    held_out = rng.normal(size=(8, n_qubits))
    p_before = model.probabilities(held_out, theta)
    p_after = model.probabilities(held_out, pruned)
    fidelities = [
        projective_fidelity(
            model.complex_state(row, theta), model.complex_state(row, pruned)
        )
        for row in held_out[:4]
    ]

    d = model.n_parameters
    rank = active.shape[1]
    return {
        "experiment": "automatic_geometry",
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "exposed_parameters": d,
        "true_physical_parameters": model.n_physical_parameters,
        "true_gauge_parameters": model.n_gauge_parameters,
        "qfim_rank": rank,
        "qfim_nullity": null.shape[1],
        "parameter_fraction_pruned": float((d - rank) / d),
        "active_projector_error": projector_distance(active, model.true_active_basis),
        "null_projector_error": projector_distance(null, model.true_null_basis),
        "min_exposed_qfim_diagonal": float(np.min(np.diag(qfim))),
        "min_active_qfim_eigenvalue": float(np.min(values)),
        "max_active_qfim_eigenvalue": float(np.max(values)),
        "qfim_seconds": float(qfim_seconds),
        "max_prediction_error_after_automatic_pruning": float(
            np.max(np.abs(p_before - p_after))
        ),
        "min_projective_fidelity_after_automatic_pruning": float(np.min(fidelities)),
    }


def shot_noise_gradient_benchmark(seed: int = 123, trials: int = 5000) -> dict:
    """Compare physical parameter-shift noise under the same total shift-shot budget."""
    rng = np.random.default_rng(seed)
    n_qubits = 3
    n_layers = 1
    fixed = QuotientRealVQC(n_qubits, n_layers)
    augmented = GaugeAugmentedQuotientVQC(n_qubits, n_layers)
    physical = fixed.initial_parameters(seed=seed + 1)
    x = rng.normal(size=n_qubits)

    p = fixed.n_parameters
    d = augmented.n_parameters
    total_shift_shots = 24000
    s_fixed = shifted_circuit_shots_per_estimate(total_shift_shots, p)
    s_aug = shifted_circuit_shots_per_estimate(total_shift_shots, d)

    shift = np.pi / 2.0
    p_plus = np.empty(p)
    p_minus = np.empty(p)
    exact = np.empty(p)
    for j in range(p):
        plus = physical.copy(); plus[j] += shift
        minus = physical.copy(); minus[j] -= shift
        p_plus[j] = fixed.probability(x, plus)
        p_minus[j] = fixed.probability(x, minus)
        exact[j] = 0.5 * (p_plus[j] - p_minus[j])

    fixed_sq_error = []
    augmented_sq_error = []
    for j in range(p):
        fixed_est = 0.5 * (
            rng.binomial(s_fixed, p_plus[j], size=trials) / s_fixed
            - rng.binomial(s_fixed, p_minus[j], size=trials) / s_fixed
        )
        aug_est = 0.5 * (
            rng.binomial(s_aug, p_plus[j], size=trials) / s_aug
            - rng.binomial(s_aug, p_minus[j], size=trials) / s_aug
        )
        fixed_sq_error.append((fixed_est - exact[j]) ** 2)
        augmented_sq_error.append((aug_est - exact[j]) ** 2)

    fixed_mse = float(np.mean(np.asarray(fixed_sq_error)))
    aug_mse = float(np.mean(np.asarray(augmented_sq_error)))
    analytic_fixed = float(np.mean([
        ideal_parameter_shift_variance(a, b, s_fixed)
        for a, b in zip(p_plus, p_minus)
    ]))
    analytic_aug = float(np.mean([
        ideal_parameter_shift_variance(a, b, s_aug)
        for a, b in zip(p_plus, p_minus)
    ]))

    # Gauge columns have zero exact gradient but nonzero finite-shot variance.
    p0 = fixed.probability(x, physical)
    gauge_est = 0.5 * (
        rng.binomial(s_aug, p0, size=(trials, augmented.n_gauge_parameters)) / s_aug
        - rng.binomial(s_aug, p0, size=(trials, augmented.n_gauge_parameters)) / s_aug
    )

    return {
        "experiment": "fixed_budget_gradient_noise",
        "physical_parameters": p,
        "augmented_parameters": d,
        "total_shift_shots": total_shift_shots,
        "shots_per_shift_fixed": s_fixed,
        "shots_per_shift_augmented": s_aug,
        "parameter_shift_fraction_saved": float(1.0 - p / d),
        "physical_gradient_mse_fixed": fixed_mse,
        "physical_gradient_mse_augmented": aug_mse,
        "physical_gradient_mse_ratio_augmented_over_fixed": float(aug_mse / fixed_mse),
        "analytic_variance_fixed": analytic_fixed,
        "analytic_variance_augmented": analytic_aug,
        "analytic_variance_ratio_augmented_over_fixed": float(analytic_aug / analytic_fixed),
        "gauge_gradient_noise_mse": float(np.mean(gauge_est ** 2)),
    }


def _noisy_mse_step(
    model: QRealVQC,
    x: np.ndarray,
    targets: np.ndarray,
    parameters: np.ndarray,
    *,
    exposed_parameter_count: int,
    total_shots: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """One shot-noisy MSE-gradient estimate with a fixed total circuit-shot budget."""
    theta = np.asarray(parameters, dtype=np.float64)
    n_samples = x.shape[0]
    shots = max(2, total_shots // (n_samples * (1 + 2 * exposed_parameter_count)))

    exact_p = model.probabilities(x, theta)
    noisy_p = rng.binomial(shots, exact_p) / shots
    jac = np.empty((n_samples, theta.size), dtype=np.float64)
    shift = np.pi / 2.0
    for j in range(theta.size):
        plus = theta.copy(); plus[j] += shift
        minus = theta.copy(); minus[j] -= shift
        p_plus = model.probabilities(x, plus)
        p_minus = model.probabilities(x, minus)
        noisy_plus = rng.binomial(shots, p_plus) / shots
        noisy_minus = rng.binomial(shots, p_minus) / shots
        jac[:, j] = 0.5 * (noisy_plus - noisy_minus)

    gradient = jac.T @ (2.0 * (noisy_p - targets) / n_samples)
    return gradient, shots


def noisy_convergence_benchmark(
    seeds=range(20),
    *,
    total_shots_per_epoch: int = 1500,
    epochs: int = 30,
    learning_rate: float = 0.15,
) -> dict:
    """Teacher-student control under the same total shot budget per optimization step.

    The gauge-fixed model is the ordinary complex VQC.  The augmented locally-real
    parameterization has two extra quotient gauge coordinates for n=2.  Since the
    extra coordinates are unobservable, they only reduce the shots available to
    each informative shifted circuit when total shots are held fixed.
    """
    fixed_tail_losses = []
    augmented_tail_losses = []
    fixed_shots = []
    augmented_shots = []

    for seed in seeds:
        data_rng = np.random.default_rng(seed)
        model = QRealVQC(2, 1, "complex")
        x = data_rng.uniform(-2.0, 2.0, size=(6, 2))
        teacher = data_rng.normal(0.0, 0.5, size=model.n_parameters)
        targets = model.probabilities(x, teacher)
        initial = teacher + data_rng.normal(0.0, 0.15, size=model.n_parameters)

        histories = []
        shot_counts = []
        for exposed_count, rng_offset in ((4, 10000), (6, 20000)):
            theta = initial.copy()
            noise_rng = np.random.default_rng(seed + rng_offset)
            history = []
            shots = 0
            for _ in range(epochs):
                grad, shots = _noisy_mse_step(
                    model,
                    x,
                    targets,
                    theta,
                    exposed_parameter_count=exposed_count,
                    total_shots=total_shots_per_epoch,
                    rng=noise_rng,
                )
                theta -= learning_rate * grad
                exact_loss = float(np.mean((model.probabilities(x, theta) - targets) ** 2))
                history.append(exact_loss)
            histories.append(float(np.mean(history[-8:])))
            shot_counts.append(shots)

        fixed_tail_losses.append(histories[0])
        augmented_tail_losses.append(histories[1])
        fixed_shots.append(shot_counts[0])
        augmented_shots.append(shot_counts[1])

    fixed_arr = np.asarray(fixed_tail_losses)
    aug_arr = np.asarray(augmented_tail_losses)
    return {
        "experiment": "fixed_budget_noisy_convergence",
        "n_seeds": int(len(fixed_arr)),
        "epochs": epochs,
        "total_shots_per_epoch": total_shots_per_epoch,
        "complex_gauge_fixed_parameters": 4,
        "locally_real_augmented_parameters": 6,
        "shots_per_circuit_complex_gauge_fixed": int(fixed_shots[0]),
        "shots_per_circuit_locally_real_augmented": int(augmented_shots[0]),
        "mean_tail_mse_complex_gauge_fixed": float(np.mean(fixed_arr)),
        "mean_tail_mse_locally_real_augmented": float(np.mean(aug_arr)),
        "median_tail_mse_complex_gauge_fixed": float(np.median(fixed_arr)),
        "median_tail_mse_locally_real_augmented": float(np.median(aug_arr)),
        "mean_mse_ratio_augmented_over_fixed": float(np.mean(aug_arr) / np.mean(fixed_arr)),
        "fraction_trials_augmented_worse": float(np.mean(aug_arr > fixed_arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/automatic")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    seed = 70
    for n_qubits, n_layers in ((2, 1), (3, 1), (3, 2), (4, 1)):
        rows.append(geometry_benchmark(n_qubits, n_layers, seed))
        seed += 1
    rows.append(shot_noise_gradient_benchmark())
    rows.append(noisy_convergence_benchmark())

    with (out / "automatic_pruning_benchmark.json").open("w") as f:
        json.dump(rows, f, indent=2)
    keys = sorted({key for row in rows for key in row})
    with (out / "automatic_pruning_benchmark.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print("Automatic quotient-QFIM pruning summary")
    for row in rows:
        if row["experiment"] == "automatic_geometry":
            print(
                f"n={row['n_qubits']} L={row['n_layers']}: "
                f"rank {row['qfim_rank']}/{row['exposed_parameters']}, "
                f"pruned={100 * row['parameter_fraction_pruned']:.1f}%, "
                f"active-subspace error={row['active_projector_error']:.3e}, "
                f"prediction error={row['max_prediction_error_after_automatic_pruning']:.3e}"
            )
        elif row["experiment"] == "fixed_budget_gradient_noise":
            print(
                "shot gradient: "
                f"{row['shots_per_shift_augmented']} -> {row['shots_per_shift_fixed']} shots/shift after pruning, "
                f"physical-gradient MSE ratio={row['physical_gradient_mse_ratio_augmented_over_fixed']:.3f}x, "
                f"analytic variance ratio={row['analytic_variance_ratio_augmented_over_fixed']:.3f}x"
            )
        else:
            print(
                "noisy convergence: "
                f"mean tail MSE ratio augmented/fixed={row['mean_mse_ratio_augmented_over_fixed']:.3f}x, "
                f"augmented worse in {100 * row['fraction_trials_augmented_worse']:.1f}% of trials"
            )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
