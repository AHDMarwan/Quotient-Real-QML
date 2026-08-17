from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from automatic_pruning import shifted_circuit_shots_per_estimate
from qreal_vqc import QRealVQC
from quotient_real import QuotientRealVQC


@dataclass(frozen=True)
class StudyConfig:
    gradient_seeds: int = 500
    convergence_seeds: int = 200
    gradient_replicates: int = 100
    epochs: int = 30
    learning_rate: float = 0.15
    bootstrap_samples: int = 4000


def _bootstrap_paired(
    fixed: np.ndarray,
    augmented: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    fixed = np.asarray(fixed, dtype=np.float64)
    augmented = np.asarray(augmented, dtype=np.float64)
    if fixed.shape != augmented.shape or fixed.ndim != 1:
        raise ValueError("fixed and augmented must be paired one-dimensional arrays")
    if fixed.size < 2:
        raise ValueError("at least two paired observations are required")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, fixed.size, size=(bootstrap_samples, fixed.size))
    fixed_boot = fixed[indices].mean(axis=1)
    augmented_boot = augmented[indices].mean(axis=1)
    diff_boot = augmented_boot - fixed_boot
    ratio_boot = augmented_boot / np.maximum(fixed_boot, 1e-15)
    diff_ci = np.quantile(diff_boot, [0.025, 0.975])
    ratio_ci = np.quantile(ratio_boot, [0.025, 0.975])
    return {
        "mean_difference_ci95_low": float(diff_ci[0]),
        "mean_difference_ci95_high": float(diff_ci[1]),
        "mean_ratio_ci95_low": float(ratio_ci[0]),
        "mean_ratio_ci95_high": float(ratio_ci[1]),
    }


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _paired_summary(
    fixed: np.ndarray,
    augmented: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    fixed = np.asarray(fixed, dtype=np.float64)
    augmented = np.asarray(augmented, dtype=np.float64)
    diff = augmented - fixed
    std = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    cohen_dz = float(np.mean(diff) / std) if std > 0 else 0.0

    t_result = ttest_rel(augmented, fixed, alternative="two-sided")
    try:
        w_result = wilcoxon(augmented, fixed, alternative="two-sided", zero_method="wilcox")
        wilcoxon_p = float(w_result.pvalue)
    except ValueError:
        wilcoxon_p = 1.0

    worse = int(np.sum(augmented > fixed))
    worse_ci = _wilson_interval(worse, fixed.size)
    out = {
        "n_pairs": int(fixed.size),
        "mean_fixed": float(np.mean(fixed)),
        "mean_augmented": float(np.mean(augmented)),
        "median_fixed": float(np.median(fixed)),
        "median_augmented": float(np.median(augmented)),
        "mean_difference_augmented_minus_fixed": float(np.mean(diff)),
        "mean_ratio_augmented_over_fixed": float(np.mean(augmented) / max(float(np.mean(fixed)), 1e-15)),
        "fraction_augmented_worse": float(worse / fixed.size),
        "fraction_augmented_worse_ci95_low": worse_ci[0],
        "fraction_augmented_worse_ci95_high": worse_ci[1],
        "paired_cohen_dz": cohen_dz,
        "paired_t_pvalue": float(t_result.pvalue),
        "wilcoxon_pvalue": wilcoxon_p,
    }
    out.update(
        _bootstrap_paired(
            fixed,
            augmented,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    )
    return out


def _holm_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return [float(x) for x in adjusted]


def _gradient_seed_result(
    seed: int,
    total_shift_shots: int,
    replicates: int,
) -> tuple[float, float, int, int]:
    rng = np.random.default_rng(seed)
    model = QuotientRealVQC(3, 1)
    theta = model.initial_parameters(seed=10_000 + seed)
    x = rng.normal(size=3)
    p = model.n_parameters
    augmented_count = p + 3
    shots_fixed = shifted_circuit_shots_per_estimate(total_shift_shots, p)
    shots_augmented = shifted_circuit_shots_per_estimate(total_shift_shots, augmented_count)

    shift = np.pi / 2.0
    fixed_errors = []
    augmented_errors = []
    for j in range(p):
        plus = theta.copy(); plus[j] += shift
        minus = theta.copy(); minus[j] -= shift
        p_plus = float(model.probability(x, plus))
        p_minus = float(model.probability(x, minus))
        exact = 0.5 * (p_plus - p_minus)

        fixed_est = 0.5 * (
            rng.binomial(shots_fixed, p_plus, size=replicates) / shots_fixed
            - rng.binomial(shots_fixed, p_minus, size=replicates) / shots_fixed
        )
        augmented_est = 0.5 * (
            rng.binomial(shots_augmented, p_plus, size=replicates) / shots_augmented
            - rng.binomial(shots_augmented, p_minus, size=replicates) / shots_augmented
        )
        fixed_errors.append(np.mean((fixed_est - exact) ** 2))
        augmented_errors.append(np.mean((augmented_est - exact) ** 2))

    return (
        float(np.mean(fixed_errors)),
        float(np.mean(augmented_errors)),
        shots_fixed,
        shots_augmented,
    )


def gradient_noise_study(
    budgets: list[int],
    *,
    seeds: int,
    replicates: int,
    bootstrap_samples: int,
) -> tuple[list[dict], list[dict]]:
    summaries: list[dict] = []
    trials: list[dict] = []
    for budget_index, budget in enumerate(budgets):
        fixed_values = []
        augmented_values = []
        shots_fixed = shots_augmented = 0
        for seed in range(seeds):
            fixed_mse, augmented_mse, shots_fixed, shots_augmented = _gradient_seed_result(
                seed,
                budget,
                replicates,
            )
            fixed_values.append(fixed_mse)
            augmented_values.append(augmented_mse)
            trials.append(
                {
                    "study": "gradient_noise",
                    "budget": budget,
                    "seed": seed,
                    "fixed_metric": fixed_mse,
                    "augmented_metric": augmented_mse,
                    "paired_difference": augmented_mse - fixed_mse,
                }
            )

        summary = {
            "study": "gradient_noise",
            "budget": budget,
            "gradient_replicates_per_parameter": replicates,
            "shots_per_shift_fixed": shots_fixed,
            "shots_per_shift_augmented": shots_augmented,
        }
        summary.update(
            _paired_summary(
                np.asarray(fixed_values),
                np.asarray(augmented_values),
                bootstrap_samples=bootstrap_samples,
                seed=50_000 + budget_index,
            )
        )
        summaries.append(summary)

    t_adj = _holm_adjust([row["paired_t_pvalue"] for row in summaries])
    w_adj = _holm_adjust([row["wilcoxon_pvalue"] for row in summaries])
    for row, t_p, w_p in zip(summaries, t_adj, w_adj):
        row["paired_t_pvalue_holm"] = t_p
        row["wilcoxon_pvalue_holm"] = w_p
        row["significant_fixed_advantage_95pct"] = bool(
            row["mean_difference_ci95_low"] > 0.0
            and min(t_p, w_p) < 0.05
        )
    return summaries, trials


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


def _convergence_seed_result(
    seed: int,
    total_shots_per_epoch: int,
    *,
    epochs: int,
    learning_rate: float,
) -> tuple[float, float, int, int]:
    data_rng = np.random.default_rng(seed)
    model = QRealVQC(2, 1, "complex")
    x = data_rng.uniform(-2.0, 2.0, size=(6, 2))
    teacher = data_rng.normal(0.0, 0.5, size=model.n_parameters)
    targets = model.probabilities(x, teacher)
    initial = teacher + data_rng.normal(0.0, 0.15, size=model.n_parameters)

    tail_losses = []
    shot_counts = []
    for exposed_count, rng_offset in ((4, 100_000), (6, 200_000)):
        theta = initial.copy()
        noise_rng = np.random.default_rng(seed + rng_offset + total_shots_per_epoch)
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
        tail = max(5, min(10, epochs // 3))
        tail_losses.append(float(np.mean(history[-tail:])))
        shot_counts.append(shots)

    return tail_losses[0], tail_losses[1], shot_counts[0], shot_counts[1]


def convergence_study(
    budgets: list[int],
    *,
    seeds: int,
    epochs: int,
    learning_rate: float,
    bootstrap_samples: int,
) -> tuple[list[dict], list[dict]]:
    summaries: list[dict] = []
    trials: list[dict] = []
    for budget_index, budget in enumerate(budgets):
        fixed_values = []
        augmented_values = []
        shots_fixed = shots_augmented = 0
        for seed in range(seeds):
            fixed_loss, augmented_loss, shots_fixed, shots_augmented = _convergence_seed_result(
                seed,
                budget,
                epochs=epochs,
                learning_rate=learning_rate,
            )
            fixed_values.append(fixed_loss)
            augmented_values.append(augmented_loss)
            trials.append(
                {
                    "study": "noisy_convergence",
                    "budget": budget,
                    "seed": seed,
                    "fixed_metric": fixed_loss,
                    "augmented_metric": augmented_loss,
                    "paired_difference": augmented_loss - fixed_loss,
                }
            )

        summary = {
            "study": "noisy_convergence",
            "budget": budget,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "shots_per_circuit_fixed": shots_fixed,
            "shots_per_circuit_augmented": shots_augmented,
        }
        summary.update(
            _paired_summary(
                np.asarray(fixed_values),
                np.asarray(augmented_values),
                bootstrap_samples=bootstrap_samples,
                seed=60_000 + budget_index,
            )
        )
        summaries.append(summary)

    t_adj = _holm_adjust([row["paired_t_pvalue"] for row in summaries])
    w_adj = _holm_adjust([row["wilcoxon_pvalue"] for row in summaries])
    for row, t_p, w_p in zip(summaries, t_adj, w_adj):
        row["paired_t_pvalue_holm"] = t_p
        row["wilcoxon_pvalue_holm"] = w_p
        row["significant_fixed_advantage_95pct"] = bool(
            row["mean_difference_ci95_low"] > 0.0
            and min(t_p, w_p) < 0.05
        )
    return summaries, trials


def _parse_budgets(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("budgets must be a comma-separated list of positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/statistical")
    parser.add_argument("--gradient-seeds", type=int, default=500)
    parser.add_argument("--convergence-seeds", type=int, default=200)
    parser.add_argument("--gradient-replicates", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--gradient-budgets", default="6000,12000,24000,48000")
    parser.add_argument("--convergence-budgets", default="750,1500,3000,6000")
    args = parser.parse_args()

    config = StudyConfig(
        gradient_seeds=args.gradient_seeds,
        convergence_seeds=args.convergence_seeds,
        gradient_replicates=args.gradient_replicates,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        bootstrap_samples=args.bootstrap_samples,
    )
    gradient_budgets = _parse_budgets(args.gradient_budgets)
    convergence_budgets = _parse_budgets(args.convergence_budgets)

    gradient_summary, gradient_trials = gradient_noise_study(
        gradient_budgets,
        seeds=config.gradient_seeds,
        replicates=config.gradient_replicates,
        bootstrap_samples=config.bootstrap_samples,
    )
    convergence_summary, convergence_trials = convergence_study(
        convergence_budgets,
        seeds=config.convergence_seeds,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        bootstrap_samples=config.bootstrap_samples,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_rows = gradient_summary + convergence_summary
    trial_rows = gradient_trials + convergence_trials

    payload = {
        "config": config.__dict__,
        "gradient_budgets": gradient_budgets,
        "convergence_budgets": convergence_budgets,
        "summary": summary_rows,
    }
    (out / "statistical_summary.json").write_text(json.dumps(payload, indent=2))

    summary_keys = sorted({key for row in summary_rows for key in row})
    with (out / "statistical_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_keys)
        writer.writeheader(); writer.writerows(summary_rows)

    trial_keys = sorted({key for row in trial_rows for key in row})
    with (out / "statistical_trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trial_keys)
        writer.writeheader(); writer.writerows(trial_rows)

    print("Large-scale statistical shot study")
    print(
        f"gradient seeds={config.gradient_seeds}, convergence seeds={config.convergence_seeds}, "
        f"bootstrap={config.bootstrap_samples}"
    )
    for row in summary_rows:
        label = "gradient" if row["study"] == "gradient_noise" else "convergence"
        print(
            f"{label:11s} budget={row['budget']:6d} ratio={row['mean_ratio_augmented_over_fixed']:.3f}x "
            f"diff95=[{row['mean_difference_ci95_low']:.3e}, {row['mean_difference_ci95_high']:.3e}] "
            f"Holm-p(t)={row['paired_t_pvalue_holm']:.3e} Holm-p(W)={row['wilcoxon_pvalue_holm']:.3e} "
            f"dz={row['paired_cohen_dz']:.3f} significant={row['significant_fixed_advantage_95pct']}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
