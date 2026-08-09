from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from qreal_vqc import QRealVQC
from quotient_real import QuotientRealVQC, compress_quotient_state


def state_storage(n_qubits: int) -> dict[str, int]:
    dim = 1 << n_qubits
    return {
        "complex_bytes": dim * np.dtype(np.complex128).itemsize,
        "global_realification_bytes": 2 * dim * np.dtype(np.float64).itemsize,
        "quotient_coordinates_bytes": 2 * dim * np.dtype(np.float64).itemsize,
        "quotient_explicit_canonical_bytes": (1 << (2 * n_qubits))
        * np.dtype(np.float64).itemsize,
    }


def run(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []

    for n_qubits in (1, 2, 3, 4, 5, 6):
        x = rng.normal(size=(12, n_qubits))
        complex_model = QRealVQC(n_qubits, 1, "complex")
        real_model = QRealVQC(n_qubits, 1, "structured_real")
        quotient_model = QuotientRealVQC(n_qubits, 1)
        theta = rng.normal(scale=0.3, size=complex_model.n_parameters)
        reference = complex_model.probabilities(x, theta)

        for name, model in (
            ("complex", complex_model),
            ("structured_real", real_model),
            ("quotient_explicit", quotient_model),
        ):
            model.probabilities(x[:1], theta)
            samples = []
            for _ in range(3):
                t0 = time.perf_counter()
                prediction = model.probabilities(x, theta)
                samples.append(time.perf_counter() - t0)
            rows.append(
                {
                    "experiment": "inference",
                    "n_qubits": n_qubits,
                    "backend": name,
                    "seconds_median": float(np.median(samples)),
                    "max_abs_error_vs_complex": float(
                        np.max(np.abs(prediction - reference))
                    ),
                }
            )

        memory = state_storage(n_qubits)
        rows.append(
            {
                "experiment": "storage",
                "n_qubits": n_qubits,
                **memory,
                "explicit_overhead_vs_quotient_coords": memory[
                    "quotient_explicit_canonical_bytes"
                ]
                / memory["quotient_coordinates_bytes"],
            }
        )

        angles = x[0]
        complex_state = complex_model.state(angles, theta)
        quotient_state = quotient_model.state(angles, theta)
        coords = compress_quotient_state(quotient_state, n_qubits)
        dim = 1 << n_qubits
        coordinate_error = max(
            np.max(np.abs(coords[:dim] - complex_state.real)),
            np.max(np.abs(coords[dim:] - complex_state.imag)),
        )
        rows.append(
            {
                "experiment": "state_equivalence",
                "n_qubits": n_qubits,
                "max_coordinate_error": float(coordinate_error),
            }
        )

    # Exact quotient realification should produce the same gradients and hence
    # the same optimization path when initialized identically.
    n_qubits = 3
    x = rng.normal(size=(16, n_qubits))
    y = (np.sum(x, axis=1) > 0).astype(float)
    complex_model = QRealVQC(n_qubits, 1, "complex")
    quotient_model = QuotientRealVQC(n_qubits, 1)
    theta0 = rng.normal(scale=0.2, size=complex_model.n_parameters)

    def train(model, theta):
        theta = theta.copy()
        history = []
        eps = 1e-9
        for _ in range(3):
            prediction = np.clip(model.probabilities(x, theta), eps, 1 - eps)
            history.append(
                float(
                    -np.mean(
                        y * np.log(prediction)
                        + (1 - y) * np.log(1 - prediction)
                    )
                )
            )
            jacobian = model.parameter_shift_probability_jacobian(x, theta)
            dloss_dp = (
                (prediction - y)
                / (prediction * (1 - prediction))
                / y.size
            )
            theta -= 0.05 * (jacobian.T @ dloss_dp)
        return theta, history, model.probabilities(x, theta)

    theta_complex, loss_complex, prediction_complex = train(complex_model, theta0)
    theta_quotient, loss_quotient, prediction_quotient = train(
        quotient_model, theta0
    )
    rows.append(
        {
            "experiment": "training_equivalence",
            "n_qubits": n_qubits,
            "max_parameter_error": float(
                np.max(np.abs(theta_complex - theta_quotient))
            ),
            "max_prediction_error": float(
                np.max(np.abs(prediction_complex - prediction_quotient))
            ),
            "max_loss_error": float(
                np.max(np.abs(np.asarray(loss_complex) - np.asarray(loss_quotient)))
            ),
        }
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/quotient")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = run()
    with (out / "quotient_benchmark.json").open("w") as f:
        json.dump(rows, f, indent=2)

    keys = sorted({key for row in rows for key in row})
    with (out / "quotient_benchmark.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print("Quotient-real inference summary")
    for n_qubits in (1, 2, 3, 4, 5, 6):
        inference = [
            row
            for row in rows
            if row["experiment"] == "inference"
            and row["n_qubits"] == n_qubits
        ]
        values = {row["backend"]: row for row in inference}
        ratio = (
            values["quotient_explicit"]["seconds_median"]
            / values["complex"]["seconds_median"]
        )
        storage = next(
            row
            for row in rows
            if row["experiment"] == "storage"
            and row["n_qubits"] == n_qubits
        )
        equivalence = next(
            row
            for row in rows
            if row["experiment"] == "state_equivalence"
            and row["n_qubits"] == n_qubits
        )
        print(
            f"n={n_qubits}: quotient/complex time={ratio:.2f}x "
            f"explicit-storage-overhead="
            f"{storage['explicit_overhead_vs_quotient_coords']:.1f}x "
            f"state-error={equivalence['max_coordinate_error']:.3e}"
        )

    training = next(
        row for row in rows if row["experiment"] == "training_equivalence"
    )
    print(
        "training-equivalence "
        f"parameter-error={training['max_parameter_error']:.3e} "
        f"prediction-error={training['max_prediction_error']:.3e} "
        f"loss-error={training['max_loss_error']:.3e}"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
