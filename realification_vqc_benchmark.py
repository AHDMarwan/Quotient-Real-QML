from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from realification_simulator import parameter_shift_jacobian, probabilities


def _median_time(fn, repeats: int = 3):
    values = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - t0)
    return float(np.median(values)), result


def _mse_train(backend: str, xs, targets, theta0, n: int, epochs: int = 3, lr: float = 0.2):
    theta = np.asarray(theta0, dtype=np.float64).copy()
    history = []
    t0 = time.perf_counter()
    for _ in range(epochs):
        p = probabilities(backend, xs, theta, n)
        history.append(float(np.mean((p - targets) ** 2)))
        jac = parameter_shift_jacobian(backend, xs, theta, n)
        grad = jac.T @ (2.0 * (p - targets) / len(targets))
        theta -= lr * grad
    elapsed = time.perf_counter() - t0
    final = probabilities(backend, xs, theta, n)
    return theta, history, final, float(elapsed)


def benchmark_configuration(n: int, seed: int):
    rng = np.random.default_rng(seed)
    samples = 6
    xs = rng.normal(size=(samples, n))
    theta = rng.normal(scale=0.2, size=2 * n)

    tc, pc = _median_time(lambda: probabilities("complex", xs, theta, n), 3)
    tr, pr = _median_time(lambda: probabilities("realified", xs, theta, n), 3)
    jtc, jc = _median_time(lambda: parameter_shift_jacobian("complex", xs, theta, n), 2)
    jtr, jr = _median_time(lambda: parameter_shift_jacobian("realified", xs, theta, n), 2)

    row = {
        "n_qubits": n,
        "samples": samples,
        "inference_complex_seconds": tc,
        "inference_realified_seconds": tr,
        "inference_ratio_realified_over_complex": tr / tc,
        "jacobian_complex_seconds": jtc,
        "jacobian_realified_seconds": jtr,
        "jacobian_ratio_realified_over_complex": jtr / jtc,
        "max_prediction_error": float(np.max(np.abs(pr - pc))),
        "max_jacobian_error": float(np.max(np.abs(jr - jc))),
    }

    if n <= 8:
        teacher = rng.normal(scale=0.35, size=2 * n)
        targets = probabilities("complex", xs, teacher, n)
        start = teacher + rng.normal(scale=0.15, size=2 * n)
        ctheta, ch, cp, ctime = _mse_train("complex", xs, targets, start, n)
        rtheta, rh, rp, rtime = _mse_train("realified", xs, targets, start, n)
        labels = targets >= np.median(targets)
        row.update({
            "training_epochs": 3,
            "training_complex_seconds": ctime,
            "training_realified_seconds": rtime,
            "training_ratio_realified_over_complex": rtime / ctime,
            "training_parameter_error": float(np.max(np.abs(rtheta - ctheta))),
            "training_prediction_error": float(np.max(np.abs(rp - cp))),
            "training_loss_error": float(abs(rh[-1] - ch[-1])),
            "complex_accuracy": float(np.mean((cp >= 0.5) == labels)),
            "realified_accuracy": float(np.mean((rp >= 0.5) == labels)),
        })
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/realification")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [benchmark_configuration(n, 900 + i) for i, n in enumerate((2, 4, 6, 8, 10))]
    with (out / "vqc_end_to_end.json").open("w") as f:
        json.dump(rows, f, indent=2)
    keys = sorted({key for row in rows for key in row})
    with (out / "vqc_end_to_end.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print("End-to-end VQC summary")
    for row in rows:
        msg = (
            f"n={row['n_qubits']:2d}: inference={row['inference_ratio_realified_over_complex']:.3f}x "
            f"jacobian={row['jacobian_ratio_realified_over_complex']:.3f}x "
            f"pred-error={row['max_prediction_error']:.3e} jac-error={row['max_jacobian_error']:.3e}"
        )
        if "training_ratio_realified_over_complex" in row:
            msg += f" training={row['training_ratio_realified_over_complex']:.3f}x"
        print(msg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
