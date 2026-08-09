from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qreal_vqc import QRealVQC


def state_bytes(n_qubits: int, backend: str) -> int:
    dim = 1 << n_qubits
    if backend == "complex":
        return dim * np.dtype(np.complex128).itemsize
    if backend == "structured_real":
        return 2 * dim * np.dtype(np.float64).itemsize
    return dim * np.dtype(np.float64).itemsize


def inference_benchmark(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for n_qubits in (2, 3, 4, 5, 6):
        x = rng.normal(size=(24, n_qubits))
        init_model = QRealVQC(n_qubits, 1, "complex")
        theta = rng.normal(scale=0.3, size=init_model.n_parameters)
        reference = init_model.probabilities(x, theta)
        for backend in ("complex", "structured_real", "naive_real"):
            model = QRealVQC(n_qubits, 1, backend)
            model.probabilities(x[:2], theta)
            samples = []
            for _ in range(3):
                t0 = time.perf_counter()
                p = model.probabilities(x, theta)
                samples.append(time.perf_counter() - t0)
            rows.append({
                "experiment": "inference",
                "n_qubits": n_qubits,
                "backend": backend,
                "seconds_median": float(np.median(samples)),
                "state_bytes": state_bytes(n_qubits, backend),
                "max_abs_error_vs_complex": float(np.max(np.abs(p - reference))) if backend == "structured_real" else None,
            })
    return rows


def training_benchmark(seeds=(0, 1, 2)):
    data = load_breast_cancer()
    rows = []
    for seed in seeds:
        x_train, x_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.25, random_state=seed, stratify=data.target
        )
        pipe = make_pipeline(StandardScaler(), PCA(n_components=3, random_state=seed))
        x_train = pipe.fit_transform(x_train)
        x_test = pipe.transform(x_test)
        scale = np.maximum(np.std(x_train, axis=0, keepdims=True), 1e-9)
        x_train = np.clip(x_train / scale, -np.pi, np.pi)[:120]
        y_train = y_train[:120]
        x_test = np.clip(x_test / scale, -np.pi, np.pi)

        init = QRealVQC(3, 1, "complex").initial_parameters(seed=100 + seed)
        preds = {}
        for backend in ("complex", "structured_real", "naive_real"):
            model = QRealVQC(3, 1, backend)
            t0 = time.perf_counter()
            theta, hist = model.fit(
                x_train, y_train, parameters=init, epochs=5, learning_rate=0.08
            )
            elapsed = time.perf_counter() - t0
            p = model.probabilities(x_test, theta)
            preds[backend] = p
            rows.append({
                "experiment": "training",
                "seed": seed,
                "backend": backend,
                "seconds": float(elapsed),
                "final_loss": float(hist[-1]),
                "accuracy": float(accuracy_score(y_test, p >= 0.5)),
                "roc_auc": float(roc_auc_score(y_test, p)),
                "n_parameters": model.n_parameters,
            })
        eq = float(np.max(np.abs(preds["complex"] - preds["structured_real"])))
        rows.append({"experiment": "equivalence", "seed": seed, "max_prediction_error": eq})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = inference_benchmark() + training_benchmark()
    with (out / "benchmark.json").open("w") as f:
        json.dump(rows, f, indent=2)

    keys = sorted({k for row in rows for k in row})
    with (out / "benchmark.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)

    train = [r for r in rows if r["experiment"] == "training"]
    print("Training summary (mean over seeds)")
    for backend in ("complex", "structured_real", "naive_real"):
        rs = [r for r in train if r["backend"] == backend]
        print(
            f"{backend:16s} accuracy={np.mean([r['accuracy'] for r in rs]):.4f} "
            f"auc={np.mean([r['roc_auc'] for r in rs]):.4f} "
            f"time={np.mean([r['seconds'] for r in rs]):.3f}s"
        )
    eq = [r["max_prediction_error"] for r in rows if r["experiment"] == "equivalence"]
    print(f"worst trained complex/structured-real prediction error: {max(eq):.3e}")
    print(f"wrote {out / 'benchmark.json'} and {out / 'benchmark.csv'}")


if __name__ == "__main__":
    main()
