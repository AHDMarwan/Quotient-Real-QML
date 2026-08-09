from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from realification_general import apply_complex, apply_realified_3m, apply_realified_4m, random_unitary
from realification_simulator import (
    as_complex,
    probability_one_complex,
    probability_one_realified,
    simulate_complex,
    simulate_realified,
    state_bytes,
)


def _median_time(fn, repeats: int):
    values = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - t0)
    return float(np.median(values)), result


def worker(backend: str, n: int, precision: str, layers: int, repeats: int, seed: int):
    rng = np.random.default_rng(seed)
    angles = rng.normal(size=n)
    theta = rng.normal(scale=0.2, size=2 * n * layers)
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if backend == "complex":
        dtype = np.complex64 if precision == "32" else np.complex128
        elapsed, state = _median_time(lambda: simulate_complex(angles, theta, n, layers, dtype=dtype), repeats)
        probability = probability_one_complex(state, n)
    else:
        dtype = np.float32 if precision == "32" else np.float64
        elapsed, state = _median_time(lambda: simulate_realified(angles, theta, n, layers, dtype=dtype), repeats)
        probability = probability_one_realified(state, n)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "backend": backend,
        "n_qubits": n,
        "precision_bits": int(precision),
        "layers": layers,
        "repeats": repeats,
        "median_seconds": elapsed,
        "probability": probability,
        "state_bytes": state_bytes(state),
        "peak_rss_delta_kib": int(max(0, peak - baseline)),
    }


def run_worker_subprocess(backend: str, n: int, precision: str, layers: int, repeats: int, seed: int):
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        "--backend", backend,
        "--n-qubits", str(n),
        "--precision", precision,
        "--layers", str(layers),
        "--repeats", str(repeats),
        "--seed", str(seed),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def scaling_suite():
    rows = []
    for i, n in enumerate((4, 8, 12, 16, 18, 20)):
        repeats = 5 if n <= 12 else 3 if n <= 16 else 2 if n <= 18 else 1
        complex_row = run_worker_subprocess("complex", n, "64", 1, repeats, 100 + i)
        real_row = run_worker_subprocess("realified", n, "64", 1, repeats, 100 + i)
        rows.append({
            "n_qubits": n,
            "complex_seconds": complex_row["median_seconds"],
            "realified_seconds": real_row["median_seconds"],
            "realified_over_complex_time": real_row["median_seconds"] / complex_row["median_seconds"],
            "complex_state_bytes": complex_row["state_bytes"],
            "realified_state_bytes": real_row["state_bytes"],
            "state_storage_ratio_realified_over_complex": real_row["state_bytes"] / complex_row["state_bytes"],
            "complex_peak_rss_delta_kib": complex_row["peak_rss_delta_kib"],
            "realified_peak_rss_delta_kib": real_row["peak_rss_delta_kib"],
            "probability_error": abs(real_row["probability"] - complex_row["probability"]),
            "repeats": repeats,
        })
    return rows


def precision_suite():
    rows = []
    rng = np.random.default_rng(400)
    for n in (4, 8, 12, 16):
        angles = rng.normal(size=n)
        theta = rng.normal(scale=0.2, size=2 * n)
        c64 = simulate_complex(angles, theta, n, dtype=np.complex128)
        r64 = simulate_realified(angles, theta, n, dtype=np.float64)
        c32 = simulate_complex(angles, theta, n, dtype=np.complex64)
        r32 = simulate_realified(angles, theta, n, dtype=np.float32)
        rows.append({
            "n_qubits": n,
            "float64_state_error_vs_complex128": float(np.max(np.abs(as_complex(r64) - c64))),
            "complex64_state_error_vs_complex128": float(np.max(np.abs(c32.astype(np.complex128) - c64))),
            "float32_realified_error_vs_complex128": float(np.max(np.abs(as_complex(r32, np.complex64).astype(np.complex128) - c64))),
            "float32_realified_error_vs_complex64": float(np.max(np.abs(as_complex(r32, np.complex64) - c32))),
            "complex128_bytes": state_bytes(c64),
            "float64_realified_bytes": state_bytes(r64),
            "complex64_bytes": state_bytes(c32),
            "float32_realified_bytes": state_bytes(r32),
        })
    return rows


def _random_state(n: int, seed: int):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    return np.asarray(psi / np.linalg.norm(psi), dtype=np.complex128)


def three_m_suite():
    rows = []
    for i, n in enumerate((8, 12, 16, 18)):
        psi = _random_state(n, 500 + i)
        real = (psi.real.copy(), psi.imag.copy())
        gate = random_unitary(600 + i)
        q = n // 2
        repeats = 20 if n <= 8 else 10 if n <= 12 else 5 if n <= 16 else 3
        tc, target = _median_time(lambda: apply_complex(psi, gate, q, n), repeats)
        t4, out4 = _median_time(lambda: apply_realified_4m(real, gate, q, n), repeats)
        t3, out3 = _median_time(lambda: apply_realified_3m(real, gate, q, n), repeats)
        rows.append({
            "n_qubits": n,
            "complex_seconds": tc,
            "realified_4m_seconds": t4,
            "realified_3m_seconds": t3,
            "ratio_4m_over_complex": t4 / tc,
            "ratio_3m_over_complex": t3 / tc,
            "speedup_3m_over_4m": t4 / t3,
            "error_4m_vs_complex": float(np.max(np.abs(as_complex(out4) - target))),
            "error_3m_vs_complex": float(np.max(np.abs(as_complex(out3) - target))),
            "repeats": repeats,
        })
    return rows


def write_rows(path: Path, rows):
    with path.with_suffix(".json").open("w") as f:
        json.dump(rows, f, indent=2)
    with path.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/realification")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", choices=("complex", "realified"))
    parser.add_argument("--n-qubits", type=int)
    parser.add_argument("--precision", choices=("32", "64"), default="64")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.backend, args.n_qubits, args.precision, args.layers, args.repeats, args.seed)))
        return

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scaling = scaling_suite()
    precision = precision_suite()
    three_m = three_m_suite()
    write_rows(out / "scaling", scaling)
    write_rows(out / "precision", precision)
    write_rows(out / "three_m", three_m)
    with (out / "environment.json").open("w") as f:
        json.dump({
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        }, f, indent=2)

    print("Realification scaling summary")
    for row in scaling:
        print(f"n={row['n_qubits']:2d}: time ratio={row['realified_over_complex_time']:.3f}x storage={row['state_storage_ratio_realified_over_complex']:.1f}x p-error={row['probability_error']:.3e}")
    print("General-gate 3M summary")
    for row in three_m:
        print(f"n={row['n_qubits']:2d}: 3M/complex={row['ratio_3m_over_complex']:.3f}x 3M speedup vs 4M={row['speedup_3m_over_4m']:.3f}x error={row['error_3m_vs_complex']:.3e}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
