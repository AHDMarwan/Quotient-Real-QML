# Realification study results

Authoritative GitHub Actions run: `31337512300` (`realification-study` job), Ubuntu 24.04, Python 3.11.15, NumPy 2.4.6.

## Main findings

- Exact state storage is identical at matched precision: `complex128` and two `float64` arrays both use 16 bytes per amplitude; `complex64` and two `float32` arrays both use 8 bytes per amplitude.
- Specialized float64 realification reproduces complex128 probabilities/states at about `1e-16` numerical error.
- Float32 realification has state error around `1e-7` versus complex128, comparable to ordinary complex64 precision.
- For the matched RY/RZ/CNOT circuit on this CPU/NumPy runner, full realified state-vector simulation was slower than complex at every tested size. The realified/complex runtime ratios were `1.976x, 1.970x, 1.474x, 1.066x, 1.063x, 1.231x` for `N=4,8,12,16,18,20`.
- Gauss 3M consistently improved the general complex-gate realification over the standard 4M construction by `1.18x-2.12x`. At `N=16`, the 3M kernel was `0.623x` the complex-kernel time (about `1.60x` faster), but this advantage did not persist at `N=18` (`1.112x` complex time). This crossover is implementation/hardware specific and is not yet a generic QML speedup claim.
- End-to-end VQC inference, parameter-shift Jacobians, and training remained numerically equivalent, but realification was generally about `1.9x-2.0x` slower for `N=4-10` on this CPU implementation.

## Interpretation

The current CPU/NumPy evidence does **not** support a generic speed or storage advantage for exact realification. Its robust benefit is compatibility with real-valued computation while preserving the complex model exactly. The most promising computational direction is specialized real-only kernels such as 3M and accelerator-specific implementations, where individual complex operations can become cheaper even though the full CPU VQC is not yet faster.

Raw data are stored in the CSV/JSON files in this directory. The corresponding workflow artifact is `qreal-realification-results`, artifact ID `9044783807`, SHA-256 `be5f8534d9402c743f1caddbb4783a37f3e00042b7a6211c1128900a06fc46f6`.
