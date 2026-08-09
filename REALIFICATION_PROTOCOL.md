# Realification benchmark protocol

This study isolates exact realification from the quotient/QFIM experiments.

## Matched kernels

The complex and realified simulators use the same RY/RZ/CNOT circuit topology and specialized NumPy kernels. The realified backend stores `psi=x+iy` as two real arrays and never converts them to complex dtype during the circuit.

## Experiments

1. State-vector runtime and memory scaling at 4, 8, 12, 16, 18, and 20 qubits.
2. Float64/complex128 and float32/complex64 numerical-precision comparison.
3. General single-qubit complex gates using complex arithmetic, standard 4-real-multiply realification, and Gauss 3M realification.
4. End-to-end VQC inference, parameter-shift Jacobians, and matched training paths.

## Claim policy

Exact realification should preserve states, probabilities, gradients, and matched optimization paths within numerical precision. Storage is compared using actual array bytes. Runtime claims use medians from the same GitHub Actions runner. Raw CSV/JSON outputs are retained as workflow artifacts and committed to the repository after validation.

The CI study is intentionally CPU/NumPy-specific; hardware-general claims require separate accelerator measurements.
