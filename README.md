# Quotient-Real QML prototype

This repository now contains three research layers:

- `complex`: ordinary complex state-vector VQC.
- `structured_real`: exact global realification, storing `psi=x+iy` as `[x,y]`.
- `quotient_real`: explicit multipartite quotient-space construction with one local two-dimensional flag per subsystem and canonical even/odd flag representatives.
- `quotient_gauge`: a gauge-audit layer that makes local flag-phase redundancies explicit and then prunes them.
- `naive_real`: real-amplitude baseline; it is not mathematically equivalent to the complex model.

## Quotient-space implementation

`quotient_real.py` follows Barrios Hita et al., *Phys. Rev. Lett.* **136**, 240202 (2026), arXiv:2503.17307v2, in particular the canonical flag states and state map in Supplement Eqs. S37-S38.

For `N` qubit subsystems, the explicit canonical representative is

```text
R(psi) = Re(psi) x |even_N> + Im(psi) x |odd_N>.
```

The raw local-flag tensor space has dimension `4^N`, but the quotient retains only the two physical flag classes `|even_N>` and `|odd_N>`. Therefore the quotient-coordinate dimension is `2^(N+1)`, equal to ordinary exact realification. The explicit implementation is intentionally kept to test the multipartite composition rule, phase redistribution, locality, and numerical equivalence; it is not expected to be memory efficient.

Primary source: https://doi.org/10.1103/4k13-sdjh

Preprint with supplement: https://arxiv.org/abs/2503.17307

## Gauge-aware parameter pruning

`quotient_gauge.py` tests the next research hypothesis: local flag phases should not be counted as independent trainable capacity.

On the canonical quotient subspace, applying local flag rotations with angles `alpha_q` gives only the total complex phase

```text
R(alpha_1) ... R(alpha_N) R(psi) = R(exp(i * sum_q alpha_q) psi).
```

Therefore `G` local flag-phase parameters have state-vector tangent rank one: only their sum survives as a vector phase. In projective quantum geometry even that final global phase is unobservable, so the quantum Fisher information in the pure gauge block has rank zero. The gauge-fixed model can delete all such parameters without changing measurement probabilities.

The benchmark deliberately augments each variational layer with one local flag phase per qubit. For the current RY/RZ ansatz this changes the count from `2*N*L` physical parameters to `3*N*L` augmented parameters. Pruning the gauge block therefore removes one third of the parameters and one third of the parameter-shift circuit evaluations of that deliberately redundant model, while preserving predictions exactly. This is a parameter-efficiency result, not an accuracy advantage over the already gauge-fixed baseline.

Related parameter-redundancy / quantum-geometry references:

- Haug, Bharti, Kim, *Capacity and quantum geometry of parametrized quantum circuits*, arXiv:2102.01659.
- Funcke et al., *Dimensional Expressivity Analysis of Parametric Quantum Circuits*, arXiv:2011.03532.

## Reproducible checks

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_equivalence.py test_quotient.py test_gauge.py
python demo.py
python benchmark.py --output-dir results/baseline
python quotient_benchmark.py --output-dir results/quotient
python gauge_benchmark.py --output-dir results/gauge
```

The quotient and gauge tests verify:

- orthonormal canonical even/odd flag representatives;
- identification of locally redistributed phases after quotient projection;
- preservation of the canonical quotient subspace under local complex gates;
- state, probability, gradient, and optimization-path equivalence with complex QML;
- collapse of many local flag phases to one total vector phase;
- zero observable gradients along gauge parameters;
- rank-one Euclidean state tangent but rank-zero projective QFI for the pure gauge block;
- exact prediction preservation after gauge pruning.

GitHub Actions runs all tests and benchmarks and uploads `results/` as a workflow artifact.
