# Quotient-Real QML prototype

This repository now contains two levels of real-valued QML simulation:

- `complex`: ordinary complex state-vector VQC.
- `structured_real`: exact global realification, storing `psi=x+iy` as `[x,y]`.
- `quotient_real`: explicit multipartite quotient-space construction with one local two-dimensional flag per subsystem and canonical even/odd flag representatives.
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

## Reproducible checks

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_equivalence.py test_quotient.py
python demo.py
python benchmark.py --output-dir results/baseline
python quotient_benchmark.py --output-dir results/quotient
```

The quotient tests verify:

- orthonormal canonical even/odd flag representatives;
- identification of locally redistributed phases after quotient projection;
- preservation of the canonical quotient subspace under local complex gates;
- state, probability, gradient, and optimization-path equivalence with complex QML.

The benchmark records runtime, explicit-versus-compressed storage, full-state coordinate error, and training-path error. Exact quotient realification is mathematically equivalent to complex QML, so accuracy or trainability gains are not expected without introducing a genuinely different quotient-aware parameterization or algorithm.

GitHub Actions runs all tests and benchmarks and uploads `results/` as a workflow artifact.
