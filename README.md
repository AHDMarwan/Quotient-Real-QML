# Quotient-Real QML prototype

This repository contains a small research prototype for comparing three VQC backends:

- `complex`: ordinary complex state-vector simulation.
- `structured_real`: exact realification, storing `psi=x+iy` as `[x,y]` and using the block representation of complex gates.
- `naive_real`: a real-amplitude baseline that replaces phase rotations by real rotations; it is not mathematically equivalent to the complex model.

## Reproducible checks

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_equivalence.py
python demo.py
python benchmark.py --output-dir results
```

The benchmark records inference runtime/state storage and a small multi-seed breast-cancer classification experiment. Complex and structured-real predictions are expected to agree to floating-point precision. Any runtime advantage is implementation/hardware dependent; exact realification alone does not reduce the real degrees of freedom of a state vector.

GitHub Actions runs the same checks and uploads `results/` as a workflow artifact.
