# ❄️ diff-em: Differentiable Cryo-EM Fitting in JAX

[![Tests](https://github.com/elkins/diff-em/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/diff-em/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/backend-JAX-9cf.svg)](https://github.com/google/jax)

**diff-em** provides high-performance, auto-differentiable kernels for fitting atomic structures into Cryo-EM density maps. Built on **JAX**, it enables gradient-based optimization of coordinates directly against 3D experimental data.

---

## 🎯 Features

- **Gaussian Mixture Volumes:** Represent atomic models as differentiable 3D density maps using sum-of-Gaussians (electrostatic potential approximation).
- **Cross-Correlation Kernels:** Differentiable computation of map-to-model correlation coefficients (CC) for structural refinement.
- **Multi-Resolution Fitting:** Support for tunable Gaussian widths (B-factors) to handle low-to-medium resolution maps.
- **Hardware Acceleration:** Optimized for GPU/TPU execution via XLA, enabling the fitting of large complexes in seconds.

---

## 🏗️ Technical Architecture

- **Backend:** JAX (XLA-compiled).
- **Physics:** 3D Gaussian placement with B-factor smoothing.
- **Optimization:** Pure JAX implementation compatible with `optax` for high-dimensional gradient descent.

---

## 🧪 Scientific Validation

- **Density Parity:** Simulated densities are verified against standard EM map generation tools (e.g., `gemmi` or `ChimeraX`).
- **CC Gradient Stability:** Verified numerically stable gradients for structural refinement in the presence of noise.
- **Resolution Limits:** Benchmarked against known high-resolution and low-resolution experimental maps.

---

## 🚀 Roadmap

- [x] Differentiable 3D Gaussian density kernels.
- [x] Cross-correlation (CC) loss functions.
- [ ] Integration with MRC map loaders.
- [ ] Automated multi-resolution refinement schedules.

---

## 🔗 Related Projects

diff-em is part of the **differentiable biophysics** ecosystem:

- [diff-biophys](https://github.com/elkins/diff-biophys) — Core differentiable biophysics engine.
- [diff-hdx](https://github.com/elkins/diff-hdx) — Differentiable HDX-MS prediction.
- [synth-cryo-em](https://github.com/elkins/synth-cryo-em) — Cryo-EM simulation.

---

## 📖 Citation

```bibtex
@software{diff_em,
  author  = {Elkins, George},
  title   = {diff-em: Differentiable Cryo-EM map fitting in JAX},
  year    = {2026},
  url     = {https://github.com/elkins/diff-em},
  version = {0.1.0}
}
```

## ⚖️ License

MIT
