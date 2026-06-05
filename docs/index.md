# ❄️ diff-em

**diff-em** provides differentiable kernels for fitting atomic structures into Cryo-EM density maps using JAX.

## Quick Start

```python
import jax
import jax.numpy as jnp
from diff_em import simulate_density, cross_correlation

# 3D grid (20×20×20, 1 Å spacing)
x = jnp.linspace(0, 20, 20)
grid = jnp.stack(jnp.meshgrid(x, x, x, indexing='ij'), axis=-1)

# Simulate a density map for a two-atom structure
coords = jnp.array([[10.0, 10.0, 10.0], [14.0, 10.0, 10.0]])
em_map = simulate_density(coords, grid)
print(em_map.shape)  # (20, 20, 20)

# Gradient of CC w.r.t. coordinates — drives coordinates toward a target map
target_map = simulate_density(coords + 0.5, grid)
loss = lambda c: -cross_correlation(simulate_density(c, grid), target_map)
grads = jax.grad(loss)(coords)
print(grads)  # points from current position toward target
```
