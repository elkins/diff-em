# ❄️ diff-em

**diff-em** provides differentiable kernels for fitting atomic structures into Cryo-EM density maps using JAX.

## Quick Start

```python
import jax.numpy as jnp
from diff_em.kernels import simulate_density, cross_correlation

# Atomic coordinates
coords = jnp.array([[10.0, 10.0, 10.0]])

# 3D Grid (small 20x20x20 example)
x = jnp.linspace(0, 20, 20)
grid = jnp.stack(jnp.meshgrid(x, x, x, indexing='ij'), axis=-1)

# Simulate map
em_map = simulate_density(coords, grid)
print(em_map.shape) # (20, 20, 20)
```
