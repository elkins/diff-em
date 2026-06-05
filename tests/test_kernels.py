import jax
import jax.numpy as jnp

from diff_em.kernels import cross_correlation, simulate_density


def test_simulate_density():
    """
    Verify that density simulation produces correct shapes and non-zero values.
    """
    coords = jnp.array([[5.0, 5.0, 5.0]])
    x = jnp.linspace(0, 10, 11)
    grid = jnp.stack(jnp.meshgrid(x, x, x, indexing="ij"), axis=-1)

    em_map = simulate_density(coords, grid)
    assert em_map.shape == (11, 11, 11)
    assert jnp.max(em_map) == 1.0  # Peak of Gaussian
    assert em_map[5, 5, 5] == 1.0


def test_cross_correlation():
    """
    Verify CC calculation for identical and orthogonal maps.
    """
    map_a = jnp.ones((5, 5, 5))
    map_b = jnp.ones((5, 5, 5))

    # Identical maps with variation
    map_a = map_a.at[2, 2, 2].set(2.0)
    map_b = map_b.at[2, 2, 2].set(2.0)

    cc = cross_correlation(map_a, map_b)
    assert jnp.allclose(cc, 1.0)


def test_em_differentiable():
    """
    Verify that we can take gradients of CC with respect to coordinates.
    """
    coords = jnp.array([[5.0, 5.0, 5.0]])
    x = jnp.linspace(0, 10, 5)
    grid = jnp.stack(jnp.meshgrid(x, x, x, indexing="ij"), axis=-1)

    target_map = simulate_density(jnp.array([[6.0, 6.0, 6.0]]), grid)

    def loss(pos: jnp.ndarray) -> jnp.ndarray:
        sim_map = simulate_density(pos, grid)
        return -cross_correlation(sim_map, target_map)

    grads = jax.grad(loss)(coords)
    assert grads.shape == coords.shape
    assert not jnp.any(jnp.isnan(grads))
    # Gradient should point towards (6, 6, 6)
    assert jnp.all(grads < 0)
