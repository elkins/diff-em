import jax
import jax.numpy as jnp

from diff_em.kernels import cross_correlation, simulate_density


def test_simulate_density():
    """
    Verify that density simulation produces correct shapes and peak values.
    """
    coords = jnp.array([[5.0, 5.0, 5.0]])
    x = jnp.linspace(0, 10, 11)
    grid = jnp.stack(jnp.meshgrid(x, x, x, indexing="ij"), axis=-1)

    em_map = simulate_density(coords, grid)
    assert em_map.shape == (11, 11, 11)

    # The atom is at [5.0, 5.0, 5.0]; linspace(0,10,11) has a node at exactly 5.0,
    # so the peak equals exp(0) = 1.0.  Use atol rather than == so the test is
    # robust to slight grid shifts.
    assert jnp.allclose(jnp.max(em_map), 1.0, atol=1e-6)
    # Peak should be located at voxel [5, 5, 5]
    peak_idx = jnp.unravel_index(jnp.argmax(em_map), em_map.shape)
    assert peak_idx == (5, 5, 5)


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

def test_simulate_density_off_grid_atom():
    """
    Verify density is computed correctly when the atom does NOT fall on a grid node.

    The old test used an atom exactly at a grid node so exp(0)=1.0 exactly.
    This test places the atom between grid nodes and verifies the expected
    Gaussian value at the nearest grid point.

    This would have passed under both the old broadcast and new scan implementations,
    but it makes the test suite robust to any future regression in Gaussian evaluation.
    """
    # Atom at [4.5, 4.5, 4.5] — between two grid nodes at 4 and 5
    coords = jnp.array([[4.5, 4.5, 4.5]])
    sigma = 2.0
    x = jnp.linspace(0, 10, 11)  # nodes at 0,1,...,10
    grid = jnp.stack(jnp.meshgrid(x, x, x, indexing="ij"), axis=-1)

    em_map = simulate_density(coords, grid, sigma=sigma)
    assert em_map.shape == (11, 11, 11)

    # Nearest grid node is [4,4,4] at distance sqrt(3*(0.5)^2) = sqrt(0.75)
    dist_sq = 3 * (0.5 ** 2)
    expected_val = float(jnp.exp(-dist_sq / (2 * sigma**2)))
    assert jnp.allclose(em_map[4, 4, 4], expected_val, atol=1e-5), (
        f"Expected Gaussian value {expected_val:.6f} at nearest node, got {em_map[4,4,4]:.6f}"
    )


def test_simulate_density_additivity():
    """
    Density must be additive over atoms: sim(A) + sim(B) == sim(A+B).

    This is a mathematical invariant of the sum-of-Gaussians model that must
    hold regardless of whether the implementation uses broadcast, scan, or
    any other batching strategy.  A regression in the scan accumulation would
    break this property.
    """
    x = jnp.linspace(0, 10, 11)
    grid = jnp.stack(jnp.meshgrid(x, x, x, indexing="ij"), axis=-1)

    atom_a = jnp.array([[3.0, 3.0, 3.0]])
    atom_b = jnp.array([[7.0, 7.0, 7.0]])
    both   = jnp.concatenate([atom_a, atom_b], axis=0)

    map_a   = simulate_density(atom_a, grid, sigma=1.5)
    map_b   = simulate_density(atom_b, grid, sigma=1.5)
    map_ab  = simulate_density(both,   grid, sigma=1.5)

    assert jnp.allclose(map_a + map_b, map_ab, atol=1e-5), (
        "Density must be additive: simulate(A) + simulate(B) == simulate(A+B)"
    )


def test_simulate_density_sigma_width():
    """
    A larger sigma must produce a broader, lower-peak Gaussian.
    Verify the expected peak value exp(-d^2 / 2*sigma^2) at a nearby grid node.
    """
    coords = jnp.array([[5.0, 5.0, 5.0]])
    x = jnp.linspace(0, 10, 11)
    grid = jnp.stack(jnp.meshgrid(x, x, x, indexing="ij"), axis=-1)

    map_narrow = simulate_density(coords, grid, sigma=0.5)
    map_wide   = simulate_density(coords, grid, sigma=3.0)

    # Value one voxel away (1 A) from the atom centre
    val_narrow_1A = map_narrow[5, 5, 6]  # 1 A away
    val_wide_1A   = map_wide[5, 5, 6]

    # Wider sigma → higher value at 1 A offset (the tail is fatter)
    assert val_wide_1A > val_narrow_1A, (
        "Wider sigma must give higher density away from centre (broader Gaussian)"
    )

    # Analytical check
    expected_narrow = float(jnp.exp(-1.0 / (2 * 0.5**2)))  # exp(-2)
    expected_wide   = float(jnp.exp(-1.0 / (2 * 3.0**2)))  # exp(-1/18)
    assert jnp.allclose(map_narrow[5, 5, 6], expected_narrow, atol=1e-5)
    assert jnp.allclose(map_wide[5, 5, 6],   expected_wide,   atol=1e-5)


def test_cross_correlation_anticorrelated():
    """
    Anti-correlated maps should give CC close to -1.
    """
    base = jnp.zeros((5, 5, 5))
    map_a = base.at[2, 2, 2].set(1.0)   # peak at centre
    map_b = base.at[2, 2, 2].set(-1.0)  # trough at centre

    cc = cross_correlation(map_a, map_b)
    assert jnp.allclose(cc, -1.0, atol=1e-5), (
        "Anti-correlated maps must give CC = -1"
    )
