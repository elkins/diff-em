import jax.numpy as jnp


def simulate_density(
    coords: jnp.ndarray,
    grid_coords: jnp.ndarray,
    sigma: float = 1.0,
) -> jnp.ndarray:
    """
    Differentiable simulation of 3D density from atomic coordinates.
    Represent each atom as a 3D Gaussian.

    Args:
        coords: (N, 3) atomic coordinates.
        grid_coords: (D, H, W, 3) coordinates of the 3D grid points.
        sigma: Width (standard deviation) of the Gaussian blobs.

    Returns:
        3D density map (D, H, W).
    """
    # Reshape for broadcasting:
    # grid: (1, D, H, W, 3)
    # coords: (N, 1, 1, 1, 3)
    diff = grid_coords[None, ...] - coords[:, None, None, None, :]
    dist_sq = jnp.sum(diff**2, axis=-1)

    # Gaussian kernel: exp(-r^2 / (2*sigma^2))
    # We ignore the normalization constant as CC is scale-invariant
    gaussians = jnp.exp(-dist_sq / (2 * sigma**2))

    # Sum contributions from all atoms
    return jnp.sum(gaussians, axis=0)


def cross_correlation(
    map_a: jnp.ndarray,
    map_b: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the Pearson correlation coefficient between two 3D maps.
    This is a standard loss function for Cryo-EM fitting.

    Args:
        map_a: (D, H, W) First density map.
        map_b: (D, H, W) Second density map.

    Returns:
        Scalar correlation coefficient in [-1, 1].
    """
    # Flatten maps
    a = map_a.flatten()
    b = map_b.flatten()

    # Center maps
    a_centered = a - jnp.mean(a)
    b_centered = b - jnp.mean(b)

    # Compute CC
    numerator = jnp.sum(a_centered * b_centered)
    denominator = jnp.sqrt(jnp.sum(a_centered**2) * jnp.sum(b_centered**2) + 1e-9)

    return numerator / denominator
