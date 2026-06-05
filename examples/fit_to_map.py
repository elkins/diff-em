import jax
import jax.numpy as jnp

from diff_em.kernels import cross_correlation, simulate_density


def main():
    # 1. Define a simple grid
    x = jnp.linspace(0, 10, 20)
    grid = jnp.stack(jnp.meshgrid(x, x, x, indexing="ij"), axis=-1)

    # 2. Target structure (Ground Truth)
    target_coords = jnp.array([[5.0, 5.0, 5.0], [7.0, 7.0, 7.0]])
    target_map = simulate_density(target_coords, grid)

    # 3. Initial "guesses" (offset by 0.5A)
    initial_coords = target_coords + 0.5

    # 4. Optimization Loop (Manual Gradient Descent)
    lr = 0.1
    coords = initial_coords

    print("Starting structural refinement...")
    for i in range(10):

        def loss_fn(p):
            sim_map = simulate_density(p, grid)
            return -cross_correlation(sim_map, target_map)

        loss, grads = jax.value_and_grad(loss_fn)(coords)
        coords = coords - lr * grads
        print(f"Step {i}: CC = {-loss:.4f}")

    print(f"\nFinal refined coordinates:\n{coords}")
    print(f"Target coordinates:\n{target_coords}")


if __name__ == "__main__":
    main()
