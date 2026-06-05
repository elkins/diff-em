# Cryo-EM Theory

## Density Simulation

Atomic structures are often represented in Cryo-EM fitting as a sum of 3D Gaussians, representing the electrostatic potential or electron density. For an atom at position $\vec{r}_i$, the density contribution at grid point $\vec{x}$ is:

$$\rho_i(\vec{x}) = \exp\left( -\frac{|\vec{x} - \vec{r}_i|^2}{2\sigma^2} \right)$$

The total density map is the sum over all $N$ atoms:

$$\rho_{total}(\vec{x}) = \sum_{i=1}^N \rho_i(\vec{x})$$

## Cross-Correlation (CC)

The standard metric for comparing two EM maps $\rho_A$ and $\rho_B$ is the Pearson correlation coefficient:

$$CC = \frac{\sum (\rho_A - \bar{\rho}_A)(\rho_B - \bar{\rho}_B)}{\sqrt{\sum (\rho_A - \bar{\rho}_A)^2 \sum (\rho_B - \bar{\rho}_B)^2}}$$

In **diff-em**, this function is fully differentiable with respect to the atomic coordinates $\vec{r}_i$, allowing for gradient-based structural refinement.
