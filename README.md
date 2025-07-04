# Variational Path Optimisation via Gradient Descent

An independent computational physics project implementing the Principle of Least Action through numerical optimisation techniques using JAX and gradient descent methods.

## Overview

This project tackles the classical problem of finding particle trajectories that minimise the action functional—a cornerstone of variational mechanics. Rather than solving the Euler-Lagrange equation directly, we discretise the action and use modern automatic differentiation tools to find optimal paths through gradient descent.

## Project Status

🚧 **Work in Progress** 🚧

This project is currently under active development. The core harmonic oscillator implementation is semi-functional (core body of the code is still being figured out) and several advanced features are still being developed.

## Key Features

### Current Implementation
- **JAX-based automatic differentiation** for efficient gradient computation
- **Fourier basis expansion** for representing paths with boundary conditions
- **Harmonic oscillator solver** as proof-of-concept
- **Normalisation constraints** using average power-type norm
- **Adam optimiser integration** via Optax for stable convergence
- **Interactive Jupyter notebook** for visualisation and experimentation

### Numerical Methods
- Discrete action computation using finite differences
- Gradient descent optimisation in Fourier coefficient space
- Soft constraint implementation for path normalisation
- Convergence monitoring with action and gradient norm tracking

## Project Structure

```
├── action_functions.py     # Core action computation and optimisation
├── action_class.py        # PathAction class for organised computation
├── loss_functions.py      # Loss functions with constraints
├── action_notebook.ipynb  # Simulation and visualisation
└── Coding_Project_Summer_2025.pdf  # Detailed physics documentation
```

## Core Components

### Action Computation
The `action()` function computes the discretised action using:
- Kinetic energy: `½m Σ(v²)`  
- Potential energy: `Σ V(x)`
- Trapezoidal integration over time steps

### Fourier Basis Expansion
Paths are represented as:
```python
x(t) = Σ aₙ sin(nπt/T)
```
This automatically satisfies zero boundary conditions and provides a compact parametrisation.

### Optimization Pipeline
1. Initialise random Fourier coefficients
2. Compute action + normalisation constraint
3. Backpropagate gradients through JAX
4. Update coefficients using Adam optimiser
5. Monitor convergence via gradient norms

## Usage

### Basic Example
```python
import jax.numpy as jnp
from action_functions import gradient_descent, fourier_basis_expansion
from loss_functions import harmonic_loss_function

# Set up time grid and initial coefficients
t_array = jnp.linspace(0, 1, 100)
coeffs_init = jnp.array([1.0, 0.5, 0.1])  # 3 Fourier modes

# Define loss function
loss_fn = lambda c: harmonic_loss_function(
    fourier_basis_expansion(t_array, c), t_array
)

# Optimise
coeffs_opt, losses, grad_norms = gradient_descent(
    loss_fn, coeffs_init, learning_rate=0.01, steps=1000
)
```

### Interactive Analysis
Run the Jupyter notebook for:
- Parameter exploration and sensitivity analysis
- Visualisation of initial and finalised paths
- Comparison of different initialisation strategies
- Convergence diagnostics and loss landscapes

## Mathematical Framework

### Normalisation Constraint
The project implements an innovative normalisation using average power:
```
lim[τ→∞] (1/τ) ∫[0 to τ] x(t)² dt = 1
```
This preserves scale invariance while respecting causality.

### Discrete Action
The continuous action is approximated as:
```
S ≈ dt × [½m Σᵢ((xᵢ₊₁-xᵢ)/dt)² - Σᵢ V(xᵢ)]
```

## Dependencies

- **JAX**: Automatic differentiation and JIT compilation
- **Optax**: Advanced optimisation algorithms  
- **SciPy**: Integration routines for constraints
- **NumPy**: Array operations
- **Matplotlib**: Visualisation (in notebook)

## Installation

```bash
pip install jax jaxlib optax scipy numpy matplotlib jupyter
```

## Future Directions

This project aims to extend beyond the harmonic oscillator to more complex systems:

- **Anharmonic corrections**: Adding cubic corrections
- **Spectral decompositions**: Utilising other special functions such as Chebyshev
- **Joint optimisation**: Simultaneously optimising both path and system frequency
- **Inverse problems**: Applications in parameter estimation for classical mechanics
- **Multi-dimensional paths**: Extension to higher-dimensional configuration spaces

## Contributing

This is an active personal research project. Contributions, suggestions, and discussions are welcome! Areas of particular interest:

- Advanced basis functions beyond Fourier series
- Alternative constraint formulations  
- Optimisation algorithm comparisons
- Physical system extensions
- Numerical stability improvements

## References

- JAX Documentation: https://jax.readthedocs.io/
- Optax Documentation: https://optax.readthedocs.io/

## Author

**Aiman Mohd Yuzmanizeil**  
Summer 2025

---

*"The path that nature actually takes is the one for which the action is least."* - Principle of Least Action