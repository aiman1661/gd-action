'''
Functions for solving the variatioal problem of minimising the action of a path via gradient descent.

author: @im@n
'''

import jax.numpy as jnp
from jax import jit, random
from functools import partial

# ------ Trivial Boundary Conditions Protototype ------

@jit
def fourier_basis_expansion(coeffs, t_array):
    k = coeffs.shape[0]
    T = t_array[-1] - t_array[0]
    basis = jnp.stack([jnp.sin((n + 1) * jnp.pi * t_array / T) for n in range(k)])
    return jnp.dot(coeffs, basis)

@partial(jit, static_argnames=('m', 'potential'))
def compute_action(x, t_array, m:float=1., potential=None):
    N = len(t_array)
    dt = (t_array[1] - t_array[0]) / (N-1)
    v = (x[1:] - x[:-1]) / dt  # velocity at each time step
    kinetic_term = 0.5 * m * jnp.sum(v**2)
    potential_term = jnp.sum(potential(x[:-1]))  # sum all contributions
    return (kinetic_term - potential_term) * dt

@partial(jit, static_argnames=('m', 'omega'))
def harmonic_potential(x, m:float=1., omega:float=jnp.pi):
    return 0.5 * m * omega**2 * x**2

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    time_array = jnp.linspace(0, 10, 100)
    n = 1  # number of Fourier coefficients
    key = random.PRNGKey(0)
    coeffs = random.uniform(key, (n,))  # random coefficients for Fourier expansion
    x = fourier_basis_expansion(coeffs, time_array)
    plt.plot(time_array, x, label='Fourier Basis Expansion')
    plt.show()