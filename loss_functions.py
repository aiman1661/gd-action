'''
Loss functions for path optimisation in JAX.

author: @im@n
'''

import jax.numpy as jnp
import jax
from jax import jit, scipy
from functools import partial
from action_functions import compute_action, harmonic_potential, fourier_basis_expansion


def normalisation_constraint(a_samples, t_array, basis_ansatz):
    x = basis_ansatz(a_samples, t_array) # Expand coefficients to path points

    tau = t_array[-1]
    integrand = x ** 2
    integral = scipy.integrate.trapezoid(integrand, t_array)
    return ((1 / tau) * integral - 1.0) ** 2 

def action_from_coeffs(coeffs, t_array, m=1., potential=None):
    x_array = fourier_basis_expansion(coeffs, t_array)
    return compute_action(x_array, t_array, m, potential)

# ------ loss functions ------

@partial(jit, static_argnames=('m', 'potential'))
def convergence_check(coeffs, t_array, m, potential):
    return action_from_coeffs(coeffs, t_array, m, potential)

@partial(jit, static_argnames=('m', 'potential'))
def harmonic_loss_function(a_samples, t_array, alpha, m=1.0, potential=harmonic_potential):
    action_value = action_from_coeffs(a_samples, t_array, m, potential)
    norm_penalty = normalisation_constraint(a_samples, t_array, fourier_basis_expansion)
    return action_value + alpha * norm_penalty
