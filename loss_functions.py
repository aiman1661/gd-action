import jax.numpy as jnp
import jax
from jax import jit
from functools import partial
from action_functions import action, harmonic_potential
from scipy.integrate import simpson

def normalisation_constraint(x_samples, t_array):
    """
    Computes the normalisation constraint for a path, ensuring that the integral of the square of the path
    over the time interval is equal to 1.
    Args:
        x_samples (array): Path points at discrete time steps.
        t_array (array): Array of time steps.
    Returns:
        float: The value of the normalisation constraint.
    """
    tau = t_array[-1]
    integrand = x_samples ** 2
    integral = simpson(integrand, t_array)
    return ((1 / tau) * integral - 1.0) ** 2 

# ------ loss functions ------

def harmonic_loss_function(x_samples, t_array, m=1.0, omega=jnp.pi, potential=harmonic_potential):
    """
    Computes the loss function for a path, which is the action minus the normalisation constraint.
    Args:
        x_samples (array): Path points at discrete time steps.
        t_array (array): Array of time steps.
        m (float): Mass of the particle.
        omega (float): Frequency for harmonic potential.
        potential (callable): Potential energy function.
    Returns:
        float: The value of the loss function.
    """
    action_value = action(x_samples, t_array, m=m, potential=potential)
    norm_constraint = normalisation_constraint(x_samples, t_array)
    return action_value + norm_constraint