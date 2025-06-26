'''
Functions for solving the variatioal problem of minimising the action of a path via gradient descent.

author: @im@n
'''

import jax.numpy as jnp
from jax import jit, value_and_grad, random
from functools import partial
import optax

def fourier_basis_expansion(t_array, coeffs):
    k = coeffs.shape[0]
    T = t_array[-1] - t_array[0]
    basis = jnp.stack([jnp.sin((n + 1) * jnp.pi * t_array / T) for n in range(k)])
    return jnp.dot(coeffs, basis)

def action_coeff_fn(coeffs, t_a:float=0., t_b:float=1., m:float=1., N:int=20, potential=None):
    t_array = jnp.linspace(t_a, t_b, N)
    x = fourier_basis_expansion(t_array, coeffs)
    return action(x, t_a=t_a, t_b=t_b, m=m, N=N, potential=potential)

@partial(jit, static_argnames=('t_a', 't_b', 'm', 'N', 'potential'))
def action(x, t_a:float=0., t_b:float=1., m:float=1., N:int=20, potential=None):
    """
    Computes the action for a given path x(t) between times t_a and t_b.
    Args:
        x (array): Path points at discrete time steps.
        t_a (float): Initial time.
        t_b (float): Final time.
        m (float): Mass of the particle.
        N (int): Number of discrete time steps.
        potential (callable): Function to compute the potential energy.
    """
    dt = (t_b - t_a) / (N-1)
    v = (x[1:] - x[:-1]) / dt  # velocity at each time step
    kinetic_term = 0.5 * m * jnp.sum(v**2)
    potential_term = jnp.sum(potential(x[:-1]))  # sum all contributions
    return (kinetic_term - potential_term) * dt

@partial(jit, static_argnames=('m', 'omega'))
def harmonic_potential(x, m:float=1., omega:float=1.):
    return 0.5 * m * omega**2 * x**2

def gradient_descent(function, x, learning_rate: float = 0.001, steps: int = 5000):
    """
    Performs gradient descent using optax.adam to minimise the given function.

    Args:
        function (callable): The function to minimise.
        x (jnp.ndarray): Initial guess.
        learning_rate (float): Learning rate.
        steps (int): Number of gradient steps.

    Returns:
        x_opt (jnp.ndarray): The optimised variable.
        action_list (jnp.ndarray): Action values over iterations.
        grad_norm_list (jnp.ndarray): Gradient norms over iterations.
    """
    optimiser = optax.adam(learning_rate)
    opt_state = optimiser.init(x)

    action_list = []  
    grad_norm_list = []

    for i in range(steps):
        action_val, gradient = value_and_grad(function)(x)
        grad_norm = jnp.linalg.norm(gradient)

        updates, opt_state = optimiser.update(gradient, opt_state, x)
        x = optax.apply_updates(x, updates)

        action_list.append(float(action_val))  
        grad_norm_list.append(float(grad_norm))

        if i % 100 == 0:
            print(f"Step {i}, action: {action_val:.6f}, grad_norm: {grad_norm:.6f}")

        if grad_norm < 1e-3:
            print(f"Converged at step {i}.")
            print("Final action:", action_val)
            return x, jnp.array(action_list), jnp.array(grad_norm_list)

    print("Did not converge within the specified number of steps.")
    return x, jnp.array(action_list), jnp.array(grad_norm_list)

# @partial(jit, static_argnames=('function'))
# def descent_step(function, x, learning_rate:float=0.001):

#     optimiser = optax.adam(learning_rate=learning_rate)
#     opt_state = optimiser.init(x)

#     action_val, gradient = value_and_grad(function)(x)
#     grad_norm = jnp.linalg.norm(gradient)

#     # set gradient at boundary to be zero 
#     gradient = gradient.at[0].set(0)  
#     gradient = gradient.at[-1].set(0)

#     updates, opt_state = optimiser.update(gradient, opt_state)
#     x = optax.apply_updates(x, updates)

#     return x, action_val, gradient, grad_norm

# def gradient_descent(function, x, learning_rate:float=0.001, steps:int=5000):

#     action_list = []  
#     grad_norm_list = []
#     current_lr = learning_rate

#     for i in range(steps):
#         x, action_val, _, grad_norm = descent_step(function, x, current_lr)
        
#         action_list.append(float(action_val))  
#         grad_norm_list.append(float(grad_norm))
        
#         if i % 100 == 0:
#             print(f"Step {i}, action: {action_val:.6f}, grad_norm: {grad_norm:.6f}")
        
#         if grad_norm < 1e-3:
#             print(f"Converged at step {i}.")
#             print("Final action:", action_val)
#             break
    
#     print("Did not converge within the specified number of steps.")
#     return x, jnp.array(action_list), jnp.array(grad_norm_list)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    time_array = jnp.linspace(0, 10, 100)
    n = 1  # number of Fourier coefficients
    key = random.PRNGKey(0)
    coeffs = random.uniform(key, (n,))  # random coefficients for Fourier expansion
    x = fourier_basis_expansion(time_array, coeffs)
    plt.plot(time_array, x, label='Fourier Basis Expansion')
    plt.show()