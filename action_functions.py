'''
Functions to compute the action and perform gradient descent.

author: @im@n
'''

import jax.numpy as jnp
from jax import jit, value_and_grad
from functools import partial
import optax

@partial(jit, static_argnames=('t_a', 't_b', 'm', 'N', 'potential'))
def action(x, t_a:float=0, t_b:float=10, m:float=1., N:int=100, potential=None):
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

@partial(jit, static_argnames=('function'))
def descent_step(function, x, learning_rate:float=0.001):

    optimiser = optax.adam(learning_rate=learning_rate)
    opt_state = optimiser.init(x)

    action_val, gradient = value_and_grad(function)(x)
    grad_norm = jnp.linalg.norm(gradient)

    # set gradient at boundary to be zero 
    gradient = gradient.at[0].set(0)  
    gradient = gradient.at[-1].set(0)

    updates, opt_state = optimiser.update(gradient, opt_state)
    x = optax.apply_updates(x, updates)

    return x, action_val, gradient, grad_norm

#@partial(jit, static_argnames=('function', 'learning_rate', 'steps'))
def gradient_descent(function, x, learning_rate:float=0.001, steps:int=5000):

    action_list = []  
    grad_norm_list = []
    current_lr = learning_rate

    for i in range(steps):
        x, action_val, _, grad_norm = descent_step(function, x, current_lr)
        
        action_list.append(float(action_val))  
        grad_norm_list.append(float(grad_norm))
        
        if i % 100 == 0:
            print(f"Step {i}, action: {action_val:.6f}, grad_norm: {grad_norm:.6f}")
        
        if grad_norm < 1e-3:
            print(f"Converged at step {i}.")
            print("Final action:", action_val)
            break
    
    print("Did not converge within the specified number of steps.")
    return x, jnp.array(action_list), jnp.array(grad_norm_list)