import jax.numpy as jnp
from jax import jit, grad
from functools import partial

def zero_potential(x):
    return 0.

@partial(jit, static_argnames=('t_a', 't_b', 'm', 'N', 'potential'))
def action(x, t_a:float=0, t_b:float=10, m:float=1., N:int=50, potential=zero_potential):
    delta_t = (t_b - t_a) / N
    kinetic_term = (1/2) * m * jnp.sum((((x[1:] - x[:-1])/delta_t))**2) # sum all contributions
    potential_term = jnp.sum(potential(x))  # sum all contributions
    return (kinetic_term + potential_term) * delta_t

@partial(jit, static_argnames=('k'))
def harmonic_potential(x, m:float=1., omega:float=1.):
    return (1/2) * m * (omega**2) * jnp.sum(x**2)


@partial(jit, static_argnames=('function', 'learning_rate'))
def descent_step(function, x, learning_rate:float=0.1):
    gradient = grad(function)(x)

    # set gradient at boundary to be zero 
    gradient = gradient.at[0].set(0)  
    gradient = gradient.at[-1].set(0)

    return x - learning_rate * gradient

#@partial(jit, static_argnames=('function', 'learning_rate', 'steps'))
def gradient_descent(function, x, learning_rate:float=0.1, steps:int=2000):
    
    action_array = jnp.zeros((steps,))
    grad_norm_array = jnp.zeros((steps,))
    
    for i in range(steps):
        x = descent_step(function, x, learning_rate)  
        action_array = action_array.at[i].set(function(x))
        grad_norm_array = grad_norm_array.at[i].set(jnp.linalg.norm(grad(function)(x)))

        if i % 100 == 0:
            print(f"Step {i}, action: {function(x)}")

        if jnp.linalg.norm(grad(function)(x)) < 1e-3:
            print(f"Converged at step {i}.")
            print("Final action:", function(x))
            break

    else:
        print("Did not converge within the specified number of steps.")

    return x, action_array, grad_norm_array