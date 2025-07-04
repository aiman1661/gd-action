'''
The Path Action class.

author: @im@n
'''

import jax.numpy as jnp
from jax import jit, value_and_grad, random
from functools import partial
import optax
import action_functions as funcs

class PathAction():

    def __init__(self, x, coeffs, omega=jnp.pi, m=1.0, t_a=0., t_b=1., N=20, potential=None, basis_ansatz=None):
        self.x = x
        self.m = m
        self.omega = omega
        self.time_array = jnp.linspace(t_a, t_b, N)
        self.potential = potential
        self.basis_ansatz = basis_ansatz
        self.coeffs = coeffs
        self.action = self.compute_action_in_x()
    
    def compute_action_in_coeffs(self):
        """
        Computes the action for a given path defined by coefficients of eigenbasis expansion.
        Used in gradient descent in Fourier space.
        """
        return funcs.action(
            self.x, 
            self.time_array, 
            m=self.m, 
            potential=self.potential, 
            basis_ansatz=self.basis_ansatz, 
            coeffs=self.coeffs
        )
 