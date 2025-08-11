from dataclasses import dataclass
import sympy
from typing import List, Any
import jax.numpy as jnp

@dataclass
class SymbolicODE:
    """Holds the symbolic representation of an ODE system and its derivatives."""
    expressions: sympy.Matrix
    states: List[Any]
    params: List[sympy.Symbol]
    time: sympy.Symbol
    n_states: int
    n_params: int
    J_y: sympy.Matrix
    J_p: sympy.Matrix
    dJ_y_dp_list: List[Any]
    dJ_p_dp_list: List[Any]
    

def generate_aug_dynamcis_ode_sympy(symbolic_ode:SymbolicODE):
    # Placeholders for sensitivities
    S = sympy.MatrixSymbol("S", symbolic_ode.n_states, symbolic_ode.n_params)
    # H is a tensor (n_states x n_params x n_params). We represent it
    # as a list of matrices, where H_k = d(S)/dp_k.
    H_list = [sympy.MatrixSymbol(f"H_{k}", symbolic_ode.n_states, symbolic_ode.n_params) 
                for k in range(symbolic_ode.n_params)]

    # Dynamics for y
    dydt = symbolic_ode.expressions
    # Dynamics for S
    dSdt = symbolic_ode.J_y * S + symbolic_ode.J_p
    

   # Dynamics for H
    dHdt_list = []
    for k in range(symbolic_ode.n_params):
        H_k = H_list[k]
        dJp_dpk = symbolic_ode.dJ_p_dp_list[k]
        
        # 1. Get the explicit part of the derivative
        dJy_dpk_explicit = symbolic_ode.dJ_y_dp_list[k]
        
        # 2. Calculate the implicit (chain rule) part of the derivative
        dJy_dpk_implicit = sympy.zeros(symbolic_ode.n_states, symbolic_ode.n_states)
        for j in range(symbolic_ode.n_states):
            dJ_y_dyj = sympy.diff(symbolic_ode.J_y, symbolic_ode.states[j])
            dJy_dpk_implicit += dJ_y_dyj * S[j, k]
        
        # 3. Combine them to get the total derivative
        total_dJy_dpk = dJy_dpk_explicit + dJy_dpk_implicit
        
        # 4. Use the total derivative in the final expression
        dHk_dt = symbolic_ode.J_y * H_k + total_dJy_dpk * S + dJp_dpk
        dHdt_list.append(dHk_dt)
    
    # Flatten all symbolic components for the function signature
    aug_states_sym = (list(symbolic_ode.states) + 
                        list(S) + 
                        [item for sublist in H_list for item in sublist])
                        
    aug_dynamics_sym = (list(dydt) + 
                        list(dSdt) + 
                        [item for sublist in dHdt_list for item in sublist])

    lambdified_fn = sympy.lambdify(
        [symbolic_ode.time, aug_states_sym, symbolic_ode.params],
        aug_dynamics_sym,
        'jax'
    )

    # Create a final wrapper to handle Diffrax's PyTree state (y, S, H)
    def final_aug_ode_fn(t, aug_y, args):
        y, s_matrix, h_tensor = aug_y
        s_flat = list(s_matrix.flatten())
        h_flat = list(h_tensor.flatten())
        aug_y_flat = list(y) + s_flat + h_flat
        
        derivatives_flat = lambdified_fn(t, aug_y_flat, args)
        
        dydt = jnp.array(derivatives_flat[0 : symbolic_ode.n_states])
        dsdt = jnp.array(derivatives_flat[symbolic_ode.n_states : symbolic_ode.n_states + symbolic_ode.n_states * symbolic_ode.n_params]).reshape((symbolic_ode.n_states, symbolic_ode.n_params))
        dhdt = jnp.array(derivatives_flat[symbolic_ode.n_states + symbolic_ode.n_states * symbolic_ode.n_params :]).reshape((symbolic_ode.n_params, symbolic_ode.n_states, symbolic_ode.n_params))
        
        return (dydt, dsdt, dhdt)

    return final_aug_ode_fn
