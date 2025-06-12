import numpy as np
import jax.numpy as jnp
import pandas as pd
import jax
from functools import partial


def make_jittable_pk_coeff(expected_len_out):
    @jax.jit
    def generate_pk_model_coeff_jax(pop_coeffs, thetas, theta_data):


        model_coeffs = {}
        pop_coeffs_l0 = [i[0] for i in pop_coeffs]
        thetas_l0 = [i[0] for i in thetas]
        theta_data_l0 = [i[0] for i in theta_data]
        for c in pop_coeffs:
            pop_coeff = pop_coeffs[c]
            theta = (
                jnp.array([thetas[i] for i in thetas if i[0] == c[0]]).flatten()
                if c[0] in thetas_l0
                else jnp.zeros_like(pop_coeff)
            )
            X = (
                jnp.vstack(jnp.array([theta_data[i] for i in theta_data if i[0] == c[0]])).T
                if c[0] in theta_data_l0
                #else jnp.zeros_like(pop_coeff)
                else jnp.zeros((expected_len_out, 1))
            )
            #jax.debug.print("--- Debugging Shapes ---")
            #jax.debug.print("X shape: {x_shape}", x_shape=X.shape)
            #jax.debug.print("theta shape: {theta_shape}", theta_shape=theta.shape)
            data_contribution = (X @ theta)
            out = jnp.exp(data_contribution + pop_coeff)   + 1e-6
            #if len(out) != expected_len_out:
            #    out = jnp.repeat(out, expected_len_out)
            #out = jnp.where(len(out) != expected_len_out,jnp.repeat(out, expected_len_out), out )
            model_coeffs[c[0]] = out

        return model_coeffs
    return generate_pk_model_coeff_jax

def _predict_jax_jacobian(
    jac_pop_coeff_values, # Differentiable argument: A tuple of JAX arrays
    jac_pop_coeff_keys,   # Static argument: A tuple of string keys
    pop_coeffs,
    pop_coeffs_order,
    thetas,
    theta_data,
    ode_t0_vals,
    gen_coeff_jit,
    compiled_ivp_solver
):
    """
    Prediction function refactored for stable nested differentiation.
    It takes coefficient VALUES as the primary differentiable argument and rebuilds
    the dictionary internally from the static keys.
    """
    # Reconstruct the dictionary from the static keys and differentiable values.
    # This keeps the non-differentiable structure separate from the values being traced.
    jac_pop_coeffs = dict(zip(jac_pop_coeff_keys, jac_pop_coeff_values))

    # The rest of the function proceeds as before.
    pop_coeffs.update(jac_pop_coeffs)
    model_coeffs_jit = gen_coeff_jit(
        pop_coeffs, thetas, theta_data,
    )

    model_coeffs_jit = {i: model_coeffs_jit[i[0]] for i in pop_coeffs_order}
    model_coeffs_jit_a = jnp.vstack([model_coeffs_jit[i] for i in model_coeffs_jit]).T
    
    _full_preds, pred_y = compiled_ivp_solver(
        ode_t0_vals,
        model_coeffs_jit_a
    )
    return pred_y

@partial(jax.jit, static_argnames=(
    "pop_coeffs_order", "gen_coeff_jit", "compiled_ivp_solver"
))
def _estimate_jacobian_jax(
    jac_pop_coeffs, # The original dictionary of coefficients
    pop_coeffs,
    pop_coeffs_order,
    thetas,
    theta_data,
    ode_t0_vals,
    gen_coeff_jit,
    compiled_ivp_solver
):
    """
    Estimates the Jacobian by differentiating with respect to array values,
    not the dictionary structure, to ensure compatibility with second-order gradients.
    """
    # Deconstruct the dictionary into differentiable values (JAX arrays) and
    # static keys (strings). This is the key step for stable differentiation.
    jac_keys = tuple(jac_pop_coeffs.keys())
    jac_values = tuple(jac_pop_coeffs.values())

    # Create a partial function where only the values will be the dynamic,
    # differentiable argument. All other arguments, including the keys, are
    # treated as static for the purpose of this differentiation.
    predict_fn_partial = partial(
        _predict_jax_jacobian,
        jac_pop_coeff_keys=jac_keys, # Pass keys as a static argument
        pop_coeffs=pop_coeffs,
        pop_coeffs_order=pop_coeffs_order,
        thetas=thetas,
        theta_data=theta_data,
        ode_t0_vals=ode_t0_vals,
        gen_coeff_jit=gen_coeff_jit,
        compiled_ivp_solver=compiled_ivp_solver
    )
    
    # Calculate the jacobian of the partial function with respect to its first
    # argument, which is now `jac_pop_coeff_values` (a tuple of arrays).
    jac_fn = jax.jacobian(predict_fn_partial, argnums=0)

    # Execute the jacobian function on the differentiable values.
    jacobian_as_pytree = jac_fn(jac_values)
    
    # The output `jacobian_as_pytree` has the same pytree structure as the input
    # `jac_values` (a tuple). We can now safely reconstruct the dictionary.
    jacobian_dict = dict(zip(jac_keys, jacobian_as_pytree))

    return jacobian_dict


def neg2_ll_chol(J, y_groups_idx,
                 y_groups_unique,
                 y, residuals, sigma2, omegas2,):
    # V_all = []
    log_det_V = 0
    L_all = []
    for sub in y_groups_unique:
        filt = y_groups_idx == sub
        J_sub = J[filt]
        n_timepoints = len(J_sub)
        R_i = sigma2 * np.eye(n_timepoints)  # Constant error
        # Omega = np.diag(omegas**2) # Construct D matrix from omegas
        V_i = R_i + J_sub @ omegas2 @ J_sub.T

        L_i, lower = jax.scipy.linalg.cho_factor(V_i)  # Cholesky of each V_i
        L_all.append(L_i)
        log_det_V += 2 * np.sum(np.log(np.diag(L_i)))  # log|V_i|

    L_block = jax.scipy.linalg.block_diag(*L_all)  # key change from before
    V_inv_residuals = jax.scipy.linalg.cho_solve((L_block, True), residuals)
    neg2_ll_chol = (
        log_det_V + residuals.T @ V_inv_residuals + len(y) * np.log(2 * np.pi)
    )

    return neg2_ll_chol



def _calculate_per_subject_likelihood_terms(
    J_sub, residuals_sub, mask_sub, sigma2, omegas2
):
    """
    Calculates likelihood components for a single subject on dense/padded arrays.

    Args:
      J_sub: Dense Jacobian for one subject. Shape: (max_obs, num_effects)
      residuals_sub: Dense residuals for one subject. Shape: (max_obs,)
      mask_sub: Boolean mask for one subject. Shape: (max_obs,)
      sigma2: Scalar variance of the residual error.
      omegas2: Dense covariance matrix of random effects. Shape: (num_effects, num_effects)
    """
    # Get the maximum number of observations for padding purposes.
    max_obs = J_sub.shape[0]

    # Construct the dense covariance matrix V_i for this subject.
    # The padded sections will contain garbage values, but we will fix this.
    R_i_dense = sigma2 * jnp.eye(max_obs)
    V_i_dense = R_i_dense + J_sub @ omegas2 @ J_sub.T

    # To handle the variable number of observations, we create a "masked" V_i.
    # Where the mask is False (i.e., for padded data), we want V_i to be
    # the identity matrix. This ensures that its log-determinant is 0
    # and it doesn't affect the Cholesky solve for the valid data.
    identity_matrix = jnp.eye(max_obs)
    
    # Create a 2D mask for the matrix from the 1D time mask
    # This is True where both the row and column correspond to real observations.
    mask_2d = mask_sub[:, None] & mask_sub[None, :]

    V_i_masked = jnp.where(mask_2d, V_i_dense, identity_matrix)

    # Now perform the Cholesky decomposition on the stabilized, dense matrix.
    # We add a small "jitter" for numerical stability, preventing errors
    # if the matrix is not perfectly positive definite.
    jitter = 1e-6
    L_i, lower = jax.scipy.linalg.cho_factor(V_i_masked + identity_matrix * jitter, lower=True)

    # The log-determinant is now correctly calculated on the dense matrix.
    # The identity part contributes log(1) = 0 to the sum.
    log_det_Vi = 2 * jnp.sum(jnp.log(jnp.diag(L_i)))

    # Use the Cholesky factor to solve V_i * x = residuals_sub.
    # Since residuals_sub is already masked with zeros in the padded section,
    # this solve will produce the correct result for the valid data.
    V_inv_residuals_i = jax.scipy.linalg.cho_solve((L_i, lower), residuals_sub)

    # Calculate the quadratic form: residuals.T @ V_inv @ residuals
    quadratic_term = residuals_sub.T @ V_inv_residuals_i

    return log_det_Vi, quadratic_term

@jax.jit
def neg2_ll_chol_jit(
    J_dense,           # Shape: (n_subjects, max_obs, n_effects)
    masked_residuals,  # Shape: (n_subjects, max_obs)
    mask,              # Shape: (n_subjects, max_obs)
    sigma2,            # Shape: scalar
    omegas2,           # Shape: (n_effects, n_effects)
    num_total_obs,     # Static integer: total number of actual observations
):
    """
    Calculates the negative 2 log-likelihood using a batched Cholesky decomposition.
    """
    # Vmap the per-subject function over the subjects axis (axis 0).
    # `in_axes` tells vmap how to handle each argument:
    #   0: Map over the first axis of J_dense, masked_residuals, and mask.
    #   None: Do not map over sigma2 and omegas2; broadcast them to every call.
    vmapped_calculator = jax.vmap(
        _calculate_per_subject_likelihood_terms, in_axes=(0, 0, 0, None, None)
    )

    # Run the vmapped function. It returns two arrays, one for each value returned
    # by the helper function.
    all_log_dets, all_quadratic_terms = vmapped_calculator(
        J_dense, masked_residuals, mask, sigma2, omegas2
    )

    # Aggregate the results from all subjects using jnp.sum
    total_log_det = jnp.sum(all_log_dets)
    total_quadratic = jnp.sum(all_quadratic_terms)

    # Final likelihood calculation
    neg2_ll = total_log_det + total_quadratic + num_total_obs * jnp.log(2 * jnp.pi)

    return neg2_ll


@partial(jax.jit, static_argnames = ("params_order",
                                     "n_population_coeff", 
                                    "n_subject_level_effects",
                                    "compiled_ivp_solver",
                                    "compiled_gen_ode_coeff",
                                    "solve_for_omegas"
                                     ))
def FO_approx_ll_loss_jax(
    params,
    params_order,
    theta_data,
    padded_y,
    unpadded_y_len,
    y_groups_idx, 
    y_groups_unique,
    n_population_coeff, 
    n_subject_level_effects,
    time_mask_y,
    time_mask_J,
    compiled_ivp_solver,
    ode_t0_vals,
    compiled_gen_ode_coeff,
    solve_for_omegas=False,
):
    # unpack some variables locally for clarity
    params_order = list(params_order)
    params = {i:params[i] for i in params_order}
    pop_coeffs = {i:params[i] 
                      for idx, i in enumerate(params) 
                      if idx < n_population_coeff}
    pop_coeffs_order = tuple(params_order[:n_population_coeff])
      
    start_idx = n_population_coeff
    end_idx = start_idx + 1
    
    sigma = {i:params[i] 
                    for idx, i in enumerate(params) 
                    if idx >= start_idx and idx < end_idx}
    sigma_order = params_order[start_idx:end_idx]
    
    start_idx = end_idx
    end_idx = start_idx + n_subject_level_effects
    omegas = {i:params[i] 
                    for idx, i in enumerate(params) 
                    if idx >= start_idx and idx < end_idx}
    omegas_order = params_order[start_idx:end_idx]
    
    start_idx = end_idx
    thetas = {i:params[i] 
                    for idx, i in enumerate(params) 
                    if idx >= start_idx}
    thetas_order = params_order[start_idx:]
    

    # omegas are SD, we want Variance, thus **2 below
    # FO assumes that there is no cov btwn the random effects, thus off diags are zero
    #this is not actually FO's assumption, but a simplification,
    omegas = jnp.array([omegas[i] for i in omegas]).flatten()
    omegas2 = jnp.diag(
        omegas**2
    )  
     
    sigma = jnp.array([sigma[i] for i in sigma]).flatten()
    sigma2 = sigma**2
    n_individuals = len(y_groups_unique)

    # estimate model coeffs when the omegas are zero -- the first term of the taylor exapansion apprx
    model_coeffs = compiled_gen_ode_coeff(
        pop_coeffs, thetas, theta_data,
    )
    model_coeffs = {i:model_coeffs[i[0]] for i in pop_coeffs_order}
    
    model_coeffs_a = jnp.vstack([model_coeffs[i] for i in model_coeffs]).T
    padded_full_preds, padded_pred_y = compiled_ivp_solver(
        ode_t0_vals,
        model_coeffs_a
    )
    
    masked_residuals = jnp.where(time_mask_y, padded_y - padded_pred_y, 0.0)

    pop_coeffs_j = {i:pop_coeffs[i] for i in pop_coeffs if i[0] in [i[0] for i in omegas_order]}
    j = _estimate_jacobian_jax(jac_pop_coeffs = pop_coeffs_j, pop_coeffs=pop_coeffs, pop_coeffs_order=pop_coeffs_order,
                               thetas=thetas, theta_data = theta_data, ode_t0_vals=ode_t0_vals,
                               gen_coeff_jit=compiled_gen_ode_coeff, compiled_ivp_solver=compiled_ivp_solver
                               )
    
    # We create a dense J by stacking, then apply the mask to zero-out rows.
    J_dense = jnp.concatenate([j[key] for key in pop_coeffs_j],axis = 2) #shape (n_subject, max_obs_per_subject, n_s)
    
    J = jnp.where(time_mask_J, J_dense, 0.0) # time_mask_J shape  (n_subject, max_obs_per_subject, n_s)
    #J = 
    # Estimate the covariance matrix, then estimate neg log likelihood
    neg2_ll = neg2_ll_chol_jit(
        J,           # Shape: (n_subjects, max_obs, n_effects)
        masked_residuals,  # Shape: (n_subjects, max_obs)
        time_mask_y,              # Shape: (n_subjects, max_obs)
        sigma2,            # Shape: scalar
        omegas2,           # Shape: (n_effects, n_effects)
        unpadded_y_len,
    )

    # if predicting or debugging, solve for the optimal b_i given the first order apprx
    # perhaps b_i approx is off in the 2cmpt case bc the model was learned on t1+ w/
    # t0 as the intial condition but now t0 is in the data w/out a conc in the DV
    # col, this should make the resdiduals very wrong
    b_i_approx = jnp.zeros((n_individuals, n_subject_level_effects))

    return neg2_ll, (b_i_approx, (padded_pred_y, padded_full_preds))