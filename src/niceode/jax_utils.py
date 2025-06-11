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
            jax.debug.print("--- Debugging Shapes ---")
            jax.debug.print("X shape: {x_shape}", x_shape=X.shape)
            jax.debug.print("theta shape: {theta_shape}", theta_shape=theta.shape)
            data_contribution = (X @ theta)
            out = jnp.exp(data_contribution + pop_coeff)   + 1e-6
            #if len(out) != expected_len_out:
            #    out = jnp.repeat(out, expected_len_out)
            #out = jnp.where(len(out) != expected_len_out,jnp.repeat(out, expected_len_out), out )
            model_coeffs[c[0]] = out

        return model_coeffs
    return generate_pk_model_coeff_jax

@partial(jax.jit, static_argnames = ("pop_coeffs_order", 
                                     "gen_coeff_jit", "compiled_ivp_solver"
                                     ) )
def _predict_jax_jacobian(jac_pop_coeffs, pop_coeffs,
                 pop_coeffs_order, thetas,
                 theta_data, ode_t0_vals, 
                 gen_coeff_jit,
                 compiled_ivp_solver):

    pop_coeffs.update(jac_pop_coeffs)
    model_coeffs_jit = gen_coeff_jit(
        pop_coeffs, thetas, theta_data,
    )
    
    model_coeffs_jit = {i:model_coeffs_jit[i[0]] for i in pop_coeffs_order}
    # would be good to create wrapper methods inside of the model obj with these args (eg. `parallel = model_obj.parallel`)
    # prepopulated with partial
    #return model_coeffs
    model_coeffs_jit_a = jnp.vstack([model_coeffs_jit[i] for i in model_coeffs_jit]).T
    full_preds, pred_y = compiled_ivp_solver(
        ode_t0_vals,
        model_coeffs_jit_a
    )
    return pred_y

@partial(jax.jit, static_argnames = ("pop_coeffs_order", 
                                     "gen_coeff_jit", "compiled_ivp_solver"
                                     ) )
def _estimate_jacobian_jax(jac_pop_coeffs, pop_coeffs,
                 pop_coeffs_order, thetas,
                 theta_data, ode_t0_vals, 
                 gen_coeff_jit,
                 compiled_ivp_solver):
    
    jac_fn = jax.jacobian(_predict_jax_jacobian)

    jacobian_dict = jac_fn(jac_pop_coeffs, pop_coeffs,
                 pop_coeffs_order, thetas,
                 theta_data, ode_t0_vals, 
                 gen_coeff_jit,
                 compiled_ivp_solver)
    
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
    y,
    y_groups_idx, 
    y_groups_unique,
    n_population_coeff, 
    n_subject_level_effects,
    time_mask,
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
    full_preds, pred_y = compiled_ivp_solver(
        ode_t0_vals,
        model_coeffs_a
    )
    
    preds = pred_y[time_mask]
    
    
    residuals = y - preds

    pop_coeffs_j = {i:pop_coeffs[i] for i in pop_coeffs if i[0] in [i[0] for i in omegas_order]}
    j = _estimate_jacobian_jax(jac_pop_coeffs = pop_coeffs_j, pop_coeffs=pop_coeffs, pop_coeffs_order=pop_coeffs_order,
                               thetas=thetas, theta_data = theta_data, ode_t0_vals=ode_t0_vals,
                               gen_coeff_jit=compiled_gen_ode_coeff, compiled_ivp_solver=compiled_ivp_solver
                               )
    j = {i:j[i][time_mask] for i in j if i in (i for i in pop_coeffs_j)}
    J = jnp.vstack([j[i].flatten() for i in j]).T
    #J = 
    # Estimate the covariance matrix, then estimate neg log likelihood
    neg2_ll = neg2_ll_chol(
        J,
        y_groups_idx,
        y_groups_unique,
        y,
        residuals,
        sigma2,
        omegas2,
    )

    # if predicting or debugging, solve for the optimal b_i given the first order apprx
    # perhaps b_i approx is off in the 2cmpt case bc the model was learned on t1+ w/
    # t0 as the intial condition but now t0 is in the data w/out a conc in the DV
    # col, this should make the resdiduals very wrong
    b_i_approx = jnp.zeros((n_individuals, n_subject_level_effects))
    if solve_for_omegas:
        for sub_idx, sub in enumerate(y_groups_unique):
            filt = y_groups_idx == sub
            J_sub = J[filt]
            residuals_sub = residuals[filt]
            try:
                # Ensure omegas2 is invertible (handle near-zero omegas)
                # Add a small value to diagonal for stability if needed, or use pinv
                min_omega_var = 1e-9  # Example threshold
                stable_omegas2 = jnp.diag(jnp.maximum(jnp.diag(omegas2), min_omega_var))
                omega_inv = jnp.linalg.inv(stable_omegas2)

                # Corrected matrix A
                A = J_sub.T @ J_sub + sigma2 * omega_inv
                # Right-hand side
                rhs = J_sub.T @ residuals_sub
                # Solve
                b_i_approx[sub_idx, :] = jnp.linalg.solve(A, rhs)

            except jnp.linalg.LinAlgError:
                print(
                    f"Warning: Linear algebra error (likely singular matrix) for subject {sub}. Setting b_i to zero."
                )

            except Exception as e:
                print(f"Error calculating b_i for subject {sub}: {e}")

    return neg2_ll, b_i_approx, (preds, full_preds)