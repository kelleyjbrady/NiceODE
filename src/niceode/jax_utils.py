import numpy as np
import jax.numpy as jnp
import pandas as pd
import jax
from functools import partial
import numdifftools as nd
import optax


def make_jittable_pk_coeff(expected_len_out):
    #@jax.jit
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
    pop_coeff_for_J_idx,   # Static argument: A tuple of string keys
    pop_coeffs,
    #pop_coeffs_order,
    data_contribution,
    #thetas,
    #theta_data,
    ode_t0_vals,
    #gen_coeff_jit,
    compiled_ivp_solver_arr,
):
    """
    Prediction function refactored for stable nested differentiation.
    It takes coefficient VALUES as the primary differentiable argument and rebuilds
    the dictionary internally from the static keys.
    """
    # Reconstruct the dictionary from the static keys and differentiable values.
    # This keeps the non-differentiable structure separate from the values being traced.
    #jac_pop_coeffs = dict(zip(jac_pop_coeff_keys, jac_pop_coeff_values))

    # The rest of the function proceeds as before.
    #combined_pop_coeffs = {**pop_coeffs, **jac_pop_coeffs}
    #arrays_to_stack = tuple(combined_pop_coeffs[k] for k in pop_coeffs_order)
    #pop_coeffs_work = jnp.stack(arrays_to_stack).flatten()
    #model_coeffs_jit = gen_coeff_jit(
    #    pop_coeffs, thetas, theta_data,
    #)
    pop_coeffs_work = pop_coeffs.at[pop_coeff_for_J_idx].set(jac_pop_coeff_values)
    
    model_coeffs_i = jnp.exp(data_contribution + pop_coeffs_work)# + 1e-6
    #model_coeffs_jit = {i: model_coeffs_jit[i[0]] for i in pop_coeffs_order}
    #model_coeffs_jit_a = jnp.vstack([model_coeffs_jit[i] for i in model_coeffs_jit]).T
    
    # --- DEBUGGING ---
    # Print the shapes and dtypes of the arguments going into the solver.
    # The label is important for finding the output in the console.
    #jax.debug.print("--- Jacobian's diffeqsolve input ---")
    #jax.debug.print("y0 shape: {x}", x=ode_t0_vals.shape)
    #jax.debug.print("args (model_coeffs_i) shape: {x}", x=model_coeffs_i.shape)
    # --- END DEBUGGING ---
    
    
    _full_preds, pred_y = compiled_ivp_solver_arr(
        ode_t0_vals,
        model_coeffs_i
    )
    return pred_y

#@partial(jax.jit, static_argnames=(
#    "pop_coeffs_order", "gen_coeff_jit", "compiled_ivp_solver_arr"
#)) #this decorator was from the old function, possibly not relevant to the new one
def _estimate_jacobian_jax(
    jac_pop_coeffs,
    pop_coeffs,
    pop_coeff_for_J_idx,
    ode_t0_vals,
    compiled_ivp_solver_arr,
    data_contribution,
):
    """
    Estimates the Jacobian and stops its gradient for the FO approximation.
    """
    # 1. Use functools.partial to create a clean function of only the
    #    variable we want to differentiate. This does NOT create a leaky closure.
    predict_fn_with_baked_args = partial(
        _predict_jax_jacobian,
        pop_coeff_for_J_idx=pop_coeff_for_J_idx,
        pop_coeffs=pop_coeffs,
        data_contribution=data_contribution,
        ode_t0_vals=ode_t0_vals,
        compiled_ivp_solver_arr=compiled_ivp_solver_arr,
    )

    # 2. Calculate the Jacobian of this simplified function.
    J = jax.jacobian(predict_fn_with_baked_args)(jac_pop_coeffs)

    # 3. Use stop_gradient to ensure no gradients flow back through J.
    #    This achieves the goal of the custom_vjp in a much clearer way.
    return jax.lax.stop_gradient(J)

def _estimate_jacobian_finite_diff(
    jac_pop_coeffs,
    pop_coeffs,
    pop_coeffs_order,
    thetas,
    theta_data,
    ode_t0_vals,
    gen_coeff_jit,
    compiled_ivp_solver_arr,
    data_contribution,
    time_mask_y,
):
    """
    Final, correct version using finite differences and a non-recursive custom_jvp.
    """
    # 1. Define a function that computes the Jacobian using pure_callback.
    def get_jacobian_value(jac_values_tuple):
        pred_shape = (ode_t0_vals.shape[0], time_mask_y.shape[1])
        result_shape_dtype = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(shape=pred_shape, dtype=x.dtype),
            jac_pop_coeffs
        )

        def get_jacobian_via_numpy(jac_values_tuple_np):
            def func_for_fd(x_vector):
                params_as_tuple = tuple(jnp.array(val) for val in x_vector)
                pred_y = _predict_jax_jacobian(
                    jac_pop_coeff_values=params_as_tuple,
                    jac_pop_coeff_keys=tuple(jac_pop_coeffs.keys()),
                    pop_coeffs=pop_coeffs, pop_coeffs_order=pop_coeffs_order,
                    thetas=thetas, theta_data=theta_data, ode_t0_vals=ode_t0_vals,
                    gen_coeff_jit=gen_coeff_jit, compiled_ivp_solver_arr=compiled_ivp_solver_arr,
                    data_contribution=data_contribution
                )
                return np.array(pred_y.flatten())

            jac_values_np_vector = np.array([v.item() for v in jac_values_tuple_np])
            full_jacobian_matrix_np = nd.Jacobian(func_for_fd)(jac_values_np_vector)
            reshaped_jacobians = tuple(
                col.reshape(pred_shape) for col in full_jacobian_matrix_np.T
            )
            return reshaped_jacobians

        return jax.pure_callback(
            get_jacobian_via_numpy, result_shape_dtype, jac_values_tuple
        )

    # 2. Define the VJP-wrapped function.
    @jax.custom_jvp
    def jacobian_fd_with_fo_grad(jac_values_tuple):
        # The primal is still defined by calling the underlying implementation.
        return get_jacobian_value(jac_values_tuple)

    # 3. Define the JVP rule for this function.
    @jacobian_fd_with_fo_grad.defjvp
    def jacobian_fd_jvp_rule(primals, tangents):
        jac_values_tuple, = primals
        
        # --- THE FIX ---
        # Calculate the primal output by calling the NON-DECORATED logic directly.
        primal_out = get_jacobian_value(jac_values_tuple)

        # For the FO method, the derivative of the Jacobian is zero.
        # The output tangent is a pytree of zeros with the same structure as the output.
        tangent_out = jax.tree.map(jnp.zeros_like, primal_out)
        
        return primal_out, tangent_out

    # 4. Execute the final, fully-defined function.
    jac_values_jax = tuple(jac_pop_coeffs.values())
    # This call will now trigger the corrected JVP rule during the grad trace.
    jacobian_dict = jacobian_fd_with_fo_grad(jac_values_jax)

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

#@jax.jit
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

def estimate_ebes_jax(
    padded_J, padded_residuals, time_mask, omegas2, sigma2, **kwargs
):
    """
    Calculates the Empirical Bayes Estimates (EBEs) for all subjects in parallel.

    Args:
        padded_J: Padded Jacobian matrix.
                  Shape: (n_individuals, max_obs, n_random_effects)
        padded_residuals: Padded residuals.
                          Shape: (n_individuals, max_obs)
        time_mask: Boolean mask indicating valid (not padded) observations.
                   Shape: (n_individuals, max_obs)
        omegas2: Covariance matrix of the random effects (shared).
                 Shape: (n_random_effects, n_random_effects)
        sigma2: Variance of the residual error (shared).
                Shape: scalar

    Returns:
        b_i_approx: The estimated EBEs for each individual.
                    Shape: (n_individuals, n_random_effects)
    """

    # 1. Pre-calculate the inverse of omega^2 once (it's shared across all subjects)
    # Using the pseudo-inverse (pinv) is more numerically stable than inv.
    omega_inv = jnp.linalg.pinv(omegas2)

    # 2. Define the estimation function for a SINGLE individual
    def _calculate_single_b_i(J_sub, residuals_sub, mask_sub):
        """Performs the EBE calculation for one subject."""
        # Zero out the padded values to exclude them from calculations
        J_masked = jnp.where(mask_sub[:, None], J_sub, 0)
        residuals_masked = jnp.where(mask_sub, residuals_sub, 0)

        # A = J'J + sigma^2 * Omega^-1
        A = J_masked.T @ J_masked + sigma2 * omega_inv
        # rhs = J' * residuals
        rhs = J_masked.T @ residuals_masked

        # Solve for b_i: A * b_i = rhs
        b_i = jnp.linalg.solve(A, rhs)
        return b_i

    # 3. Use jax.vmap to apply the single-subject function to all subjects
    # The in_axes argument tells vmap to map over the first axis of the
    # padded arrays and the mask.
    b_i_approx = jax.vmap(
        _calculate_single_b_i, in_axes=(0, 0, 0)
    )(padded_J, padded_residuals, time_mask)

    return b_i_approx

#@partial(jax.jit, static_argnames = ( 
#                                        # SHAPES for jnp.zeros
#                                    "theta_total_len",
#                                    "total_n_params",  
#                                    "omega_diag_size",
#                                    
#                                    # DICTIONARY for lookups                                
#                                    "params_idx", 
#                                    
#                                    # BOOLEAN FLAGS for potential `if` statements
#                                    "use_full_omega", 
#                                    "optimize_omega_on_log_scale", 
#                                    "optimize_sigma_on_log_scale", 
#                                    "use_surrogate_neg2ll",) )
def _jittable_param_unpack(opt_params, 
                            theta_data_tensor, 
                            theta_update_to_indices,
                            theta_update_from_indices,
                            theta_total_len,
                            padded_y, 
                            time_mask_y, 
                            time_mask_J, 
                            unpadded_y_len,
                            params_idx,
                            fixed_params = None,
                            #pop_coeff_cols = None,
                            #omega_diag_cols = None,
                            pop_coeffs_for_J_idx = None,
                            opt_params_combined_params_idx = None, 
                            fixed_params_combined_params_idx = None,
                            total_n_params = None,
                            init_params_for_scaling = None,
                            ode_t0_vals = None,
                            use_full_omega = None, 
                            omega_lower_chol_idx = None,
                            omega_diag_size = None,
                            optimize_omega_on_log_scale = None,
                            optimize_sigma_on_log_scale = None,
                            use_surrogate_neg2ll = None,
                            ):
    
    #pop_coeffs_for_J_idx = [pop_coeff_cols[i] for i in pop_coeff_cols if i[0] in [i[0][0] for i in omega_diag_cols]]
    
    combined_params = jnp.zeros(total_n_params)
    
    #combined_params[opt_params_combined_params_idx] = opt_params
    combined_params = combined_params.at[opt_params_combined_params_idx].set(opt_params)
    #combined_params[fixed_params_combined_params_idx] = fixed_params
    combined_params = combined_params.at[fixed_params_combined_params_idx].set(fixed_params)
    params = combined_params + init_params_for_scaling

    #unpack pop coeffs
    pop_coeffs = params[params_idx.pop[0]:params_idx.pop[-1]]
    
    #unpack sigma, assume opt on log scale
    sigma = params[params_idx.sigma[0]:params_idx.sigma[-1]]
    sigma2 = jnp.exp(sigma)**2
    
    
    #unpack omega, assume opt on log scale
    omega = params[params_idx.omega[0]:params_idx.omega[-1]]
    #update_idx = omega_lower_chol_idx
    rows, cols = omega_lower_chol_idx
    omega_lchol = jnp.zeros((omega_diag_size, omega_diag_size), dtype = np.float64)
    omega_lchol = omega_lchol.at[rows, cols].set(omega)
    omegas_diag = jnp.diag(omega_lchol)
    omegas_diag = jnp.exp(omegas_diag)
    omega_lchol = omega_lchol.at[jnp.diag_indices_from(omega_lchol)].set(omegas_diag)
    omega2 = omega_lchol @ omega_lchol.T
    
    #unpack theta
    theta = params[params_idx.theta[0]:params_idx.theta[-1]]
    thetas_work = jnp.zeros(theta_total_len)
    thetas_work = thetas_work.at[theta_update_to_indices].set(theta[theta_update_from_indices])
    
    loss_kwargs = {
        'pop_coeff':pop_coeffs, 
        'pop_coeff_for_J_idx':pop_coeffs_for_J_idx,
        'sigma2':sigma2, 
        'omega2':omega2, 
        'theta':thetas_work, 
        'theta_data':theta_data_tensor, 
        'padded_y':padded_y,
        'time_mask_y':time_mask_y,
        'time_mask_J':time_mask_J,
        'unpadded_y_len':unpadded_y_len, 
        'ode_t0_vals':ode_t0_vals,
        'omega2_diag_size':omega_diag_size
    } 
    return loss_kwargs


def create_jax_objective(static_opt_kwargs,
                         dynamic_opt_kwargs,
                         compiled_augdyn_solver,
                         compiled_solver,
                         jittable_loss, 
                         
                         ):
              
    all_other_kwargs = {**static_opt_kwargs, **dynamic_opt_kwargs}
    
    def _jax_objective_function(opt_params, ):
        
        loss_kwargs = _jittable_param_unpack(opt_params=opt_params, **all_other_kwargs)
        loss_kwargs['compiled_augdyn_ivp_solver_arr'] = compiled_augdyn_solver
        loss_kwargs['compiled_ivp_solver_arr'] = compiled_solver
        
        loss, _ = jittable_loss(
                **loss_kwargs
            )
        
        return loss
    
    
    def _jax_objective_function_predict(opt_params, ):
        
        loss_kwargs = _jittable_param_unpack(opt_params=opt_params, **all_other_kwargs)
        loss_kwargs['compiled_augdyn_ivp_solver_arr'] = compiled_augdyn_solver
        loss_kwargs['compiled_ivp_solver_arr'] = compiled_solver
        
        loss_bundle = jittable_loss(
                **loss_kwargs
            )
        
        return loss_bundle
    #static_names_for_jit = ["compiled_solver"] + list(static_opt_kwargs.keys())
    #
    #jitted_objective = partial(jax.jit(
    #    _jax_objective_function, 
    #    static_argnames=static_names_for_jit 
    #),
    #
    #compiled_solver = compiled_solver,   
    #**static_opt_kwargs, 
    #**dynamic_opt_kwargs                 
    #                           )
    return jax.jit(_jax_objective_function), jax.jit(_jax_objective_function_predict)
    #return _jax_objective_function, _jax_objective_function_predict

#@partial(jax.jit, static_argnames = (
#                                    "compiled_ivp_solver_arr",
#
# ))
def create_aug_dynamics_ode(diffrax_ode,  pop_coeff_for_J_idx):
    
    def aug_dynamics_ode(t, aug_y, args):
        y, S = aug_y 
        j_args = args[pop_coeff_for_J_idx]
        
        
        #jax is able to handle a lambda liike this, but not a closure
        #with a def
        j_ode = lambda y_in, j_args_in: diffrax_ode(
            t,
            y_in,
            args.at[pop_coeff_for_J_idx].set(j_args_in)
        )
            
        
        jac_f_wrt_y = jax.jacobian(j_ode, argnums=0)(y, j_args)
        jac_f_wrt_p = jax.jacobian(j_ode, argnums=1)(y, j_args)

        
        dydt = diffrax_ode(t, y, args)
        
        dSdt = jac_f_wrt_y @ S + jac_f_wrt_p 

        return (dydt, dSdt)
    
    return aug_dynamics_ode

def aug_dynamics_jax_worker():
    raise NotImplementedError

def estimate_b_i_vmapped(
    # Batched inputs (one entry per subject)
    initial_b_i_batch,
    padded_y_batch,
    data_contribution_batch,
    ode_t0_vals_batch,
    time_mask_y_batch,
    unpadded_y_len_batch,
    # Shared inputs (same for all subjects)
    pop_coeff,
    sigma2,
    omega2,
    n_random_effects,
    compiled_ivp_solver,
):
    """
    Estimates b_i for all subjects in parallel using a vmapped Optax optimizer.
    """
    # Define the optimization for a single subject
    def _estimate_single_b_i(initial_b_i,
                             padded_y_i,
                             data_contrib_i,
                             ode_t0_i,
                             time_mask_y_i,
                             unpadded_y_i_len,
                             ):
        # The objective function with data for this subject closed over
        # We also unpad the subject's observation vector here
        #y_i = padded_y_batch[:n_obs]
        
        obj_fn = lambda b_i: FOCE_inner_loss_fn(
            b_i = b_i, 
            padded_y_i = padded_y_i,
            unpadded_y_i_len=unpadded_y_i_len,
            pop_coeff_i=pop_coeff,
            data_contrib_i = data_contrib_i,
            sigma2 = sigma2,
            omega2 = omega2,
            ode_t0_i = ode_t0_i,
            time_mask_y_i=time_mask_y_i,
            n_random_effects=n_random_effects, 
            compiled_ivp_solver=compiled_ivp_solver,
            
        )

        # Initialize the L-BFGS optimizer
        optimizer = optax.lbfgs(maxiter=50, tol=1e-4)
        
        # Run the optimization to find the parameters (b_i) that minimize the objective
        # The `run` method is a convenient wrapper for stateful optimizers
        results = optimizer.run(obj_fn, initial_b_i)
        
        # Return the optimized parameters
        return results.params

    # Vmap the single-subject optimization function
    # `in_axes` specifies which arguments to map over. `None` means broadcast.
    vmapped_optimizer = jax.vmap(
        _estimate_single_b_i,
        in_axes=(0, 0, 0, 0, 0, 0),  # Map over all batched inputs
    )
    
    # Execute the vmapped function
    all_b_i_estimates = vmapped_optimizer(
        initial_b_i_batch,
        padded_y_batch,
        data_contribution_batch,
        ode_t0_vals_batch,
        time_mask_y_batch,
        unpadded_y_len_batch
    )
    
    return all_b_i_estimates

def FOCE_inner_loss_fn(
    b_i,  # The parameters we are optimizing (random effects)
    # --- Static data for this subject (won't change during inner opt) ---
    padded_y_i,
    unpadded_y_i_len,
    pop_coeff_i,
    data_contribution_i,
    sigma2,
    omega2,
    ode_t0_val_i,
    time_mask_y_i,
    n_random_effects,
    compiled_ivp_solver,
):
    """
    Calculates the conditional negative 2 log-likelihood for a single subject.
    This function is pure and written entirely in JAX.
    """
    # Combine population and random effects to get subject-specific coefficients
    # This assumes b_i are additive adjustments on the log-scale
    combined_coeffs = pop_coeff_i + b_i
    model_coeffs_i = jnp.exp(data_contribution_i + combined_coeffs)

    # --- Data Likelihood Part ---
    # Solve the ODE for this individual with the current b_i guess
    padded_preds_i = compiled_ivp_solver(model_coeffs_i, ode_t0_val_i,)
    masked_residuals_i = jnp.where(time_mask_y_i, padded_y_i - padded_preds_i, 0.0)
    # Compute residuals only for observed time points
    
    n_t = unpadded_y_i_len # Number of actual observations for subject i
    sum_sq_residuals = jnp.sum(masked_residuals_i**2)
    
    # This assumes an additive error model, as in the original code
    neg2_ll_data = (n_t * jnp.log(2 * jnp.pi) 
                    + n_t * jnp.log(sigma2) 
                    + sum_sq_residuals / sigma2)

    # --- Prior Penalty Part ---
    # Calculate the penalty from the random effects distribution
    # b_i.T @ inv(omega2) @ b_i + log(det(omega2))
    L, low = jax.scipy.linalg.cho_factor(omega2, lower=True)
    log_det_omega2 = 2 * jnp.sum(jnp.log(jnp.diag(L)))
    
    # Use Cholesky solve for stability and efficiency
    prior_penalty = b_i @ jnp.linalg.cho_solve((L, True), b_i)

    neg2ll_prior = (n_random_effects * jnp.log(2 * jnp.pi) 
                    + log_det_omega2 
                    + prior_penalty)

    return neg2_ll_data + neg2ll_prior

def estimate_b_i_foce_passthrough(pop_coeff, **kwargs):
    return pop_coeff

def estimate_b_i_fo_passthrough(b_i, **kwargs):
    return b_i

def FO_approx_neg2ll_loss_jax(
    pop_coeff, 
    sigma2, 
    omega2, 
    theta, 
    theta_data,
    padded_y,
    unpadded_y_len,
    time_mask_y,
    time_mask_J,
    compiled_augdyn_ivp_solver_arr,
    compiled_ivp_solver_arr,
    ode_t0_vals,
    pop_coeff_for_J_idx,
    **kwargs,
):
    loss = approx_neg2ll_loss_jax(
        pop_coeff = pop_coeff, 
        sigma2 = sigma2, 
        omega2 = omega2, 
        theta = theta, 
        theta_data = theta_data,
        padded_y = padded_y,
        unpadded_y_len = unpadded_y_len,
        time_mask_y = time_mask_y,
        time_mask_J = time_mask_J,
        compiled_augdyn_ivp_solver_arr = compiled_augdyn_ivp_solver_arr,
        compiled_ivp_solver_arr = compiled_ivp_solver_arr,
        ode_t0_vals = ode_t0_vals,
        pop_coeff_for_J_idx = pop_coeff_for_J_idx,
        compiled_estimate_b_i_foce = estimate_b_i_foce_passthrough,
        compiled_estimate_b_i_fo = estimate_ebes_jax
    )
    
    return loss

def approx_neg2ll_loss_jax(
    pop_coeff, 
    sigma2, 
    omega2, 
    theta, 
    theta_data,
    #params,
    #params_order,
    #theta_data,
    #theta_data_tensor,
    #theta_data_tensor_names,
    padded_y,
    unpadded_y_len,
    #y_groups_idx, 
    #y_groups_unique,
    #n_population_coeff, 
    #pop_coeff_names,
    #omega2_diag_size,
    #subject_level_effect_names,
    #sigma_names,
    #n_thetas,
    #theta_names,
    time_mask_y,
    time_mask_J,
    #compiled_ivp_solver_keys,
    compiled_augdyn_ivp_solver_arr,
    compiled_ivp_solver_arr,
    ode_t0_vals,
    #compiled_gen_ode_coeff,
    pop_coeff_for_J_idx,
    #compiled_ivp_predictor,
    #solve_for_omegas=False,
    compiled_estimate_b_i_foce, 
    compiled_estimate_b_i_fo,
):
    
    #jax.debug.print("theta shape: {s}", s = theta.shape )
    n_subjects = time_mask_y.shape[0]
    n_coeffs = pop_coeff.shape[0]
    n_subject_level_eff = pop_coeff_for_J_idx.shape[0]
    if theta.shape[0] == 0:
        
        data_contribution = jnp.zeros((n_subjects, n_coeffs))
    else:
        data_contribution = theta @ theta_data
    #jax.debug.print("time_mask_y shape: {s}", s = time_mask_y.shape )
    #jax.debug.print("data_contribution shape: {j}", j=data_contribution.shape)
    #jax.debug.print("pop_coeff shape: {j}", j=pop_coeff.shape)
    
    #estimate b_i here
    initial_b_i_batch = jnp.zeros((n_subjects, n_subject_level_eff))
    subject_coeff = compiled_estimate_b_i_foce(
                                               initial_b_i_batch = initial_b_i_batch,
                                                padded_y_batch = padded_y,
                                                data_contribution_batch = data_contribution,
                                                ode_t0_vals_batch = ode_t0_vals,
                                                time_mask_y_batch = time_mask_y,
                                                unpadded_y_len_batch = unpadded_y_len,
                                                pop_coeff = pop_coeff,
                                                sigma2 = sigma2,
                                                omega2 = omega2,
                                                n_random_effects = n_subject_level_eff,
                                                compiled_ivp_solver = compiled_ivp_solver_arr,
                                               
                                               
                                               )
    model_coeffs_i = jnp.exp(data_contribution + subject_coeff)# + 1e-6
    is_zero = (model_coeffs_i == 0)
    not_finite = ~jnp.isfinite(model_coeffs_i)
    is_bad_state = jnp.any(not_finite | is_zero)
    #jax.debug.print("model_coeffs_i shape: {j}", j=model_coeffs_i.shape)
    padded_full_preds, padded_pred_y, J_full, J_conc_full = compiled_augdyn_ivp_solver_arr(
        ode_t0_vals,
        model_coeffs_i
    )
    #jax.debug.print("padded_pred_y shape: {padded_pred_y}", padded_pred_y=padded_pred_y.shape)
    #jax.debug.print("J_conc_full shape: {J_conc_full}", J_conc_full=J_conc_full.shape)
    #jax.debug.print("model_coeffs_i shape: {j}", j=model_coeffs_i.shape)
    
    masked_residuals = jnp.where(time_mask_y, padded_y - padded_pred_y, 0.0)

    J_dense = J_conc_full#shape (n_subject, max_obs_per_subject, n_s)
    J = jnp.where(time_mask_J, J_dense, 0.0) # time_mask_J shape  (n_subject, max_obs_per_subject, n_s)
    
    

    # Estimate the covariance matrix, then estimate neg log likelihood
    neg2_ll = neg2_ll_chol_jit(
        J,           # Shape: (n_subjects, max_obs, n_effects)
        masked_residuals,  # Shape: (n_subjects, max_obs)
        time_mask_y,              # Shape: (n_subjects, max_obs)
        sigma2,            # Shape: scalar
        omega2,           # Shape: (n_effects, n_effects)
        unpadded_y_len,
    )
    #jax.debug.print("neg2_ll: {neg2_ll}", neg2_ll=neg2_ll)
    #jax.debug.print("is_bad_state: {is_bad_state}", is_bad_state=is_bad_state)
    b_i_approx = compiled_estimate_b_i_fo(padded_J=J, padded_residuals=masked_residuals, 
                                   time_mask=time_mask_y,
                                   omegas2=omega2, sigma2=sigma2
                                   )
    neg2_ll = jnp.where(is_bad_state, jnp.inf, neg2_ll)
    return neg2_ll, (b_i_approx, (padded_pred_y, padded_full_preds))