import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
import numdifftools as nd
import optax
import finitediffx as fdx
import warnings


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
def outer_neg2ll_chol_jit(
    J_dense,           # Shape: (n_subjects, max_obs, n_effects)
    masked_residuals,  # Shape: (n_subjects, max_obs)
    mask,              # Shape: (n_subjects, max_obs)
    sigma2,            # Shape: scalar
    omegas2,           # Shape: (n_effects, n_effects)
    num_total_obs,     # Static integer: total number of actual observations
    use_surrogate_neg2ll,
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
    if use_surrogate_neg2ll:
        #calculate surrogate neg2ll
        outer_loss = total_log_det + total_quadratic
    else:
        #calculate neg2ll
        neg2_ll = total_log_det + total_quadratic + num_total_obs * jnp.log(2 * jnp.pi)
        outer_loss = neg2_ll

    return outer_loss, (all_log_dets, all_quadratic_terms)

def surrogate_neg2_ll_chol_jit(
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
    neg2_ll = total_log_det + total_quadratic

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

def estimate_ebes_jax_vectorized(
    padded_J: jnp.ndarray,
    padded_residuals: jnp.ndarray,
    time_mask: jnp.ndarray,
    omegas2: jnp.ndarray,
    sigma2: float,
    jitter: float = 1e-6,
) -> jnp.ndarray:
    """
    Calculates Empirical Bayes Estimates (EBEs) in a fully vectorized way.

    Args:
        padded_J: Padded Jacobian matrix.
                  Shape: (n_individuals, max_obs, n_random_effects)
        padded_residuals: Padded residuals.
                          Shape: (n_individuals, max_obs)
        time_mask: Boolean mask indicating valid observations.
                   Shape: (n_individuals, max_obs)
        omegas2: Covariance matrix of the random effects.
                 Shape: (n_random_effects, n_random_effects)
        sigma2: Variance of the residual error.
                Shape: scalar
        jitter: Small value added to the diagonal of omega for numerical stability.

    Returns:
        The estimated EBEs for each individual.
        Shape: (n_individuals, n_random_effects)
    """
    n_effects = omegas2.shape[-1]

    # 1. Calculate the inverse of omega^2 using Cholesky decomposition.
    # This is more numerically stable than a direct inverse.
    # Add jitter to the diagonal to ensure the matrix is positive-definite.
    omega_stable = omegas2 + jnp.eye(n_effects) * jitter
    C, low = jax.scipy.linalg.cho_factor(omega_stable, lower=True)
    omega_inv = jax.scipy.linalg.cho_solve((C, low), jnp.eye(n_effects))

    # 2. Mask the padded values in the batched inputs.
    # `jnp.where` is vectorized by default. `[:, :, None]` broadcasts the mask
    # correctly over the last dimension of the Jacobian.
    J_masked = jnp.where(time_mask[:, :, None], padded_J, 0.0)
    residuals_masked = jnp.where(time_mask, padded_residuals, 0.0)

    # 3. Calculate the left-hand side (A) and right-hand side (rhs)
    #    for all subjects at once using batched matrix multiplication.

    # Transpose the last two dimensions of J_masked for matmul.
    # Shape becomes (n_individuals, n_random_effects, max_obs).
    J_masked_T = jnp.swapaxes(J_masked, -1, -2)

    # Batched matrix-matrix multiplication: (B, N, M) @ (B, M, N) -> (B, N, N)
    A_J = J_masked_T @ J_masked

    # Batched matrix-vector multiplication: (B, N, M) @ (B, M, 1) -> (B, N, 1)
    # We add a dimension to the residuals and squeeze it out after.
    rhs = (J_masked_T @ residuals_masked[..., None]).squeeze(axis=-1)

    # 4. Construct the full A matrix for the linear system.
    # The (sigma2 * omega_inv) term is automatically broadcast across the
    # batch dimension of A_J.
    A = A_J + sigma2 * omega_inv

    # 5. Solve the batched linear system A * b_i = rhs.
    # `jnp.linalg.solve` automatically maps over the leading batch dimension.
    b_i_approx = jnp.linalg.solve(A, rhs)

    return b_i_approx


def _jittable_param_unpack(opt_params, #parameters being optmized by something like scipy's minimize
                            
                            
                            theta_update_to_indices,
                            theta_update_from_indices,
                            theta_total_len,

                            params_idx,
                            fixed_params = None,
                            opt_params_combined_params_idx = None, 
                            fixed_params_combined_params_idx = None,
                            total_n_params = None,
                            init_params_for_scaling = None,

                            use_full_omega = None, 
                            omega_lower_chol_idx = None,
                            omega_diag_size = None,

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

    if use_full_omega:
        rows, cols = omega_lower_chol_idx
        omega_lchol = jnp.zeros((omega_diag_size, omega_diag_size), dtype = np.float64)
        omega_lchol = omega_lchol.at[rows, cols].set(omega)
        omegas_diag = jnp.diag(omega_lchol)
        omegas_diag = jnp.exp(omegas_diag)
        omega_lchol = omega_lchol.at[jnp.diag_indices_from(omega_lchol)].set(omegas_diag)
        omega2 = omega_lchol @ omega_lchol.T
    else:
        omega = jnp.exp(omega)
        omega2 = jnp.diag(omega**2)
    
    #unpack theta
    theta = params[params_idx.theta[0]:params_idx.theta[-1]]
    thetas_work = jnp.zeros(theta_total_len)
    thetas_work = thetas_work.at[theta_update_to_indices].set(theta[theta_update_from_indices])
    
    dynamic_loss_kwargs = {
        'pop_coeff':pop_coeffs, 
        'sigma2':sigma2, 
        'omega2':omega2, 
        'theta':thetas_work, 
        
    } 

    return dynamic_loss_kwargs

def predict_with_params(pop_coeff, 
    sigma2, 
    omega2, 
    theta, 
    theta_data,
    padded_y,
    unpadded_y_len,
    time_mask_y,
    time_mask_J,
    compiled_augdyn_ivp_solver_arr,
    compiled_augdyn_ivp_solver_novmap_arr,
    compiled_ivp_solver_arr,
    compiled_ivp_solver_novmap_arr,
    ode_t0_vals, 
    pop_coeff_for_J_idx,
    use_surrogate_neg2ll,
    b_i):

    n_subjects = time_mask_y.shape[0]
    n_coeffs = pop_coeff.shape[0]
    n_subject_level_eff = pop_coeff_for_J_idx.shape[0]
    
    if theta.shape[0] == 0:
        data_contribution = jnp.zeros((n_subjects, n_coeffs))
    else:
        data_contribution = theta @ theta_data
    
    if b_i is None:
        b_i = jnp.zeros((n_subjects, n_subject_level_eff))
        
    b_i_work = jnp.zeros((n_subjects, n_coeffs))
    b_i_work = b_i_work.at[:,pop_coeff_for_J_idx].set(b_i)
    subject_coeff = b_i_work + pop_coeff
    model_coeffs_i = jnp.exp(data_contribution + subject_coeff)
    padded_full_preds, padded_pred_y = compiled_ivp_solver_arr(ode_t0_vals,
        model_coeffs_i, )
    preds_out = padded_pred_y[time_mask_y]
    
    return preds_out, (padded_pred_y, time_mask_y), model_coeffs_i
    
    
    

def create_jax_objective(
                         unpacker_static_kwargs, 
                         loss_static_kwargs,
                         jittable_loss, 
                         jit_returns = True,
                         ):
    
    initialized_loss = jittable_loss() 
    loss_idx, grad_idx = initialized_loss.loss_val_idx, initialized_loss.grad_val_idx
    grad_is_fdx = initialized_loss.grad_is_fdx
    jittable_loss_fn = initialized_loss.loss_fn
    _grad ,_value_and_grad = initialized_loss.grad_method()       
    
    p_jittable_param_unpack_fit = partial(_jittable_param_unpack, **unpacker_static_kwargs)
    jittable_loss_p_fit = partial(jittable_loss_fn, **loss_static_kwargs)
    
    p_jittable_param_unpack_predict = partial(_jittable_param_unpack, **unpacker_static_kwargs)
    #the predict version of this will be called post fit to generate the fitted set of b_i, the neg2ll for reporting/model comparison, and a set of "fit preds"
    loss_static_kwargs["use_surrogate_neg2ll"] = False
    jittable_loss_p_predict = partial(jittable_loss_fn, **loss_static_kwargs)
    
    def _jax_objective_function(opt_params, ):
        
        dynamic_loss_kwargs = p_jittable_param_unpack_fit(opt_params=opt_params,
                                                                         )
        
        #jittable_loss_p = partial(jittable_loss, **static_loss_kwargs)
        loss, _ = jittable_loss_p_fit(
                **dynamic_loss_kwargs
            )
        
        return loss
    
    def _jax_objective_function_predict(opt_params, ):
        
        dynamic_loss_kwargs = p_jittable_param_unpack_predict(opt_params=opt_params,
                                                                         )
        
        loss_bundle = jittable_loss_p_predict(
                **dynamic_loss_kwargs
            )
        
        return loss_bundle
    
    def _jax_objective_function_focei(opt_params, ):
        
        def f(opt_params):
            dynamic_loss_kwargs = p_jittable_param_unpack_predict(opt_params=opt_params,
                                                                            )
            

            
            loss_bundle = jittable_loss_p_fit(
                **dynamic_loss_kwargs
            )
            return loss_bundle
        if jit_returns:
            f_work = jax.jit(f)
        else:
            f_work = f
        loss_bundle = f_work(opt_params)
        #jax.debug.print("Iteration ")
        loss = loss_bundle[1]['outer_objective_loss']
        
        return loss
    
    def _jax_loss_and_grad_focei(opt_params, ):
        
        def f(opt_params):
            dynamic_loss_kwargs = p_jittable_param_unpack_predict(opt_params=opt_params,
                                                                            )
            

            
            loss_bundle = jittable_loss_p_fit(
                **dynamic_loss_kwargs
            )
            return loss_bundle
        if jit_returns:
            f_work = jax.jit(f)
        else:
            f_work = f
        fvg = _value_and_grad(f_work, has_aux = True)
        loss_bundle, grad = fvg(opt_params)
        #jax.debug.print("Iteration ")
        loss = loss_bundle[1]['outer_objective_loss']
        #loss = loss_bundle[0]
        
        return loss, grad
    
    def _jax_grad_focei(opt_params, ):
        
        def f(opt_params):
            dynamic_loss_kwargs = p_jittable_param_unpack_predict(opt_params=opt_params,
                                                                            )
            

            
            loss_bundle = jittable_loss_p_fit(
                **dynamic_loss_kwargs
            )
            return loss_bundle
        if jit_returns:
            f_work = jax.jit(f)
        else:
            f_work = f
        grad_f = _grad(f_work, has_aux = False)
        grad = grad_f(opt_params)

        
        return grad
    
    def _jax_objective_function_predict_focei(opt_params, ):
        
        dynamic_loss_kwargs = p_jittable_param_unpack_predict(opt_params=opt_params,
                                                                         )
        
        loss_bundle = jittable_loss_p_fit(
                **dynamic_loss_kwargs
            )
        
        loss = loss_bundle[1]['outer_objective_loss']
        
        loss_bundle_out = (
            loss, 
            loss_bundle[1]
        )
        
        return loss_bundle_out
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
    #
    if jit_returns:
        
        if loss_idx == grad_idx:
            fit_obj = jax.jit(_jax_objective_function)
            predict_objective = jax.jit(_jax_objective_function_predict)
            if grad_is_fdx:
                fit_grad = _grad(fit_obj) if _grad is not None else None
                fit_obj_and_grad = _value_and_grad(fit_obj) if _grad is not None else None
            else:
                fit_grad = jax.jit(_grad(fit_obj)) if _grad is not None else None
                fit_obj_and_grad = jax.jit(_value_and_grad(fit_obj)) if _grad is not None else None
            
        else:
            fit_obj = jax.jit(_jax_objective_function_focei)
            predict_objective = jax.jit(_jax_objective_function_predict_focei)
            if grad_is_fdx:
                fit_grad = _jax_grad_focei if _grad is not None else None
                fit_obj_and_grad = _jax_loss_and_grad_focei if _grad is not None else None
            else:
                fit_grad = jax.jit(_jax_grad_focei) if _grad is not None else None
                fit_obj_and_grad = jax.jit(_jax_loss_and_grad_focei) if _grad is not None else None
                
            
        predict_unpack = jax.jit(p_jittable_param_unpack_predict)
    else:
        
        if loss_idx == grad_idx:
            fit_obj = _jax_objective_function
            fit_grad = _grad(fit_obj) if _grad is not None else None
            fit_obj_and_grad = _value_and_grad(fit_obj) if _grad is not None else None
            predict_objective = _jax_objective_function_predict
        else:
            fit_obj = _jax_objective_function_focei
            fit_grad = _jax_grad_focei if _grad is not None else None
            fit_obj_and_grad = _jax_loss_and_grad_focei if _grad is not None else None
            predict_objective = _jax_objective_function_predict_focei
        
        
        predict_unpack = p_jittable_param_unpack_predict
    return (fit_obj, fit_grad, fit_obj_and_grad), (predict_objective, predict_unpack)
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


def estimate_b_i_vmapped_fdx(
    initial_b_i_batch,
    padded_y_batch,
    data_contribution_batch,
    ode_t0_vals_batch,
    time_mask_y_batch,
    pop_coeff,
    sigma2,
    omega2,
    n_random_effects,
    compiled_ivp_solver,
    compiled_augdyn_ivp_solver,
    pop_coeff_w_bi_idx,
    use_surrogate_neg2ll,
    **kwargs,  # Absorb unused kwargs
):
    """Estimates b_i for all subjects by vmapping the custom VJP function."""
    print('Compiling `estimate_b_i_vmapped` with custom VJP')

    # Vmap the single-subject optimization function with the custom VJP.
    # `in_axes` specifies which arguments to map over (0) vs. broadcast (None).
    
    @jax.jit
    def estimate_single_b_i_impl(
    initial_b_i,
    padded_y_i,
    data_contrib_i,
    ode_t0_i,
    time_mask_y_i,
    # Shared parameters we want to differentiate with respect to
    pop_coeff,
    sigma2,
    omega2,
    # Other static arguments
    #n_random_effects,
    #compiled_ivp_solver,
    #compiled_augdyn_ivp_solver,
    pop_coeff_w_bi_idx,
    #use_surrogate_neg2ll,
    ):
        """The implementation of the optimization for one subject."""
        obj_fn = lambda b_i: FOCE_inner_loss_fn(
            b_i=b_i,
            padded_y_i=padded_y_i,
            pop_coeff_i=pop_coeff,
            data_contribution_i=data_contrib_i,
            sigma2=sigma2,
            omega2=omega2,
            ode_t0_val_i=ode_t0_i,
            time_mask_y_i=time_mask_y_i,
            n_random_effects=n_random_effects,
            compiled_ivp_solver=compiled_ivp_solver,
            pop_coeff_w_bi_idx=pop_coeff_w_bi_idx,
            use_surrogate_neg2ll=use_surrogate_neg2ll,
        )

        optimizer = optax.adam(learning_rate=0.1)
        opt_state = optimizer.init(initial_b_i)
        grad_fn = jax.grad(obj_fn)
        omega_is_near_zero = jnp.diag(jnp.sqrt(omega2)) < 1e-5

        def update_step(i, state_tuple):
            params, opt_state = state_tuple
            grads = grad_fn(params)
            safe_grads = jnp.where(omega_is_near_zero, 0.0, grads)
            updates, opt_state = optimizer.update(safe_grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state

        estimated_b_i, _ = jax.lax.fori_loop(0, 100, update_step, (initial_b_i, opt_state))
        final_inner_loss_value = obj_fn(estimated_b_i)

        # --- Hessian Approximation ---
        def predict_fn(b_i_for_pred):
                # This logic is copied from your FOCE_inner_loss_fn
                b_i_work = jnp.zeros_like(pop_coeff)
                b_i_work = b_i_work.at[pop_coeff_w_bi_idx].set(b_i_for_pred)
                combined_coeffs = pop_coeff + b_i_work
                model_coeffs_i = jnp.exp(data_contrib_i + combined_coeffs)
                
                # We don't need the "safe_coeffs" logic here, as we're already at the optimum
                _, _, _, J_conc_full = compiled_augdyn_ivp_solver(ode_t0_i, model_coeffs_i)
                
                # We need the elements in J corresponding to the mask, implemented below
                return  J_conc_full

        J = predict_fn(estimated_b_i)
        mask_expanded = time_mask_y_i[:, None]
        J_masked = J * mask_expanded
        #mask J here, but what shape is it, how should `time_mask_y_i` be reshaped/tiled?
        _sigma2 = sigma2[0]
        H_approx = (J_masked.T @ J_masked) / _sigma2
        
                
        L, _ = jax.scipy.linalg.cho_factor(omega2, lower=True)

        # 2. Create an identity matrix of the same size
        identity = jnp.eye(omega2.shape[0], dtype=omega2.dtype)

        # 3. Efficiently solve for the inverse using the Cholesky factor
        inv_omega2 = jax.scipy.linalg.cho_solve((L, True), identity)

        # 4. Compute the full Hessian needed for the FOCEi term
        H_foce = H_approx + 2 * inv_omega2

        return estimated_b_i, H_foce, final_inner_loss_value
    
    def _estimate_single_b_i_fwd(*args):
        # args now only contains the 9 JAX array arguments
        outputs = estimate_single_b_i_impl(*args)
        residuals = args # Save all array args as residuals
        return outputs, residuals
    
    def _estimate_single_b_i_bwd(residuals, g):
        (initial_b_i, padded_y_i, data_contrib_i, ode_t0_i, time_mask_y_i,
         pop_coeff, sigma2, omega2, pop_coeff_w_bi_idx) = residuals
        g_b_i, g_H, g_loss = g

        # This helper function automatically closes over the static arguments
        def f(dc, pc, s2, o2):
            return estimate_single_b_i_impl(
                initial_b_i, padded_y_i, dc, ode_t0_i, time_mask_y_i,
                pc, s2, o2, pop_coeff_w_bi_idx
            )

        def scalar_loss(dc, pc, s2, o2):
            y_b_i, y_H, y_loss = f(dc, pc, s2, o2)
            return (jnp.sum(g_b_i * y_b_i) +
                    jnp.sum(g_H * y_H) +
                    jnp.sum(g_loss * y_loss))

        grad_fn = fdx.fgrad(scalar_loss, argnums=(0, 1, 2, 3), offsets=fdx.Offset(accuracy=2))
        grads_tuple = grad_fn(data_contrib_i, pop_coeff, sigma2, omega2)
        grad_data_contrib, grad_pop_coeff, grad_sigma2, grad_omega2 = grads_tuple

        # Return tuple matches the new, shorter signature of `estimate_single_b_i`
        return (None, None, grad_data_contrib, None, None,
                grad_pop_coeff, grad_sigma2, grad_omega2, None)
    
    @jax.custom_vjp
    def estimate_single_b_i(
        initial_b_i, padded_y_i, data_contrib_i, ode_t0_i, time_mask_y_i,
        pop_coeff, sigma2, omega2, pop_coeff_w_bi_idx
    ):
        return estimate_single_b_i_impl(
            initial_b_i, padded_y_i, data_contrib_i, ode_t0_i, time_mask_y_i,
            pop_coeff, sigma2, omega2, pop_coeff_w_bi_idx
        )
        
    estimate_single_b_i.defvjp(_estimate_single_b_i_fwd, _estimate_single_b_i_bwd)
    
    vmapped_optimizer = jax.vmap(
        estimate_single_b_i,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None),
        out_axes=0
    )

    # Execute the vmapped function
    all_b_i_estimates, all_hessians, all_final_inner_loss = vmapped_optimizer(
        initial_b_i_batch,
        padded_y_batch,
        data_contribution_batch,
        ode_t0_vals_batch,
        time_mask_y_batch,
        pop_coeff,
        sigma2,
        omega2,
        pop_coeff_w_bi_idx
    )

    return all_b_i_estimates, all_hessians, all_final_inner_loss

def estimate_b_i_vmapped_ift_ALT(
    initial_b_i_batch,
    padded_y_batch,
    data_contribution_batch,
    ode_t0_vals_batch,
    time_mask_y_batch,
    pop_coeff,
    sigma2,
    omega2,
    n_random_effects,
    compiled_ivp_solver,
    compiled_augdyn_ivp_solver,
    pop_coeff_w_bi_idx,
    use_surrogate_neg2ll,
    **kwargs,  # Absorb unused kwargs
):
    """Estimates b_i for all subjects by vmapping the custom VJP function."""
    print('Compiling `estimate_b_i_vmapped` with custom VJP')

    # Vmap the single-subject optimization function with the custom VJP.
    # `in_axes` specifies which arguments to map over (0) vs. broadcast (None).
    
    def update_step(i, state_tuple):
        # Unpack everything from the state
        params_b_i, opt_state, padded_y_i, pop_coeff, data_contrib_i, \
        sigma2, omega2, ode_t0_i, time_mask_y_i, pop_coeff_w_bi_idx = state_tuple

        omega_is_near_zero = jnp.diag(jnp.sqrt(omega2)) < 1e-5
        
        # Reconstruct the loss function call for this step
        loss, grads = jax.value_and_grad(FOCE_inner_loss_fn)(
            params_b_i, padded_y_i=padded_y_i, pop_coeff_i=pop_coeff,
            data_contribution_i=data_contrib_i, sigma2=sigma2, omega2=omega2,
            ode_t0_val_i=ode_t0_i, time_mask_y_i=time_mask_y_i,
            n_random_effects=n_random_effects,
            compiled_ivp_solver=compiled_ivp_solver,
            pop_coeff_w_bi_idx=pop_coeff_w_bi_idx,
            use_surrogate_neg2ll=use_surrogate_neg2ll
        )
        safe_grads = jnp.where(omega_is_near_zero, 0.0, grads)
        # The optimizer logic is the same
        updates, opt_state = optimizer.update(safe_grads, opt_state, params_b_i)
        new_params_b_i = optax.apply_updates(params_b_i, updates)
        
        # Repack the state for the next iteration
        return (new_params_b_i, opt_state, padded_y_i, pop_coeff, data_contrib_i,
                sigma2, omega2, ode_t0_i, time_mask_y_i, pop_coeff_w_bi_idx)
    
    optimizer = optax.adam(learning_rate=0.1)
    
    #@jax.jit
    def estimate_single_b_i_impl(
        initial_b_i, padded_y_i, data_contrib_i, ode_t0_i, time_mask_y_i,
        pop_coeff, sigma2, omega2, pop_coeff_w_bi_idx
    ):
        opt_state = optimizer.init(initial_b_i)

        # The state tuple now includes ALL data needed by the update_step
        initial_state = (initial_b_i, opt_state, padded_y_i, pop_coeff,
                         data_contrib_i, sigma2, omega2, ode_t0_i,
                         time_mask_y_i, pop_coeff_w_bi_idx)
        #jax.debug.print("Starting Inner Optimization")
        # Run the loop with the static update function
        final_state = jax.lax.fori_loop(0, 100, update_step, initial_state)
        #jax.debug.print("Completed Inner Optimization")
        estimated_b_i = final_state[0] # Unpack the final b_i
        
        final_inner_loss_value = FOCE_inner_loss_fn(
            b_i=estimated_b_i, padded_y_i=padded_y_i, data_contribution_i=data_contrib_i, 
            ode_t0_val_i=ode_t0_i, time_mask_y_i=time_mask_y_i, pop_coeff_i=pop_coeff,
            sigma2=sigma2, omega2=omega2, pop_coeff_w_bi_idx=pop_coeff_w_bi_idx, 
            n_random_effects=n_random_effects, compiled_ivp_solver=compiled_ivp_solver, 
            use_surrogate_neg2ll=use_surrogate_neg2ll
        )
        #jax.debug.print("Completed Estimation of Final Inner Optimization Loss")
        # --- Hessian Approximation ---
        def predict_fn(b_i_for_pred):
                # This logic is copied from your FOCE_inner_loss_fn
                b_i_work = jnp.zeros_like(pop_coeff)
                b_i_work = b_i_work.at[pop_coeff_w_bi_idx].set(b_i_for_pred)
                combined_coeffs = pop_coeff + b_i_work
                model_coeffs_i = jnp.exp(data_contrib_i + combined_coeffs)
                
                # We don't need the "safe_coeffs" logic here, as we're already at the optimum
                _, _, _, J_conc_full = compiled_augdyn_ivp_solver(ode_t0_i, model_coeffs_i)
                
                # We need the elements in J corresponding to the mask, implemented below
                return  J_conc_full

        J = predict_fn(estimated_b_i)
        mask_expanded = time_mask_y_i[:, None]
        J_masked = J * mask_expanded
        #mask J here, but what shape is it, how should `time_mask_y_i` be reshaped/tiled?
        _sigma2 = sigma2[0]
        H_approx = (J_masked.T @ J_masked) / _sigma2
        
                
        L, _ = jax.scipy.linalg.cho_factor(omega2, lower=True)

        # 2. Create an identity matrix of the same size
        identity = jnp.eye(omega2.shape[0], dtype=omega2.dtype)

        # 3. Efficiently solve for the inverse using the Cholesky factor
        inv_omega2 = jax.scipy.linalg.cho_solve((L, True), identity)

        # 4. Compute the full Hessian needed for the FOCEi term
        H_foce = H_approx + 2 * inv_omega2

        return estimated_b_i, H_foce, final_inner_loss_value
    
    def _estimate_single_b_i_fwd(*args):
        # args now only contains the 9 JAX array arguments
        outputs = estimate_single_b_i_impl(*args)
        estimated_b_i, H_foce, _ = outputs

        residuals = (estimated_b_i, H_foce) + args
        return outputs, residuals
    
    def _estimate_single_b_i_bwd(residuals, g):
        (estimated_b_i, H_foce, initial_b_i, padded_y_i, data_contrib_i, ode_t0_i, time_mask_y_i,
         pop_coeff, sigma2, omega2, pop_coeff_w_bi_idx) = residuals
        g_b_i, g_H, g_loss = g
        #jax.debug.print("Starting Reverse Pass")
        def grad_inner_loss_fn(b_i, dc, pc, s2, o2):
            return jax.grad(FOCE_inner_loss_fn, argnums=0)(
                b_i, padded_y_i=padded_y_i, pop_coeff_i=pc,
                data_contribution_i=dc, sigma2=s2, omega2=o2,
                ode_t0_val_i=ode_t0_i, time_mask_y_i=time_mask_y_i,
                n_random_effects=n_random_effects,
                compiled_ivp_solver=compiled_ivp_solver,
                pop_coeff_w_bi_idx=pop_coeff_w_bi_idx,
                use_surrogate_neg2ll=use_surrogate_neg2ll
            )

        # 3. Solve the core linear system of the IFT: H^T @ v = g
        #    Since H is symmetric, this is just H @ v = g.
        #    We only use g_b_i, the cotangent for the optimized parameters.
        #H_foce_analytical = jax.jacobian(grad_inner_loss_fn, argnums=0)(
        #    estimated_b_i, data_contrib_i, pop_coeff, sigma2, omega2
        #)
        v = jax.scipy.linalg.solve(H_foce, g_b_i,)# sym_pos=True)
        
        # 4. Compute the cross-term Jacobians and final gradients.
        #    This tells us how the inner gradient changes w.r.t. outer parameters.
        
        # Gradient w.r.t. data_contrib_i
        J_cross_dc = jax.jacfwd(grad_inner_loss_fn, argnums=1)(
            estimated_b_i, data_contrib_i, pop_coeff, sigma2, omega2
        )
        grad_data_contrib = -v @ J_cross_dc
        
        # Gradient w.r.t. pop_coeff
        J_cross_pc = jax.jacobian(grad_inner_loss_fn, argnums=2)(
            estimated_b_i, data_contrib_i, pop_coeff, sigma2, omega2
        )
        grad_pop_coeff = -v @ J_cross_pc

        # Gradient w.r.t. sigma2
        J_cross_s2 = jax.jacobian(grad_inner_loss_fn, argnums=3)(
            estimated_b_i, data_contrib_i, pop_coeff, sigma2, omega2
        )
        grad_sigma2 = -v @ J_cross_s2

        # Gradient w.r.t. omega2
        J_cross_o2 = jax.jacobian(grad_inner_loss_fn, argnums=4)(
            estimated_b_i, data_contrib_i, pop_coeff, sigma2, omega2
        )
        grad_omega2 = -v @ J_cross_o2

        # Note: A complete implementation would also add the gradient contributions
        # from g_H and g_loss. This version includes the main IFT path, which is
        # the most significant and complex part.
        
        # Return tuple matches the new, shorter signature of `estimate_single_b_i`
        return (None, None, grad_data_contrib, None, None,
                grad_pop_coeff, grad_sigma2, grad_omega2, None)
    
    @jax.custom_vjp
    def estimate_single_b_i(
        initial_b_i, padded_y_i, data_contrib_i, ode_t0_i, time_mask_y_i,
        pop_coeff, sigma2, omega2, pop_coeff_w_bi_idx
    ):
        return estimate_single_b_i_impl(
            initial_b_i, padded_y_i, data_contrib_i, ode_t0_i, time_mask_y_i,
            pop_coeff, sigma2, omega2, pop_coeff_w_bi_idx
        )
        
    estimate_single_b_i.defvjp(_estimate_single_b_i_fwd, _estimate_single_b_i_bwd)
    
    vmapped_optimizer = jax.vmap(
        estimate_single_b_i,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None),
        out_axes=0
    )

    # Execute the vmapped function
    all_b_i_estimates, all_hessians, all_final_inner_loss = vmapped_optimizer(
        initial_b_i_batch,
        padded_y_batch,
        data_contribution_batch,
        ode_t0_vals_batch,
        time_mask_y_batch,
        pop_coeff,
        sigma2,
        omega2,
        pop_coeff_w_bi_idx
    )

    return all_b_i_estimates, all_hessians, all_final_inner_loss

def estimate_b_i_vmapped_ift(
    initial_b_i_batch,
    padded_y_batch,
    data_contribution_batch,
    ode_t0_vals_batch,
    time_mask_y_batch,
    pop_coeff,
    sigma2,
    omega2,
    n_random_effects,
    compiled_ivp_solver,
    compiled_augdyn_ivp_solver,
    pop_coeff_w_bi_idx,
    use_surrogate_neg2ll,
    **kwargs,  # Absorb unused kwargs
):
    """Estimates b_i for all subjects by vmapping the custom VJP function."""
    print('Compiling `estimate_b_i_vmapped` with custom VJP')

    # Vmap the single-subject optimization function with the custom VJP.
    # `in_axes` specifies which arguments to map over (0) vs. broadcast (None).
    
    @jax.jit
    def estimate_single_b_i_impl(
    initial_b_i,
    padded_y_i,
    data_contrib_i,
    ode_t0_i,
    time_mask_y_i,
    # Shared parameters we want to differentiate with respect to
    pop_coeff,
    sigma2,
    omega2,
    # Other static arguments
    #n_random_effects,
    #compiled_ivp_solver,
    #compiled_augdyn_ivp_solver,
    pop_coeff_w_bi_idx,
    #use_surrogate_neg2ll,
    ):
        """The implementation of the optimization for one subject."""
        obj_fn = lambda b_i: FOCE_inner_loss_fn(
            b_i=b_i,
            padded_y_i=padded_y_i,
            pop_coeff_i=pop_coeff,
            data_contribution_i=data_contrib_i,
            sigma2=sigma2,
            omega2=omega2,
            ode_t0_val_i=ode_t0_i,
            time_mask_y_i=time_mask_y_i,
            n_random_effects=n_random_effects,
            compiled_ivp_solver=compiled_ivp_solver,
            pop_coeff_w_bi_idx=pop_coeff_w_bi_idx,
            use_surrogate_neg2ll=use_surrogate_neg2ll,
        )

        optimizer = optax.adam(learning_rate=0.1)
        opt_state = optimizer.init(initial_b_i)
        grad_fn = jax.grad(obj_fn)
        omega_is_near_zero = jnp.diag(jnp.sqrt(omega2)) < 1e-5

        def update_step(i, state_tuple):
            params, opt_state = state_tuple
            grads = grad_fn(params)
            safe_grads = jnp.where(omega_is_near_zero, 0.0, grads)
            updates, opt_state = optimizer.update(safe_grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state

        estimated_b_i, _ = jax.lax.fori_loop(0, 100, update_step, (initial_b_i, opt_state))
        final_inner_loss_value = obj_fn(estimated_b_i)

        # --- Hessian Approximation ---
        def predict_fn(b_i_for_pred):
                # This logic is copied from your FOCE_inner_loss_fn
                b_i_work = jnp.zeros_like(pop_coeff)
                b_i_work = b_i_work.at[pop_coeff_w_bi_idx].set(b_i_for_pred)
                combined_coeffs = pop_coeff + b_i_work
                model_coeffs_i = jnp.exp(data_contrib_i + combined_coeffs)
                
                # We don't need the "safe_coeffs" logic here, as we're already at the optimum
                _, _, _, J_conc_full = compiled_augdyn_ivp_solver(ode_t0_i, model_coeffs_i)
                
                # We need the elements in J corresponding to the mask, implemented below
                return  J_conc_full

        J = predict_fn(estimated_b_i)
        mask_expanded = time_mask_y_i[:, None]
        J_masked = J * mask_expanded
        #mask J here, but what shape is it, how should `time_mask_y_i` be reshaped/tiled?
        _sigma2 = sigma2[0]
        H_approx = (J_masked.T @ J_masked) / _sigma2
        
                
        L, _ = jax.scipy.linalg.cho_factor(omega2, lower=True)

        # 2. Create an identity matrix of the same size
        identity = jnp.eye(omega2.shape[0], dtype=omega2.dtype)

        # 3. Efficiently solve for the inverse using the Cholesky factor
        inv_omega2 = jax.scipy.linalg.cho_solve((L, True), identity)

        # 4. Compute the full Hessian needed for the FOCEi term
        H_foce = H_approx + 2 * inv_omega2

        return estimated_b_i, H_foce, final_inner_loss_value
    
    def _estimate_single_b_i_fwd(*args):
        # args now only contains the 9 JAX array arguments
        outputs = estimate_single_b_i_impl(*args)
        estimated_b_i, H_foce, _ = outputs

        residuals = (estimated_b_i, H_foce) + args
        return outputs, residuals
    
    def _estimate_single_b_i_bwd(residuals, g):
        (estimated_b_i, H_foce, initial_b_i, padded_y_i, data_contrib_i, ode_t0_i, time_mask_y_i,
         pop_coeff, sigma2, omega2, pop_coeff_w_bi_idx) = residuals
        g_b_i, g_H, g_loss = g

        def grad_inner_loss_fn(b_i, dc, pc, s2, o2):
            return jax.grad(FOCE_inner_loss_fn, argnums=0)(
                b_i, padded_y_i=padded_y_i, pop_coeff_i=pc,
                data_contribution_i=dc, sigma2=s2, omega2=o2,
                ode_t0_val_i=ode_t0_i, time_mask_y_i=time_mask_y_i,
                n_random_effects=n_random_effects,
                compiled_ivp_solver=compiled_ivp_solver,
                pop_coeff_w_bi_idx=pop_coeff_w_bi_idx,
                use_surrogate_neg2ll=use_surrogate_neg2ll
            )

        # 3. Solve the core linear system of the IFT: H^T @ v = g
        #    Since H is symmetric, this is just H @ v = g.
        #    We only use g_b_i, the cotangent for the optimized parameters.
        #H_foce_analytical = jax.jacobian(grad_inner_loss_fn, argnums=0)(
        #    estimated_b_i, data_contrib_i, pop_coeff, sigma2, omega2
        #)
        v = jax.scipy.linalg.solve(H_foce, g_b_i,)# sym_pos=True)
        
        # 4. Compute the cross-term Jacobians and final gradients.
        #    This tells us how the inner gradient changes w.r.t. outer parameters.
        
        # Gradient w.r.t. data_contrib_i
        J_cross_dc = jax.jacobian(grad_inner_loss_fn, argnums=1)(
            estimated_b_i, data_contrib_i, pop_coeff, sigma2, omega2
        )
        grad_data_contrib = -v @ J_cross_dc
        
        # Gradient w.r.t. pop_coeff
        J_cross_pc = jax.jacobian(grad_inner_loss_fn, argnums=2)(
            estimated_b_i, data_contrib_i, pop_coeff, sigma2, omega2
        )
        grad_pop_coeff = -v @ J_cross_pc

        # Gradient w.r.t. sigma2
        J_cross_s2 = jax.jacobian(grad_inner_loss_fn, argnums=3)(
            estimated_b_i, data_contrib_i, pop_coeff, sigma2, omega2
        )
        grad_sigma2 = -v @ J_cross_s2

        # Gradient w.r.t. omega2
        J_cross_o2 = jax.jacobian(grad_inner_loss_fn, argnums=4)(
            estimated_b_i, data_contrib_i, pop_coeff, sigma2, omega2
        )
        grad_omega2 = -v @ J_cross_o2

        # Note: A complete implementation would also add the gradient contributions
        # from g_H and g_loss. This version includes the main IFT path, which is
        # the most significant and complex part.
        
        # Return tuple matches the new, shorter signature of `estimate_single_b_i`
        return (None, None, grad_data_contrib, None, None,
                grad_pop_coeff, grad_sigma2, grad_omega2, None)
    
    @jax.custom_vjp
    def estimate_single_b_i(
        initial_b_i, padded_y_i, data_contrib_i, ode_t0_i, time_mask_y_i,
        pop_coeff, sigma2, omega2, pop_coeff_w_bi_idx
    ):
        return estimate_single_b_i_impl(
            initial_b_i, padded_y_i, data_contrib_i, ode_t0_i, time_mask_y_i,
            pop_coeff, sigma2, omega2, pop_coeff_w_bi_idx
        )
        
    estimate_single_b_i.defvjp(_estimate_single_b_i_fwd, _estimate_single_b_i_bwd)
    
    vmapped_optimizer = jax.vmap(
        estimate_single_b_i,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None),
        out_axes=0
    )

    # Execute the vmapped function
    all_b_i_estimates, all_hessians, all_final_inner_loss = vmapped_optimizer(
        initial_b_i_batch,
        padded_y_batch,
        data_contribution_batch,
        ode_t0_vals_batch,
        time_mask_y_batch,
        pop_coeff,
        sigma2,
        omega2,
        pop_coeff_w_bi_idx
    )

    return all_b_i_estimates, all_hessians, all_final_inner_loss

def estimate_b_i_vmapped(
    # Batched inputs (one entry per subject)
    initial_b_i_batch,
    padded_y_batch,
    data_contribution_batch,
    ode_t0_vals_batch,
    time_mask_y_batch,
    #unpadded_y_len_batch,
    # Shared inputs (same for all subjects)
    pop_coeff,
    sigma2,
    omega2,
    n_random_effects,
    compiled_ivp_solver,
    compiled_augdyn_ivp_solver,
    pop_coeff_w_bi_idx,
    use_surrogate_neg2ll,
    **kwargs
):
    """
    Estimates b_i for all subjects in parallel using a vmapped Optax optimizer.
    """
    print('Compiling `estimate_b_i_vmapped`')
    #jax.debug.print("""
    #                initial_b_i_batch = {initial_b_i_batch}
    #                padded_y_batch = {padded_y_batch}
    #                data_contribution_batch = {data_contribution_batch}
    #                ode_t0_vals_batch = {ode_t0_vals_batch}
    #                time_mask_y_batch = {time_mask_y_batch}
    #                pop_coeff = {pop_coeff}
    #                sigma2 = {sigma2}
    #                omega2 = {omega2}
    #                n_random_effects = {n_random_effects}
    #                compiled_ivp_solver = {compiled_ivp_solver}
    #                pop_coeff_w_bi_idx = {pop_coeff_w_bi_idx}
    #                """, 
    #                initial_b_i_batch = initial_b_i_batch.shape,
    #                padded_y_batch = padded_y_batch.shape,
    #                data_contribution_batch = data_contribution_batch.shape,
    #                ode_t0_vals_batch = ode_t0_vals_batch.shape,
    #                time_mask_y_batch = time_mask_y_batch.shape,
    #                pop_coeff = pop_coeff.shape,
    #                sigma2 = sigma2.shape,
    #                omega2 = omega2.shape,
    #                n_random_effects = n_random_effects,
    #                compiled_ivp_solver = compiled_ivp_solver,
    #                pop_coeff_w_bi_idx = pop_coeff_w_bi_idx.shape,
    #                )
    # Define the optimization for a single subject
    def _estimate_single_b_i(initial_b_i,
                             padded_y_i,
                             data_contrib_i,
                             ode_t0_i,
                             time_mask_y_i,
                             #unpadded_y_i_len,
                             ):
        
        # The objective function with data for this subject closed over
        obj_fn = lambda b_i: FOCE_inner_loss_fn(
            b_i = b_i, 
            padded_y_i = padded_y_i,
            pop_coeff_i=pop_coeff,
            data_contribution_i = data_contrib_i,
            sigma2 = sigma2,
            omega2 = omega2,
            ode_t0_val_i = ode_t0_i,
            time_mask_y_i=time_mask_y_i,
            n_random_effects=n_random_effects, 
            compiled_ivp_solver=compiled_ivp_solver,
            pop_coeff_w_bi_idx = pop_coeff_w_bi_idx,
            use_surrogate_neg2ll = use_surrogate_neg2ll,
            
        )

        # 2. Set up an optax optimizer
        learning_rate = 0.08  # Tune as needed
        num_inner_steps = 200 # Tune as needed
        lr_schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=num_inner_steps,
        decay_rate=0.1 
    )
        optimizer = optax.adam(learning_rate=lr_schedule)
        opt_state = optimizer.init(initial_b_i)
        
        
        grad_fn = jax.grad(obj_fn)
        #if omega is near zero, then optimzing b_i will probably lead to 
        #very large pop coeffs w/ very small b_i, effectively canceling 
        #eachother out in a mathematically valid, but practicaly invalid way
        omega_is_near_zero = jnp.diag(jnp.sqrt(omega2)) < 1e-5 
        
        def update_step(i, state_tuple):
            params, opt_state, has_converged, loss_history, norm_history = state_tuple
            
            def do_update(operand):
                params, opt_state, lh, nh = operand
                grads = grad_fn(params)
                
                #sanitize the grads for the whole inner optmization for the current iteration 
                #based on the heuristic calculated above (`omega_is_near_zero`)
                safe_grads = jnp.where(omega_is_near_zero, 0.0, grads)
                grad_norm = jnp.linalg.norm(safe_grads)
                converged = grad_norm < 1e-2
                updates, opt_state = optimizer.update(safe_grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                
                current_loss = obj_fn(new_params)
                new_lh = lh.at[i].set(current_loss)
                new_nh = nh.at[i].set(grad_norm)
                return new_params, opt_state, converged, new_lh, new_nh
            def do_nothing(operand):
                params, opt_state, lh, nh = operand
                #current_loss = obj_fn(params)
                new_lh = lh.at[i].set(0.0)
                new_nh = nh.at[i].set(0.0)
                #jax.debug.print("Inner Optmizer Converged")
                return params, opt_state, True, new_lh, new_nh
            
            return jax.lax.cond(has_converged,
                do_nothing,
                do_update,
                (params, opt_state, loss_history, norm_history))
            
        
        loss_history_init = jnp.zeros(num_inner_steps)
        gradnorm_history_init = jnp.zeros(num_inner_steps)
        initial_state = (initial_b_i, opt_state, False, loss_history_init, gradnorm_history_init)
        estimated_b_i, opt_state, has_converged, loss_history, norm_history = jax.lax.fori_loop(
            0, num_inner_steps, update_step, initial_state
        )
        #jax.debug.print("""Inner Optimization finished with convergence status: {s}
        #                loss history: {h}
        #                norm history: {n}
        #                """, s = has_converged, h = loss_history, n = norm_history)
        
        def predict_fn(b_i_for_pred):
            # This logic is copied from your FOCE_inner_loss_fn
            b_i_work = jnp.zeros_like(pop_coeff)
            b_i_work = b_i_work.at[pop_coeff_w_bi_idx].set(b_i_for_pred)
            combined_coeffs = pop_coeff + b_i_work
            model_coeffs_i = jnp.exp(data_contrib_i + combined_coeffs)
            
            # We don't need the "safe_coeffs" logic here, as we're already at the optimum
            _, _, _, J_conc_full = compiled_augdyn_ivp_solver(ode_t0_i, model_coeffs_i)
            
            # We need the predictions corresponding to the mask
            return  J_conc_full
        
        final_inner_loss_value = obj_fn(estimated_b_i)

        J = predict_fn(estimated_b_i)
        mask_expanded = time_mask_y_i[:, None]
        J_masked = J * mask_expanded
        #mask J here, but what shape is it, how should `time_mask_y_i` be reshaped/tiled?
        _sigma2 = sigma2[0]
        H_approx = 2*((J_masked.T @ J_masked) / _sigma2)
        
        L, _ = jax.scipy.linalg.cho_factor(omega2, lower=True)

        # 2. Create an identity matrix of the same size
        identity = jnp.eye(omega2.shape[0], dtype=omega2.dtype)

        # 3. Efficiently solve for the inverse using the Cholesky factor
        inv_omega2 = jax.scipy.linalg.cho_solve((L, True), identity)

        # 4. Compute the full Hessian needed for the FOCEi term
        H_foce = H_approx + (2 * inv_omega2)
        
        
        #hessian_matrix = jnp.ones_like(estimated_b_i)
        # Return the optimized parameters
        return estimated_b_i, H_foce, final_inner_loss_value

    # Vmap the single-subject optimization function
    # `in_axes` specifies which arguments to map over. `None` means broadcast.
    vmapped_optimizer = jax.vmap(
        _estimate_single_b_i,
        in_axes=(0, 0, 0, 0, 0, ),  # Map over all batched inputs
    )
    
    # Execute the vmapped function
    all_b_i_estimates, all_hessians, all_final_inner_loss = vmapped_optimizer(
        initial_b_i_batch,
        padded_y_batch,
        data_contribution_batch,
        ode_t0_vals_batch,
        time_mask_y_batch,
        #unpadded_y_len_batch
    )
    
    return all_b_i_estimates, all_hessians, all_final_inner_loss

def FOCE_inner_loss_fn(
    b_i,  # The parameters we are optimizing (random effects)
    # --- Static data for this subject (won't change during inner opt) ---
    padded_y_i,
    #unpadded_y_i_len,
    pop_coeff_i,
    data_contribution_i,
    sigma2,
    omega2,
    ode_t0_val_i,
    time_mask_y_i,
    n_random_effects,
    compiled_ivp_solver,
    pop_coeff_w_bi_idx,
    use_surrogate_neg2ll,
):
    """
    Calculates the conditional negative 2 log-likelihood for a single subject.
    This function is pure and written entirely in JAX.
    """
    
    print("Compiling `FOCE_inner_loss_fn`")
    # Combine population and random effects to get subject-specific coefficients
    # This assumes b_i are additive adjustments on the log-scale
    b_i_work = jnp.zeros(pop_coeff_i.shape[0])
    b_i_work = b_i_work.at[pop_coeff_w_bi_idx].set(b_i)
    #jax.debug.print("pop_coeff_i shape: {s}", s = pop_coeff_i.shape)
    #jax.debug.print("pop_coeff_i val: {s}", s = pop_coeff_i)
    #jax.debug.print("b_i_work shape: {s}", s = b_i_work.shape)
    #jax.debug.print("b_i_work val: {s}", s = b_i_work)
    combined_coeffs = pop_coeff_i + b_i_work
    model_coeffs_i = jnp.exp(data_contribution_i + combined_coeffs)
    #jax.debug.print("model_coeffs_i shape: {s}", s = model_coeffs_i.shape)
    #jax.debug.print("model_coeffs_i val: {s}", s = model_coeffs_i)
    #jax.debug.print("combined_coeffs calc: {s} + {x} = {y}", s = pop_coeff_i, x = b_i_work, y = combined_coeffs)
    is_bad_state = jnp.any(~jnp.isfinite(model_coeffs_i)) | jnp.any(model_coeffs_i < 1e-9)
    #jax.debug.print("is_bad_state val: {s}", s = is_bad_state)
    safe_coeffs = jnp.ones_like(model_coeffs_i)
    solver_coeffs = jnp.where(is_bad_state, safe_coeffs, model_coeffs_i)
    # --- Data Likelihood Part ---
    # Solve the ODE for this individual with the current b_i guess
    #jax.debug.print("model_coeffs_i shape: {s}", s = model_coeffs_i.shape)
    #jax.debug.print("ode_t0_val_i shape: {s}", s = ode_t0_val_i.shape)
    
    #_solver_model_coeffs_i = jnp.reshape(solver_coeffs, (1, -1))
    #_ode_t0_val_i = jnp.reshape(ode_t0_val_i, (1, -1))
    #jax.debug.print("model_coeffs_i shape: {s}", s = model_coeffs_i.shape)
    #jax.debug.print("ode_t0_val_i shape: {s}", s = ode_t0_val_i.shape)
    padded_full_preds_i, padded_pred_y_i = compiled_ivp_solver( ode_t0_val_i, solver_coeffs,)
    #jax.debug.print("padded_preds_i shape: {s}", s = padded_pred_y_i.shape)
    #jax.debug.print("time_mask_y_i shape: {s}", s = time_mask_y_i.shape)
    #jax.debug.print("padded_y_i shape: {s}", s = padded_y_i.shape)
    masked_residuals_i = jnp.where(time_mask_y_i, padded_y_i - padded_pred_y_i, 0.0).flatten()
    # Compute residuals only for observed time points
    #jax.debug.print("masked_residuals_i shape: {s}", s = masked_residuals_i.shape)
    #n_t = unpadded_y_i_len # Number of actual observations for subject i
    #jax.debug.print("unpadded_y_i_len shape: {s}", s = unpadded_y_i_len.shape)
    sum_sq_residuals = jnp.sum(masked_residuals_i**2)
    #jax.debug.print("sum_sq_residuals shape: {s}", s = sum_sq_residuals.shape)
    # This assumes an additive error model, as in the original code
    _sigma2 = sigma2[0]
    if use_surrogate_neg2ll:
        log_likelihood_constant = jnp.log(_sigma2)
    else:
        log_likelihood_constant = jnp.log(2 * jnp.pi) + jnp.log(_sigma2)
    #tmp_ll_const = jnp.repeat(log_likelihood_constant, time_mask_y_i.shape[0])
    n_t_term = jnp.sum(jnp.where(time_mask_y_i, log_likelihood_constant, 0.0))
    residuals_term = sum_sq_residuals / _sigma2
    loss_data = n_t_term + residuals_term
    
    #below does not work when n_t is different for each subject
    #apparently this litteraly lays out a loop like 'sum jnp.log( . . . ) n_t times'
    #which means this line has a different length when n_t is not the same for all subs
    #neg2_ll_data = (n_t * jnp.log(2 * jnp.pi) 
    #                + n_t * jnp.log(_sigma2) 
    #                + sum_sq_residuals / _sigma2)

    # --- Prior Penalty Part ---
    # Calculate the penalty from the random effects distribution
    # b_i.T @ inv(omega2) @ b_i + log(det(omega2))
    L, low = jax.scipy.linalg.cho_factor(omega2, lower=True)
    log_det_omega2 = 2 * jnp.sum(jnp.log(jnp.diag(L)))
    
    # Use Cholesky solve for stability and efficiency
    prior_penalty = b_i @ jax.scipy.linalg.cho_solve((L, True), b_i)
    if use_surrogate_neg2ll:
        loss_prior = log_det_omega2 + prior_penalty
    else:
        neg2ll_prior = (n_random_effects * jnp.log(2 * jnp.pi) 
                        + log_det_omega2 
                        + prior_penalty)
        loss_prior = neg2ll_prior
    #jax.debug.print("neg2ll_prior shape: {s}", s = neg2ll_prior.shape)
    #jax.debug.print("neg2_ll_data shape: {s}", s = neg2_ll_data.shape)
    loss =  loss_data + loss_prior
    
    large_penalty = 1e12
    loss_out = jnp.where(is_bad_state, large_penalty, loss)
    #jax.debug.print("""
    #                -----
    #                b_i: {b_i}
    #                -----
    #                b_i_work: {b_i_work}
    #                -----
    #                model_coeffs_i: {model_coeffs_i}
    #                -----
    #                solver_coeffs: {solver_coeffs}
    #                -----
    #                padded_pred_y_i: {padded_pred_y_i}
    #                -----
    #                masked_residuals_i: {masked_residuals_i}
    #                -----
    #                sum_sq_residuals: {sum_sq_residuals}
    #                -----
    #                log_likelihood_constant: {log_likelihood_constant}
    #                -----
    #                neg2_ll_data: {neg2_ll_data}
    #                -----
    #                loss: {loss}
    #                -----
    #                loss_out: {loss_out}
    #                
    #                """,
    #                b_i = b_i.shape,
    #                b_i_work = b_i_work.shape, 
    #                model_coeffs_i = model_coeffs_i.shape, 
    #                solver_coeffs = solver_coeffs.shape, 
    #                padded_pred_y_i = padded_pred_y_i.shape, 
    #                masked_residuals_i = masked_residuals_i.shape, 
    #                sum_sq_residuals = sum_sq_residuals.shape, 
    #                log_likelihood_constant = log_likelihood_constant.shape, 
    #                neg2_ll_data = neg2_ll_data.shape, 
    #                loss = loss,
    #                loss_out = loss_out
    #                
    #                
    #                )
    #jax.debug.print("model_coeffs_i coeffs calc AND state: EXP( {s} + {x} ) = {y}, State: {z}, Loss: {l}",
    #                s = data_contribution_i,
    #                x = combined_coeffs,
    #                y = model_coeffs_i,
    #                z = is_bad_state, 
    #                l = loss_out
    #                )

    #jax.debug.print("Inner Loss Out val: {s}", s = loss_out)
    return loss_out

def foc_interaction_term(all_hessians: jnp.ndarray, jitter: float = 1e-6) -> jnp.ndarray:
    """
    Calculates the FOCEi interaction term in a vectorized manner.

    Args:
        all_hessians: A batched array of Hessian matrices, with shape
                      (num_subjects, n_params, n_params).
        jitter: A small value added to the diagonal for numerical stability.

    Returns:
        A scalar JAX array containing the interaction term.
    """
    # 1. Add a small "jitter" to the diagonal of each Hessian.
    # This improves numerical stability and helps ensure the matrices are
    # positive-definite, preventing the Cholesky decomposition from failing.
    num_params = all_hessians.shape[-1]
    jitter_matrix = jnp.eye(num_params) * jitter
    stable_hessians = all_hessians + jitter_matrix

    # 2. Perform Cholesky decomposition on the entire batch of Hessians at once.
    # `cholesky` will operate on the last two dimensions.
    try:
        L = jnp.linalg.cholesky(stable_hessians)
    except Exception as e:
        # This will only be caught in eager mode, but it's good for debugging.
        print("Cholesky decomposition failed. Hessians may not be positive-definite.")
        raise e

    # 3. Extract the diagonal elements from each matrix in the batch.
    # The result `diags` will have a shape of (num_subjects, n_params).
    diags = jnp.diagonal(L, axis1=-2, axis2=-1)
    
    # 4. Calculate the log-determinant for each subject.
    # This sums the log-diagonals over the last axis. The result is a
    # vector of shape (num_subjects,).
    # We use nan_to_num to handle cases where a diagonal is zero or negative.
    log_determinants = 2 * jnp.sum(jnp.nan_to_num(jnp.log(diags), nan=0.0, neginf=-1e6), axis=-1)

    # 5. Sum the log-determinants across all subjects for the final term.
    interaction_term = jnp.sum(log_determinants)
    
    return interaction_term

def foc_interaction_term_chol(all_hessians: jnp.ndarray, jitter: float = 1e-6) -> jnp.ndarray:
    """
    Calculates the FOCEi interaction term using a stable Cholesky decomposition.

    Args:
        all_hessians: A batched array of Hessian matrices, with shape
                      (num_subjects, n_params, n_params).
        jitter: A small value added to the diagonal for numerical stability.

    Returns:
        A scalar JAX array containing the interaction term.
    """
    # 1. Stabilize: Add jitter to the diagonal of each Hessian to ensure
    #    they are all strictly positive-definite.
    num_params = all_hessians.shape[-1]
    jitter_matrix = jnp.eye(num_params) * jitter
    stable_hessians = all_hessians + jitter_matrix

    # 2. Calculate: Perform Cholesky decomposition on the stabilized matrices.
    #    This is now guaranteed to be safe.
    L = jnp.linalg.cholesky(stable_hessians)

    # 3. Extract the diagonal elements from each Cholesky factor `L`.
    diags = jnp.diagonal(L, axis1=-2, axis2=-1)
    
    # 4. Compute the log-determinant from the diagonals.
    #    The log-determinant of H is 2 * sum(log(diag(L))).
    #    We use nan_to_num as a final safeguard against log(0) if jitter is
    #    insufficient, preventing NaNs from poisoning the entire loss.
    interaction_term_i = 2 * jnp.sum(jnp.nan_to_num(jnp.log(diags), nan=0.0, neginf=0.0), axis=-1)

    # 5. Sum across all subjects for the final interaction term.
    interaction_term = jnp.sum(interaction_term_i)

    return interaction_term, interaction_term_i

def foc_interaction_term_passthrough(all_hessians, **kwargs):
    interaction_term_placeholder = 0.0
    interaction_term_i_placeholder = 0.0
    return interaction_term_placeholder, interaction_term_i_placeholder

def estimate_b_i_foce_passthrough(initial_b_i_batch, **kwargs):
    hessian_placeholder = 0.0
    inner_loss_placeholder = 0.0
    return initial_b_i_batch, hessian_placeholder, inner_loss_placeholder

def estimate_b_i_fo_passthrough(b_i, **kwargs):
    return b_i

def sum_neg2ll_terms_passthrough(neg2ll, **kwargs):
    grad_objective = neg2ll
    loss_objective = neg2ll
    return grad_objective, loss_objective

def focei_sum_neg2ll_terms(neg2ll, interaction_term, **kwargs):
    grad_objective = neg2ll + interaction_term
    loss_objective = neg2ll
    return grad_objective, loss_objective

class FOCEi_approx_neg2ll_loss_jax():
    """FOCEi loss for use as the `jax_loss` init argument for `CompartmentalModel`. 
    This version DOES NOT support `jax.grad` or finite differences via `finitediffx.fgrad`
    
    """
    def __init__(self):
        warnings.warn(f"""The `jax_loss` {self.__name__} only supports gradient estimation with Scipy's finite difference.
                      `FOCEi_approx_neg2ll_loss_jax_fdxOUTER` supports gradient estimation with finitediffx. There is 
                      currently no implementation of FOCEi supporting jax.grad and thus this objective is quite slow. 
                      You may be better off simply fitting a fully bayesian pymc model. The time to fit with pymc
                      will be about the same as this objective, and the richness of the pymc result is much greater.
                      """)
        self.loss_val_idx = 0
        self.grad_val_idx = 1
        self.grad_is_fdx = True
    
    @staticmethod
    def loss_fn(
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
    compiled_augdyn_ivp_solver_novmap_arr,
    compiled_ivp_solver_arr,
    compiled_ivp_solver_novmap_arr,
    ode_t0_vals,
    pop_coeff_for_J_idx,
    use_surrogate_neg2ll,
    inner_optimizer_tol = None, 
    inner_optimizer_maxiter = None,
    **kwargs,
    ):
        print("Compiling `FOCEi_approx_neg2ll_loss_jax`")
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
            compiled_augdyn_ivp_solver_novmap_arr = compiled_augdyn_ivp_solver_novmap_arr,
            compiled_ivp_solver_arr = compiled_ivp_solver_arr,
            compiled_ivp_solver_novmap_arr = compiled_ivp_solver_novmap_arr,
            ode_t0_vals = ode_t0_vals,
            pop_coeff_for_J_idx = pop_coeff_for_J_idx,
            compiled_estimate_b_i_foce = estimate_b_i_vmapped,
            compiled_estimate_b_i_fo = estimate_b_i_fo_passthrough, 
            jittable_estimate_foc_i=foc_interaction_term_chol, 
            jittable_sum_neg2ll_terms=focei_sum_neg2ll_terms,
            use_surrogate_neg2ll = use_surrogate_neg2ll,
            inner_optimizer_tol = inner_optimizer_tol, 
            inner_optimizer_maxiter = inner_optimizer_maxiter,
            
        )
        
        return loss
    @staticmethod
    def grad_method():
        return None, None

class FOCE_approx_neg2ll_loss_jax_iftINNER_ALT():
    """FOCE loss for debugging ONLY. You should not use this loss. 
    
    """
    def __init__(self, bypass_notimplemented = False):

        if bypass_notimplemented:
            self.loss_val_idx = 0
            self.grad_val_idx = 0
            self.grad_is_fdx = False
        else:
            raise NotImplementedError(f"{self.__name__} is for debugging only. Use `FOCE_approx_neg2ll_loss_jax_fdxOUTER` or `FOCE_approx_neg2ll_loss_jax`")
    
    @staticmethod
    def loss_fn(
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
    compiled_augdyn_ivp_solver_novmap_arr,
    compiled_ivp_solver_arr,
    compiled_ivp_solver_novmap_arr,
    ode_t0_vals,
    pop_coeff_for_J_idx,
    use_surrogate_neg2ll,
    inner_optimizer_tol = None, 
    inner_optimizer_maxiter = None,
    **kwargs,
    ):
        print("Compiling `FOCE_approx_neg2ll_loss_jax`")
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
            compiled_augdyn_ivp_solver_novmap_arr = compiled_augdyn_ivp_solver_novmap_arr,
            compiled_ivp_solver_arr = compiled_ivp_solver_arr,
            compiled_ivp_solver_novmap_arr = compiled_ivp_solver_novmap_arr,
            ode_t0_vals = ode_t0_vals,
            pop_coeff_for_J_idx = pop_coeff_for_J_idx,
            compiled_estimate_b_i_foce = estimate_b_i_vmapped_ift_ALT,
            compiled_estimate_b_i_fo = estimate_b_i_fo_passthrough, 
            jittable_estimate_foc_i=foc_interaction_term_passthrough, 
            jittable_sum_neg2ll_terms=sum_neg2ll_terms_passthrough,
            use_surrogate_neg2ll = use_surrogate_neg2ll,
            inner_optimizer_tol = inner_optimizer_tol, 
            inner_optimizer_maxiter = inner_optimizer_maxiter,
            
        )
        
        return loss
    @staticmethod
    def grad_method():
        return jax.grad, jax.value_and_grad

class FOCE_approx_neg2ll_loss_jax_iftINNER():
    """FOCE loss for debugging ONLY. You should not use this loss. 
    
    """
    def __init__(self, bypass_notimplemented = False):
        if bypass_notimplemented:
            self.loss_val_idx = 0
            self.grad_val_idx = 0
            self.grad_is_fdx = False
        else:
            raise NotImplementedError(f"{self.__name__} is for debugging only. Use `FOCE_approx_neg2ll_loss_jax_fdxOUTER` or `FOCE_approx_neg2ll_loss_jax`")
    @staticmethod
    def loss_fn(
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
    compiled_augdyn_ivp_solver_novmap_arr,
    compiled_ivp_solver_arr,
    compiled_ivp_solver_novmap_arr,
    ode_t0_vals,
    pop_coeff_for_J_idx,
    use_surrogate_neg2ll,
    inner_optimizer_tol = None, 
    inner_optimizer_maxiter = None,
    **kwargs,
    ):
        print("Compiling `FOCE_approx_neg2ll_loss_jax`")
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
            compiled_augdyn_ivp_solver_novmap_arr = compiled_augdyn_ivp_solver_novmap_arr,
            compiled_ivp_solver_arr = compiled_ivp_solver_arr,
            compiled_ivp_solver_novmap_arr = compiled_ivp_solver_novmap_arr,
            ode_t0_vals = ode_t0_vals,
            pop_coeff_for_J_idx = pop_coeff_for_J_idx,
            compiled_estimate_b_i_foce = estimate_b_i_vmapped_ift,
            compiled_estimate_b_i_fo = estimate_b_i_fo_passthrough, 
            jittable_estimate_foc_i=foc_interaction_term_passthrough, 
            jittable_sum_neg2ll_terms=sum_neg2ll_terms_passthrough,
            use_surrogate_neg2ll = use_surrogate_neg2ll,
            inner_optimizer_tol = inner_optimizer_tol, 
            inner_optimizer_maxiter = inner_optimizer_maxiter,
            
        )
        
        return loss
    @staticmethod
    def grad_method():
        return jax.grad, jax.value_and_grad

class FOCE_approx_neg2ll_loss_jax_fdxINNER():
    """FOCE loss where finite differences is used to estimate the gradient of the inner optimizer.
    This allows `jax.grad` to be used to estimate the gradient of the outer loss, but this version is 
    still much slower than `FOCE_approx_neg2ll_loss_jax_fdxOUTER` and `FOCE_approx_neg2ll_loss_jax`.
    
    """
    def __init__(self, bypass_notimplemented = False):
        
        if bypass_notimplemented:
            self.loss_val_idx = 0
            self.grad_val_idx = 0
            self.grad_is_fdx = False
        else:
            raise NotImplementedError(f"`{self.__name__}` is for debugging or research. Use `FOCE_approx_neg2ll_loss_jax_fdxOUTER` or `FOCE_approx_neg2ll_loss_jax`" )
    
    @staticmethod
    def loss_fn(
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
    compiled_augdyn_ivp_solver_novmap_arr,
    compiled_ivp_solver_arr,
    compiled_ivp_solver_novmap_arr,
    ode_t0_vals,
    pop_coeff_for_J_idx,
    use_surrogate_neg2ll,
    inner_optimizer_tol = None, 
    inner_optimizer_maxiter = None,
    **kwargs,
    ):
        print("Compiling `FOCE_approx_neg2ll_loss_jax`")
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
            compiled_augdyn_ivp_solver_novmap_arr = compiled_augdyn_ivp_solver_novmap_arr,
            compiled_ivp_solver_arr = compiled_ivp_solver_arr,
            compiled_ivp_solver_novmap_arr = compiled_ivp_solver_novmap_arr,
            ode_t0_vals = ode_t0_vals,
            pop_coeff_for_J_idx = pop_coeff_for_J_idx,
            compiled_estimate_b_i_foce = estimate_b_i_vmapped_fdx,
            compiled_estimate_b_i_fo = estimate_b_i_fo_passthrough, 
            jittable_estimate_foc_i=foc_interaction_term_passthrough, 
            jittable_sum_neg2ll_terms=sum_neg2ll_terms_passthrough,
            use_surrogate_neg2ll = use_surrogate_neg2ll,
            inner_optimizer_tol = inner_optimizer_tol, 
            inner_optimizer_maxiter = inner_optimizer_maxiter,
            
        )
        
        return loss
    @staticmethod
    def grad_method():
        return jax.grad, jax.value_and_grad

class FOCEi_approx_neg2ll_loss_jax_fdxOUTER():
    """FOCEi loss where finite differences (`finitediffx`) is used to estimate the gradient of the outer optimizer.
    FOCE loss is used to determine if an optmize step provides an improvement. FOCEi loss is used to estimate the gradient
    of the outer optimizer. Using FOCEi loss for both tends to result in the a loss vs iteration trajectory which 
    decreases rapidly for the first several iterations but later the loss bounces up and down each iteration leading to the 
    optmizer never finishing.  
    
    """
    def __init__(self):
        self.loss_val_idx = 1
        self.grad_val_idx = 0
        self.grad_is_fdx = True
    
    @staticmethod
    def loss_fn(
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
    compiled_augdyn_ivp_solver_novmap_arr,
    compiled_ivp_solver_arr,
    compiled_ivp_solver_novmap_arr,
    ode_t0_vals,
    pop_coeff_for_J_idx,
    use_surrogate_neg2ll,
    inner_optimizer_tol = None, 
    inner_optimizer_maxiter = None,
    **kwargs,
    ):
        print("Compiling `FOCEi_approx_neg2ll_loss_jax`")
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
            compiled_augdyn_ivp_solver_novmap_arr = compiled_augdyn_ivp_solver_novmap_arr,
            compiled_ivp_solver_arr = compiled_ivp_solver_arr,
            compiled_ivp_solver_novmap_arr = compiled_ivp_solver_novmap_arr,
            ode_t0_vals = ode_t0_vals,
            pop_coeff_for_J_idx = pop_coeff_for_J_idx,
            compiled_estimate_b_i_foce = estimate_b_i_vmapped,
            compiled_estimate_b_i_fo = estimate_b_i_fo_passthrough, 
            jittable_estimate_foc_i=foc_interaction_term_chol, 
            jittable_sum_neg2ll_terms=focei_sum_neg2ll_terms,
            use_surrogate_neg2ll = use_surrogate_neg2ll,
            inner_optimizer_tol = inner_optimizer_tol, 
            inner_optimizer_maxiter = inner_optimizer_maxiter,
            
        )
        
        return loss
    @staticmethod
    def grad_method():
        return partial(fdx.fgrad, offsets = fdx.Offset(accuracy=3)), partial(fdx.value_and_fgrad, offsets = fdx.Offset(accuracy=3))


class FOCE_approx_neg2ll_loss_jax_fdxOUTER():
    """FOCE loss where finite differences (`finitediffx`) is used to estimate the gradient of the outer optimizer.
    
    
    """
    def __init__(self):
        self.loss_val_idx = 0
        self.grad_val_idx = 0
        self.grad_is_fdx = True
    
    @staticmethod
    def loss_fn(
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
    compiled_augdyn_ivp_solver_novmap_arr,
    compiled_ivp_solver_arr,
    compiled_ivp_solver_novmap_arr,
    ode_t0_vals,
    pop_coeff_for_J_idx,
    use_surrogate_neg2ll,
    inner_optimizer_tol = None, 
    inner_optimizer_maxiter = None,
    **kwargs,
    ):
        print("Compiling `FOCE_approx_neg2ll_loss_jax`")
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
            compiled_augdyn_ivp_solver_novmap_arr = compiled_augdyn_ivp_solver_novmap_arr,
            compiled_ivp_solver_arr = compiled_ivp_solver_arr,
            compiled_ivp_solver_novmap_arr = compiled_ivp_solver_novmap_arr,
            ode_t0_vals = ode_t0_vals,
            pop_coeff_for_J_idx = pop_coeff_for_J_idx,
            compiled_estimate_b_i_foce = estimate_b_i_vmapped,
            compiled_estimate_b_i_fo = estimate_b_i_fo_passthrough, 
            jittable_estimate_foc_i=foc_interaction_term_passthrough, 
            jittable_sum_neg2ll_terms=sum_neg2ll_terms_passthrough,
            use_surrogate_neg2ll = use_surrogate_neg2ll,
            inner_optimizer_tol = inner_optimizer_tol, 
            inner_optimizer_maxiter = inner_optimizer_maxiter,
            
        )
        
        return loss
    @staticmethod
    def grad_method():
        return partial(fdx.fgrad, offsets = fdx.Offset(accuracy=3)), partial(fdx.value_and_fgrad, offsets = fdx.Offset(accuracy=3))

class FOCE_approx_neg2ll_loss_jax():
    """FOCE loss where Scipy's finite difference algorithm is used to estimate the gradient of the outer optimizer.
    
    
    """
    def __init__(self):
        self.loss_val_idx = 0
        self.grad_val_idx = 0
        self.grad_is_fdx = True
    
    @staticmethod
    def loss_fn(
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
    compiled_augdyn_ivp_solver_novmap_arr,
    compiled_ivp_solver_arr,
    compiled_ivp_solver_novmap_arr,
    ode_t0_vals,
    pop_coeff_for_J_idx,
    use_surrogate_neg2ll,
    inner_optimizer_tol = None, 
    inner_optimizer_maxiter = None,
    **kwargs,
    ):
        print("Compiling `FOCE_approx_neg2ll_loss_jax`")
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
            compiled_augdyn_ivp_solver_novmap_arr = compiled_augdyn_ivp_solver_novmap_arr,
            compiled_ivp_solver_arr = compiled_ivp_solver_arr,
            compiled_ivp_solver_novmap_arr = compiled_ivp_solver_novmap_arr,
            ode_t0_vals = ode_t0_vals,
            pop_coeff_for_J_idx = pop_coeff_for_J_idx,
            compiled_estimate_b_i_foce = estimate_b_i_vmapped,
            compiled_estimate_b_i_fo = estimate_b_i_fo_passthrough, 
            jittable_estimate_foc_i=foc_interaction_term_passthrough, 
            jittable_sum_neg2ll_terms=sum_neg2ll_terms_passthrough,
            use_surrogate_neg2ll = use_surrogate_neg2ll,
            inner_optimizer_tol = inner_optimizer_tol, 
            inner_optimizer_maxiter = inner_optimizer_maxiter,
            
        )
        
        return loss
    @staticmethod
    def grad_method():
        return None, None
    
class FO_approx_neg2ll_loss_jax():
    """FO loss where the gradient of the outer optmizer is estimated with `jax.grad`. 
    
    """
    def __init__(self):
        self.loss_val_idx = 0
        self.grad_val_idx = 0
        self.grad_is_fdx = False
    
    @staticmethod
    def loss_fn(
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
    compiled_augdyn_ivp_solver_novmap_arr,
    compiled_ivp_solver_arr,
    compiled_ivp_solver_novmap_arr,
    ode_t0_vals,
    pop_coeff_for_J_idx,
    use_surrogate_neg2ll,
    inner_optimizer_tol = None, 
    inner_optimizer_maxiter = None,
    **kwargs,
    ):
        print("Compiling `FO_approx_neg2ll_loss_jax`")
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
            compiled_augdyn_ivp_solver_novmap_arr = compiled_augdyn_ivp_solver_novmap_arr,
            compiled_ivp_solver_arr = compiled_ivp_solver_arr,
            compiled_ivp_solver_novmap_arr = compiled_ivp_solver_novmap_arr,
            ode_t0_vals = ode_t0_vals,
            pop_coeff_for_J_idx = pop_coeff_for_J_idx,
            compiled_estimate_b_i_foce = estimate_b_i_foce_passthrough,
            compiled_estimate_b_i_fo = estimate_ebes_jax, 
            jittable_estimate_foc_i=foc_interaction_term_passthrough, 
            jittable_sum_neg2ll_terms=sum_neg2ll_terms_passthrough,
            use_surrogate_neg2ll = use_surrogate_neg2ll,
            inner_optimizer_tol = inner_optimizer_tol, 
            inner_optimizer_maxiter = inner_optimizer_maxiter,
        )
    
        return loss
    
    @staticmethod
    def grad_method():
        return jax.grad, jax.value_and_grad

def approx_neg2ll_loss_jax(
    pop_coeff:jnp.ndarray, 
    sigma2:jnp.ndarray, 
    omega2:jnp.ndarray, 
    theta:jnp.ndarray, 
    theta_data:jnp.ndarray,
    padded_y:jnp.ndarray,
    unpadded_y_len,
    time_mask_y,
    time_mask_J,
    compiled_augdyn_ivp_solver_arr,
    compiled_augdyn_ivp_solver_novmap_arr,
    compiled_ivp_solver_arr,
    compiled_ivp_solver_novmap_arr,
    ode_t0_vals, 
    pop_coeff_for_J_idx,
    compiled_estimate_b_i_foce, 
    compiled_estimate_b_i_fo,
    jittable_estimate_foc_i,
    jittable_sum_neg2ll_terms, 
    use_surrogate_neg2ll,
    inner_optimizer_tol, 
    inner_optimizer_maxiter,
):
    """Constructor function for FO, FOCE and FOCEi. Implements FOCEi with various parameterizations 
    defined by the Jax loss classes turn the FOCEi into FO or FOCE by substituting functions utilized 
    in  `approx_neg2ll_loss_jax` with passthroughs. For example, for FO, `compiled_estimate_b_i_foce`
    is a passthrough which returns zeros in the shape of the `b_i`, that way the taylor expansion is done 
    about zero and thus FO is implemented. For FOCE and FOCEi the same function (`compiled_estimate_b_i_foce`)
    estimates the b_i using an inner optmizer and then the taylor exapansion is done about the current 
    best b_i rather than zero, thus implementing FOCE or FOCEi. 

    Args:
        pop_coeff (jnp.ndarray): A vector of population coeffcients corresponding to the arguements to the 
        in use ODE class. 
        sigma2 (jnp.ndarray): A vector with one entry representing the additive (constant) model error
        omega2 (jnp.ndarray): A variance covariance matrix representing the variance of the subject-level 
        effects (diag) and their covariance (off-diag)  
        theta (jnp.ndarray): A vector representing the 'fixed' effects of per ODE coeff independant variables.
        theta_data (jnp.ndarray): The data which the `theta` are multiplied by to produce per ODE coeff fixed linear effects  
        padded_y (:jnp.ndarray): An array of shape (n_subjects, n_global_timepoints). A processed version of the 
        dependant variable where each subject (row) as has same number of timepoints (cols). Per-subject 'missing'
        timepoints are padded with zero.
        unpadded_y_len (_type_): The total length of y without padding
        time_mask_y (_type_): A boolean mask with shape (n_subjects, n_global_timepoints) used to mask the predictions 
        from the ivp solver such that per subject 'missing' timepoints are zero and thus match the `padded_y`. 
        time_mask_J (_type_): A boolean mask with shape (n_subjects, n_global_timepoints, n_subjectlevel_effects) used to mask the jacobian 
        representing d(pred)/d(b_i)
        compiled_augdyn_ivp_solver_arr (_type_): A pre-baked `diffrax.diffeqsolve` for vmapping over a matrix of 
        ODE coeffs (n_subjects, n_ode_coeffs) and ODE t0 values (n_subjects, n_ode_outputs). Used in the OUTER loss
        function to make per iteration predictions given the current best parameters. Simulataneously estimates d(pred)/d(b_i)
        for use in the taylor expansion for approxmating neg2ll.  
        compiled_augdyn_ivp_solver_novmap_arr (_type_):  pre-baked `diffrax.diffeqsolve` for solving ONE ivp with for one subject's 
        ODE coeffs (n_ode_coeffs,) and ODE t0 values (n_ode_outputs). Used in the INNER loss
        function to make ONE prediction following sucessful inner optimiztion. Simulataneously estimates d(pred)/d(b_i)
        used to approxiate d2(loss)/d2(b_i) for use in the FOCEi interaction term.
        compiled_ivp_solver_arr (_type_): A pre-baked `diffrax.diffeqsolve` for vmapping over a matrix of 
        ODE coeffs (n_subjects, n_ode_coeffs) and ODE t0 values (n_subjects, n_ode_outputs). DOES NOT estimate d(pred)/d(b_i)
        for use in the taylor expansion for approxmating neg2ll. Currently NOT USED.   
        compiled_ivp_solver_novmap_arr (_type_): pre-baked `diffrax.diffeqsolve` for solving ONE ivp with for one subject's 
        ODE coeffs (n_ode_coeffs,) and ODE t0 values (n_ode_outputs). Used in the INNER loss
        function each inner loss iteration to make predictions given the current best estimate of a subject's b_i.
        DOES NOT estimate d(pred)/d(b_i) used to approxiate d2(loss)/d2(b_i) for use in the FOCEi interaction term.
        ode_t0_vals (_type_): A matrix (n_subjects, n_ode_outputs) containing the t0 estimates for each ODE output. 
        pop_coeff_for_J_idx (_type_): A vector (n_subject_level_effects) denoting which of the population coeff have been
        parameterized with subject-level effects. For example, say there are three ODE coeffs and the ones with index 0 and 2 have 
        subject level effects, this vector would be [0,2].  
        compiled_estimate_b_i_foce (_type_): A jax jittable function for estimating the subject level effects  (b_i, aka eta) for 
        FOCE and FOCEi. When FO is used, thus function always returns a matrix of zeros with shape (n_subjects, n_subjectlevel_effects). 
        compiled_estimate_b_i_fo (_type_): A jax jittable function for post-hoc estimation of b_i following an FO fit using the
        empirical bayes estimate. When FOCE or FOCEi are used, this function returns the b_i estimated by `compiled_estimate_b_i_foce`
        jittable_estimate_foc_i (_type_): A jax jittable function for estimating the FOCEi interaction term from the hessian of 
        `compiled_estimate_b_i_foce`. When FOCE or FO is used, this function is a passthrough.
        jittable_sum_neg2ll_terms (_type_): For FOCE and FO, returns a tuple (outer_loss, outer_loss). For FOCEi, returns a tuple
        ()
        use_surrogate_neg2ll (_type_): If surrogate neg2ll should be the optimizaiton objective. The surrogate aligns with the nlmixr2 OBJF
        inner_optimizer_tol (_type_): `tol` argument for the inner FOCE optimizer. Currently a PLACEHOLDER which does not impact inner 
        optimizer functioning.   
        inner_optimizer_maxiter (_type_): `maxiters` for the inner FOCE optimizer. Currently a PLACEHOLDER which does not impact inner 
        optimizer functioning.   
    """
    print("Compiling `approx_neg2ll_loss_jax`")
    #jax.debug.print("theta shape: {s}", s = theta.shape )
    n_subjects = time_mask_y.shape[0]
    n_coeffs = pop_coeff.shape[0]
    n_subject_level_eff = pop_coeff_for_J_idx.shape[0]
    unpadded_y_len_batch = jnp.sum(time_mask_y, axis = 1)
    if theta.shape[0] == 0:
        
        data_contribution = jnp.zeros((n_subjects, n_coeffs))
    else:
        data_contribution = theta @ theta_data
    #jax.debug.print("time_mask_y shape: {s}", s = time_mask_y.shape )
    #jax.debug.print("data_contribution shape: {j}", j=data_contribution.shape)
    #jax.debug.print("pop_coeff shape: {j}", j=pop_coeff.shape)
    
    #estimate b_i here
    b_i = jnp.zeros((n_subjects, n_subject_level_eff))
    #jax.debug.print("initial_b_i_batch shape: {s}", s = b_i.shape )
    #jax.debug.print("padded_y shape: {s}", s = padded_y.shape )
    #jax.debug.print("data_contribution shape: {s}", s = data_contribution.shape )
    #jax.debug.print("ode_t0_vals shape: {s}", s = ode_t0_vals.shape )
    #jax.debug.print("time_mask_y shape: {s}", s = time_mask_y.shape )
    #jax.debug.print("unpadded_y_len_batch shape: {s}", s = unpadded_y_len_batch.shape )
    #jax.debug.print("pop_coeff shape: {s}", s = pop_coeff.shape )
    #jax.debug.print("sigma2 shape: {s}", s = sigma2.shape )
    #jax.debug.print("omega2 shape: {s}", s = omega2.shape )
    #jax.debug.print("n_subject_level_eff shape: {s}", s = n_subject_level_eff.shape )
    #jax.debug.print("pop_coeff_for_J_idx shape: {s}", s = pop_coeff_for_J_idx.shape )
    
#     jax.debug.print(
#     """
#     --- Calling estimate_b_i_vmapped_fdx ---
#     initial_b_i_batch type: {b_i_type}
#     padded_y_batch type: {padded_y_type}
#     data_contribution_batch type: {data_contrib_type}
#     ode_t0_vals_batch type: {ode_t0_type}
#     time_mask_y_batch type: {time_mask_type}
#     pop_coeff type: {pop_coeff_type}
#     sigma2 type: {sigma2_type}
#     n_random_effects type: {n_rand_type}
#     compiled_ivp_solver type: {ivp_solver_type}
#     compiled_augdyn_ivp_solver type: {augdyn_solver_type}
#     pop_coeff_w_bi_idx type: {pc_idx_type}
#     use_surrogate_neg2ll type: {use_surrogate_type}
#     """,
#     b_i_type=type(b_i),
#     padded_y_type=type(padded_y),
#     data_contrib_type=type(data_contribution),
#     ode_t0_type=type(ode_t0_vals),
#     time_mask_type=type(time_mask_y),
#     pop_coeff_type=type(pop_coeff),
#     sigma2_type=type(sigma2),
#     n_rand_type=type(n_subject_level_eff),
#     ivp_solver_type=type(compiled_ivp_solver_novmap_arr),
#     augdyn_solver_type=type(compiled_augdyn_ivp_solver_novmap_arr),
#     pc_idx_type=type(pop_coeff_for_J_idx),
#     use_surrogate_type=type(use_surrogate_neg2ll)
# )
    
    b_i, hessian_i, inner_loss_i = compiled_estimate_b_i_foce(
                                               initial_b_i_batch = b_i,
                                                padded_y_batch = padded_y,
                                                data_contribution_batch = data_contribution,
                                                ode_t0_vals_batch = ode_t0_vals,
                                                time_mask_y_batch = time_mask_y,
                                                #unpadded_y_len_batch = unpadded_y_len_batch,
                                                pop_coeff = pop_coeff,
                                                sigma2 = sigma2,
                                                omega2 = omega2,
                                                n_random_effects = n_subject_level_eff,
                                                compiled_ivp_solver = compiled_ivp_solver_novmap_arr,
                                                compiled_augdyn_ivp_solver = compiled_augdyn_ivp_solver_novmap_arr,
                                                pop_coeff_w_bi_idx = pop_coeff_for_J_idx,
                                                use_surrogate_neg2ll = use_surrogate_neg2ll,
                                                inner_optimizer_tol = inner_optimizer_tol,
                                                inner_optimizer_maxiter = inner_optimizer_maxiter,
                                               
                                               )
    #jax.debug.print("b_i shape: {s}", s = b_i.shape )
    b_i_work = jnp.zeros((n_subjects, n_coeffs))
    #jax.debug.print("b_i_work shape: {s}", s = b_i_work.shape )
    b_i_work = b_i_work.at[:,pop_coeff_for_J_idx].set(b_i)
    #jax.debug.print("Post set b_i_work shape: {s}", s = b_i_work.shape )
    #jax.debug.print("Post set b_i_work val: {s}", s = b_i_work )
    subject_coeff = b_i_work + pop_coeff
    #jax.debug.print("subject_coeff shape: {s}", s = subject_coeff.shape )
    #jax.debug.print("subject_coeff val: {s}", s = subject_coeff )
    model_coeffs_i = jnp.exp(data_contribution + subject_coeff)# + 1e-6
    is_apprx_zero = (model_coeffs_i < 1e-9)
    not_finite = ~jnp.isfinite(model_coeffs_i)
    is_bad_state = jnp.any(not_finite | is_apprx_zero)
    safe_coeffs = jnp.ones_like(model_coeffs_i)
    solver_coeffs = jnp.where(is_bad_state, safe_coeffs, model_coeffs_i)
    #jax.debug.print("model_coeffs_i shape: {j}", j=model_coeffs_i.shape)
    # jax.debug.print("""
    #                 solver_coeffs: {x}
                    
    #                 """, x = solver_coeffs)
    padded_full_preds, padded_pred_y, J_full, J_conc_full = compiled_augdyn_ivp_solver_arr(
        ode_t0_vals,
        solver_coeffs
    )
    #jax.debug.print("padded_pred_y shape: {padded_pred_y}", padded_pred_y=padded_pred_y.shape)
    #jax.debug.print("J_conc_full shape: {J_conc_full}", J_conc_full=J_conc_full.shape)
    #jax.debug.print("model_coeffs_i shape: {j}", j=model_coeffs_i.shape)
    
    masked_residuals = jnp.where(time_mask_y, padded_y - padded_pred_y, 0.0)

    J_dense = J_conc_full#shape (n_subject, max_obs_per_subject, n_s)
    J = jnp.where(time_mask_J, J_dense, 0.0) # time_mask_J shape  (n_subject, max_obs_per_subject, n_s)
    
    interaction_term, interaction_term_i = jittable_estimate_foc_i(all_hessians = hessian_i)

    # Estimate the covariance matrix, then estimate neg log likelihood
    outer_loss, per_subject_outer_loss = outer_neg2ll_chol_jit(
        J,           # Shape: (n_subjects, max_obs, n_effects)
        masked_residuals,  # Shape: (n_subjects, max_obs)
        time_mask_y,              # Shape: (n_subjects, max_obs)
        sigma2,            # Shape: scalar
        omega2,           # Shape: (n_effects, n_effects)
        unpadded_y_len,
        use_surrogate_neg2ll=use_surrogate_neg2ll,
    )
    outer_objective_for_grad, outer_objective_for_loss = jittable_sum_neg2ll_terms(neg2ll = outer_loss, interaction_term = interaction_term)
    #jax.debug.print("neg2_ll: {neg2_ll}", neg2_ll=neg2_ll)
    #jax.debug.print("is_bad_state: {is_bad_state}", is_bad_state=is_bad_state)
    b_i_approx = compiled_estimate_b_i_fo(padded_J=J, padded_residuals=masked_residuals, 
                                   time_mask=time_mask_y,
                                   omegas2=omega2, sigma2=sigma2, b_i = b_i
                                   )
    large_penalty = 1e12
    outer_objective_for_grad_out = jnp.where(is_bad_state, large_penalty, outer_objective_for_grad)
    outer_objective_for_loss_out = jnp.where(is_bad_state, large_penalty, outer_objective_for_loss)
    per_subject_loss = (per_subject_outer_loss, interaction_term_i, unpadded_y_len_batch)
    #jax.debug.print("Outer Loss Out val: {s}", s = neg2_ll_out)
    value_out = {
        'outer_objective_loss':outer_objective_for_loss_out,
        'b_i':b_i_approx, 
        'padded_pred_y':padded_pred_y,
        'padded_pred_full':padded_full_preds,
        'model_coeffs_i':model_coeffs_i, 
        'per_subject_loss':per_subject_loss
    }
    return outer_objective_for_grad_out, value_out

