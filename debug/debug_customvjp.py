#%%
import jax.numpy as jnp
import jax
import optax
import finitediffx as fdx
import flax
from functools import partial
import joblib as jb


def FOCE_inner_loss_fn_lax(
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
    log_coeffs = data_contribution_i + combined_coeffs
    
    #jax.debug.print("model_coeffs_i shape: {s}", s = model_coeffs_i.shape)
    #jax.debug.print("model_coeffs_i val: {s}", s = model_coeffs_i)
    #jax.debug.print("combined_coeffs calc: {s} + {x} = {y}", s = pop_coeff_i, x = b_i_work, y = combined_coeffs)
    is_bad_state = jnp.any(log_coeffs > 700) | jnp.any(log_coeffs < -20)
    #jax.lax.cond(is_bad_state, true_branch, false_branch )
    #jax.debug.print("No EXP Inner Bad State Status: {s}", s = is_bad_state)
    def good_path(operands):
        #solver_coeffs = jnp.where(is_bad_state, safe_coeffs, model_coeffs_i)
        model_coeffs_i = jnp.exp(log_coeffs)
        #jax.debug.print("POST EXP Inner Bad State Status: {s}", s = is_bad_state)
        solver_coeffs = model_coeffs_i
        # --- Data Likelihood Part ---
        # Solve the ODE for this individual with the current b_i guess
        #jax.debug.print("model_coeffs_i shape: {s}", s = model_coeffs_i.shape)
        #jax.debug.print("ode_t0_val_i shape: {s}", s = ode_t0_val_i.shape)
        
        #_solver_model_coeffs_i = jnp.reshape(solver_coeffs, (1, -1))
        #_ode_t0_val_i = jnp.reshape(ode_t0_val_i, (1, -1))
        #jax.debug.print("model_coeffs_i shape: {s}", s = model_coeffs_i.shape)
        #jax.debug.print("ode_t0_val_i shape: {s}", s = ode_t0_val_i.shape)
        #jax.debug.print("Solving Inner Optmizer IVP . . .")
        
        #commented ivp solve for debugging purposes ------------
        #padded_full_preds_i, padded_pred_y_i = compiled_ivp_solver( ode_t0_val_i, solver_coeffs,)
        
        A = jnp.eye(time_mask_y_i.shape[0], model_coeffs_i.shape[0]) 
        padded_pred_y_i = A @ model_coeffs_i
        
        #----------------
        
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
        return loss
    
    def bad_path(operands):
        (log_coeffs,
         ode_t0_val_i,
         time_mask_y_i,
         padded_y_i,
         sigma2,
         omega2,b_i  ) = operands
        #jax.debug.print("INNER OPT BAD PATH TRIGGERED: {s}", s = is_bad_state)
        large_penalty = 1e12
        loss = large_penalty + jnp.sum(b_i.flatten()**2)
        
        return loss
        
    operands = (log_coeffs, ode_t0_val_i, time_mask_y_i,
                padded_y_i, sigma2, omega2, b_i   )
    
    
    loss_out = jax.lax.cond(is_bad_state, bad_path, good_path, operands)
    
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


def DEBUG_OMEGA_estimate_b_i_vmapped(
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
        obj_fn = lambda b_i: FOCE_inner_loss_fn_lax(
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

        # Get model_coeffs at the solution to calculate the analytical Hessian
        b_i_work = jnp.zeros_like(pop_coeff).at[pop_coeff_w_bi_idx].set(estimated_b_i)
        model_coeffs_at_opt = jnp.exp(data_contrib_i + pop_coeff + b_i_work)

        # --- Calculate the Analytical Hessian for the Simplified Linear Model ---

        # A must be the same matrix as used in the inner loss
        A = jnp.eye(time_mask_y_i.shape[0], pop_coeff.shape[0])

        # This is the Jacobian d(pred)/d(log_coeffs) for the simplified model
        S_simple = A @ jnp.diag(model_coeffs_at_opt)

        # Slice to get the Jacobian with respect to b_i
        S_simple_b = S_simple[:, pop_coeff_w_bi_idx]

        # Use the Gauss-Newton formula (which is exact for a linear model)
        H_approx = 2 * (S_simple_b.T @ S_simple_b) / sigma2[0]

        # The prior part of the Hessian remains the same
        L, _ = jax.scipy.linalg.cho_factor(omega2, lower=True)
        identity = jnp.eye(omega2.shape[0], dtype=omega2.dtype)
        inv_omega2 = jax.scipy.linalg.cho_solve((L, True), identity)
        H_prior = 2 * inv_omega2

        # The final, correct Hessian for the simplified problem
        H_foce = H_approx + H_prior
        
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

def debug_omega_loss_only(pop_coeff:jnp.ndarray, 
    sigma2:jnp.ndarray, 
    omega2:jnp.ndarray, 
    #theta:jnp.ndarray, 
    #theta_data:jnp.ndarray,
    data_contribution,
    padded_y:jnp.ndarray,
    #unpadded_y_len,
    time_mask_y,
    #time_mask_J,
    #compiled_augdyn_ivp_solver_arr,
    compiled_augdyn_ivp_solver_novmap_arr,
    #compiled_2ndorder_augdyn_ivp_solver_arr,
    compiled_2ndorder_augdyn_ivp_solver_novmap_arr,
    #compiled_ivp_solver_arr,
    compiled_ivp_solver_novmap_arr,
    ode_t0_vals, 
    pop_coeff_for_J_idx,
    compiled_estimate_b_i_foce, 
    #compiled_estimate_b_i_fo,
    #jittable_estimate_foc_i,
    #jittable_sum_neg2ll_terms, 
    use_surrogate_neg2ll,
    inner_optimizer_tol, 
    inner_optimizer_maxiter,
    #raw_incoming_optimizer_vals,
    **kwargs,
    ):
    
    print("Compiling `approx_neg2ll_loss_jax`")
    #jax.debug.print("Raw Optmizer Values: {v}", v = raw_incoming_optimizer_vals)
    #jax.debug.print("theta shape: {s}", s = theta.shape )
    n_subjects = time_mask_y.shape[0]
    n_coeffs = pop_coeff.shape[0]
    n_subject_level_eff = pop_coeff_for_J_idx.shape[0]
    unpadded_y_len_batch = jnp.sum(time_mask_y, axis = 1)
    #if theta.shape[0] == 0:
    #    
    #    data_contribution = jnp.zeros((n_subjects, n_coeffs))
    #else:
    #    data_contribution = theta @ theta_data
    #jax.debug.print("time_mask_y shape: {s}", s = time_mask_y.shape )
    #jax.debug.print("data_contribution shape: {j}", j=data_contribution.shape)
    #jax.debug.print("pop_coeff shape: {j}", j=pop_coeff.shape)
    
    #estimate b_i here
    b_i = jnp.zeros((n_subjects, n_subject_level_eff))
    
    # 1. Run the inner optimization exactly as before to get the final b_i
    # This call should be identical to the one in your real loss function.
    b_i, hessian_i, inner_loss_i = compiled_estimate_b_i_foce(
        initial_b_i_batch=b_i,
        padded_y_batch=padded_y,
        data_contribution_batch=data_contribution,
        ode_t0_vals_batch=ode_t0_vals,
        time_mask_y_batch=time_mask_y,
        # unpadded_y_len_batch = unpadded_y_len_batch,
        pop_coeff=pop_coeff,
        sigma2=sigma2,
        omega2=omega2,
        n_random_effects=n_subject_level_eff,
        compiled_ivp_solver=compiled_ivp_solver_novmap_arr,
        compiled_augdyn_ivp_solver=compiled_augdyn_ivp_solver_novmap_arr,
        compiled_2ndorder_augdyn_ivp_solver=compiled_2ndorder_augdyn_ivp_solver_novmap_arr,
        pop_coeff_w_bi_idx=pop_coeff_for_J_idx,
        use_surrogate_neg2ll=use_surrogate_neg2ll,
        inner_optimizer_tol=inner_optimizer_tol,
        inner_optimizer_maxiter=inner_optimizer_maxiter,
    )

    # 2. Calculate the inverse of omega2
    L, _ = jax.scipy.linalg.cho_factor(omega2, lower=True)
    identity = jnp.eye(omega2.shape[0], dtype=omega2.dtype)
    inv_omega2 = jax.scipy.linalg.cho_solve((L, True), identity)

    # 3. The loss is ONLY the sum of prior penalties over all subjects
    # This is the simplified part. We ignore the data likelihood.
    # Note: We need to use vmap or a loop to handle the mat-vec product per subject
    prior_penalties = jax.vmap(lambda b: b @ inv_omega2 @ b)(b_i)
    total_loss = jnp.sum(prior_penalties)

    return total_loss

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
        omega_lchol = jnp.zeros((omega_diag_size, omega_diag_size), dtype = jnp.float64)
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
        "raw_incoming_optimizer_vals":opt_params
        
    } 
    
    jax.debug.print("""
                    pop_coeff:{pop_coeffs}, 
                    sigma2:{sigma2}, 
                    omega2:{omega2}, 
                    theta:{thetas_work}, 
                    raw_incoming_optimizer_vals:{opt_params}
                    """, pop_coeffs = pop_coeffs, sigma2 = sigma2, 
                    omega2 = omega2, thetas_work = thetas_work, opt_params = opt_params
                    )

    return dynamic_loss_kwargs

#%%

with open("debug_unpacker_kwargs.jb", 'rb') as f:
    unpacker_kwargs = jb.load(f)

print(str(unpacker_kwargs))

initial_b_i_batch =  jnp.array([[0., 0., 0.]]),
padded_y_batch= jnp.array([[ 0.74,  2.84,  6.57 ,10.5,   9.66,  8.58,  8.36,  7.47,  6.89,  5.94,  3.28]])
data_contribution_batch= jnp.array([[0., 0., 0.]])
opt_params =  jnp.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

unpack_partial = partial(
    _jittable_param_unpack, 
    **unpacker_kwargs
)
#%%

def debug_loss(opt_params):
    p = unpack_partial(opt_params = opt_params)
    
    print("\n--- Verifying Unpacker ---")
    print("opt_params changing:", opt_params)
    print("Resulting pop_coeff:", p['pop_coeff'])
    print("Resulting sigma2:", p['sigma2'])
    print("Resulting omega2:\n", p['omega2'])
    
    data_contribution_batch= jnp.array([[0., 0., 0.]])
    p['data_contribution'] = data_contribution_batch
    
    loss_out = debug_omega_loss_only(
    #pop_coeff=pop_coeff, 
    #sigma2=sigma2, 
    #omega2=omega2, 
    #data_contribution=data_contribution_batch, 
    padded_y=padded_y_batch,
    time_mask_y=jnp.array([jnp.repeat(True, padded_y_batch.shape[1])]),
    compiled_augdyn_ivp_solver_novmap_arr = None,
    compiled_2ndorder_augdyn_ivp_solver_novmap_arr = None,
    compiled_ivp_solver_novmap_arr = None,
    ode_t0_vals=None,
    pop_coeff_for_J_idx=jnp.array([0,1,2]), 
    compiled_estimate_b_i_foce=DEBUG_OMEGA_estimate_b_i_vmapped,
    use_surrogate_neg2ll=True, 
    inner_optimizer_tol=None,
    inner_optimizer_maxiter=None,
    **p
    )
    
    return loss_out
    


# %%

def prior_penalty_chol(b_i, omega2):
    """Calculates the prior penalty using stable Cholesky decomposition."""
    L, _ = jax.scipy.linalg.cho_factor(omega2, lower=True)
    x = jax.scipy.linalg.solve_triangular(L, b_i, lower=True)
    quadratic_term = x.T @ x
    log_det_term = 2 * jnp.sum(jnp.log(jnp.diag(L)))
    return quadratic_term + log_det_term

def DEBUG_OMEGA_estimate_b_i_vmapped_ift(
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
    compiled_2ndorder_augdyn_ivp_solver,
    pop_coeff_w_bi_idx,
    use_surrogate_neg2ll,
    #interaction_term_objective,
    **kwargs,  # Absorb unused kwargs
):
    """Estimates b_i for all subjects by vmapping the custom VJP function."""
    print('Compiling `estimate_b_i_vmapped` with custom VJP')

    # Vmap the single-subject optimization function with the custom VJP.
    # `in_axes` specifies which arguments to map over (0) vs. broadcast (None).
    
    jax.debug.print("""
                    initial_b_i_batch: {initial_b_i_batch},
                    padded_y_batch: {padded_y_batch},
                    data_contribution_batch: {data_contribution_batch},
                    ode_t0_vals_batch: {ode_t0_vals_batch},
                    time_mask_y_batch: {time_mask_y_batch},
                    pop_coeff: {pop_coeff},
                    sigma2: {sigma2},
                    omega2: {omega2},
                    n_random_effects: {n_random_effects},
                    compiled_ivp_solver: {compiled_ivp_solver},
                    compiled_2ndorder_augdyn_ivp_solver: {compiled_2ndorder_augdyn_ivp_solver},
                    pop_coeff_w_bi_idx: {pop_coeff_w_bi_idx},
                    use_surrogate_neg2ll: {use_surrogate_neg2ll},
                    
                    
                    
                    """,
                    initial_b_i_batch=initial_b_i_batch,
                    padded_y_batch=padded_y_batch,
                    data_contribution_batch=data_contribution_batch,
                    ode_t0_vals_batch=ode_t0_vals_batch,
                    time_mask_y_batch=time_mask_y_batch,
                    pop_coeff=pop_coeff,
                    sigma2=sigma2,
                    omega2=omega2,
                    n_random_effects=n_random_effects,
                    compiled_ivp_solver=compiled_ivp_solver,
                    compiled_2ndorder_augdyn_ivp_solver=compiled_2ndorder_augdyn_ivp_solver,
                    pop_coeff_w_bi_idx=pop_coeff_w_bi_idx,
                    use_surrogate_neg2ll=use_surrogate_neg2ll,
                    #interaction_term_objective=interaction_term_objective,
                    )
    
    
    #@jax.jit
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
        obj_fn = lambda b_i: FOCE_inner_loss_fn_lax(
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

                # 2. Set up an optax optimizer
        learning_rate = 0.08  # Tune as needed
        num_inner_steps = 200 # Tune as needed
        b_i_lower_bound = -8
        b_i_upper_bound = 8
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
                new_params_unconstrained = optax.apply_updates(params, updates)
                
                new_params = jnp.clip(new_params_unconstrained, 
                              b_i_lower_bound, 
                              b_i_upper_bound)
                
                
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
        jax.debug.print("""
                        --------
                        Subject Convergence Status: {hs}
                        
                        Loss History: {lh}
                        
                        Norm History: {nh}
                        """, 
                        hs = has_converged, lh = loss_history, nh = norm_history
                        )

        # --- Hessian Approximation ---
        def predict_fn(b_i_for_pred):
            #jax.debug.print("Post fit b_i: {bi}", )
            b_i_work = jnp.zeros_like(pop_coeff)
            b_i_work = b_i_work.at[pop_coeff_w_bi_idx].set(b_i_for_pred)
            combined_coeffs = pop_coeff + b_i_work
            model_coeffs_i = jnp.exp(data_contrib_i + combined_coeffs)
            is_bad_state = jnp.any(~jnp.isfinite(model_coeffs_i)) | jnp.any(model_coeffs_i < 1e-9)
            jax.debug.print("""Post Fit Inner Bad State Status: {s}
                            
                            Post fit b_i: {bi}
                            
                            model_coeffs_i: {mci}
                            
                            t0: {t0}
                            """, s = is_bad_state, bi = b_i_for_pred, mci = model_coeffs_i, t0 = ode_t0_i)
            #safe_coeffs = jnp.ones_like(model_coeffs_i)
            #solver_coeffs = jnp.where(is_bad_state, safe_coeffs, model_coeffs_i)
            #jax.debug.print("Solving Post fit 0th order IVP")
            _ = compiled_ivp_solver(ode_t0_i, model_coeffs_i)
            #jax.debug.print("Solving Post fit 2nd order IVP")
            _, pred_conc, _, S_conc_full, _, H_conc_full = compiled_2ndorder_augdyn_ivp_solver(ode_t0_i, model_coeffs_i)
            
            # We need the elements in J corresponding to the mask, implemented below
            return  pred_conc, S_conc_full, H_conc_full, model_coeffs_i
        
        final_inner_loss_value = obj_fn(estimated_b_i)
        jax.debug.print("estimated_b_i: {bi}", bi = estimated_b_i)
        #pred_conc, S, H, model_coeffs_at_opt = predict_fn(estimated_b_i)
        #
        #residuals_masked = jnp.where(time_mask_y_i, padded_y_i - pred_conc, 0.0)
        #
        #H_masked = H * time_mask_y_i[:, None, None]
        #
        #S_masked = S * time_mask_y_i[:, None]
        #
        #
        ##estimate the H_foce needed for calculating the interaction term in FOCEi
        #scaling_factors_b = model_coeffs_at_opt[pop_coeff_w_bi_idx]
        ##jax.debug.print("""
        ##                -----------------
        ##                S shape: {ss}
        ##                
        ##                pop_coeff_w_bi_idx shape: {ps}
        ##                
        ##                scaling_factors_b shape: {bs}
        ##                -----------------
        ##                """, ss = S.shape, ps = pop_coeff_w_bi_idx.shape, bs = scaling_factors_b.shape)
        #Z_i = S[:, pop_coeff_w_bi_idx] * scaling_factors_b[None, :]
        #Z_i_masked = Z_i * time_mask_y_i[:, None]
        #
        #_sigma2 = sigma2[0]
        #H_data_term1 = Z_i_masked.T @ Z_i_masked
        #H_data_term2 = jnp.einsum('tij,t->ij', H_masked[:, pop_coeff_w_bi_idx][:, :, pop_coeff_w_bi_idx], residuals_masked)
        #H_data = (H_data_term1 - H_data_term2) / _sigma2
        #     
        #L, _ = jax.scipy.linalg.cho_factor(omega2, lower=True)
        #identity = jnp.eye(omega2.shape[0], dtype=omega2.dtype)
        #inv_omega2 = jax.scipy.linalg.cho_solve((L, True), identity)
        #H_prior = inv_omega2
        #
        #H_foce = 2 * (H_data + H_prior)
        
        # --- TEMPORARY DEBUGGING HESSIAN ---
        # We need model_coeffs_at_opt to calculate the analytical Hessian
        b_i_work = jnp.zeros_like(pop_coeff).at[pop_coeff_w_bi_idx].set(estimated_b_i)
        model_coeffs_at_opt = jnp.exp(data_contrib_i + pop_coeff + b_i_work)
        scaling_factors_b = model_coeffs_at_opt[pop_coeff_w_bi_idx]
        # A should be the same matrix as used in your simplified inner loss.
        A = jnp.eye(time_mask_y_i.shape[0], pop_coeff.shape[0])
        S_simple = A @ jnp.diag(model_coeffs_at_opt)
        S_simple_b = S_simple[:, pop_coeff_w_bi_idx]

        H_data = 2 * (S_simple_b.T @ S_simple_b) / sigma2[0]

        L, _ = jax.scipy.linalg.cho_factor(omega2, lower=True)
        identity = jnp.eye(omega2.shape[0], dtype=omega2.dtype)
        inv_omega2 = jax.scipy.linalg.cho_solve((L, True), identity)
        H_prior = 2 * inv_omega2

        H_foce = H_data + H_prior # This is the new, analytically correct Hessian
        # --- END OF TEMPORARY CHANGE ---

        # NOTE: The residuals must also come from the simplified model for the bwd pass
        #       You will need to pass them through. For now, let's focus on getting
        #       H_foce correct, as it's the main suspect.
        # A placeholder is used for the unused S and H from the real model.
        #S_placeholder = jnp.zeros((time_mask_y_i.shape[0], pop_coeff.shape[0]))
        #H_placeholder = jnp.zeros((time_mask_y_i.shape[0], pop_coeff.shape[0], pop_coeff.shape[0]))
        simple_preds = A @ model_coeffs_at_opt
        simple_residuals = jnp.where(time_mask_y_i, padded_y_i - simple_preds, 0.0)
        H_simple = jnp.zeros((time_mask_y_i.shape[0], pop_coeff.shape[0], pop_coeff.shape[0]))
        
        jax.debug.print("H_foce: {hf}", hf = H_foce)
        return (estimated_b_i, H_foce, final_inner_loss_value, 
                S_simple, H_simple, simple_residuals, # Pass placeholders
                scaling_factors_b, model_coeffs_at_opt)
    
    def _estimate_single_b_i_fwd(*args):
        # args now only contains the 9 JAX array arguments
        outputs = estimate_single_b_i_impl(*args)
        estimated_b_i, H_foce, final_inner_loss, S_masked, H_masked, residuals_masked, scaling_factors_b, model_coeffs_at_opt = outputs
        fwd_output = estimated_b_i, H_foce, final_inner_loss
        residuals = (estimated_b_i, H_foce, S_masked, H_masked, residuals_masked, scaling_factors_b, model_coeffs_at_opt) + args
        return fwd_output, residuals
    
    def _estimate_single_b_i_bwd(residuals, g):
        """
        Computes the backward pass for the custom VJP.
        Rewritten from scratch to be clean and correct.
        """
        # 1. Unpack all residuals from the forward pass and the incoming cotangents
        (estimated_b_i, H_foce, S_masked, H_masked, residuals_masked, 
        scaling_factors_b, model_coeffs_at_opt, initial_b_i, padded_y_i, 
        data_contrib_i, ode_t0_i, time_mask_y_i, pop_coeff, sigma2, omega2, 
        pop_coeff_w_bi_idx) = residuals
        g_b_i, g_H_foce, g_loss = g

        # 2. Calculate the 'v' vector. We have confirmed this part is working correctly.
        v = jax.scipy.linalg.solve(H_foce, g_b_i, assume_a='pos')

        # --- Gradient w.r.t. Population Coefficients (pop_coeff) ---
        
        # 3a. Calculate sensitivities w.r.t. b_i and pop_coeff using the chain rule
        S_wrt_b = S_masked[:, pop_coeff_w_bi_idx] * scaling_factors_b[None, :]
        S_wrt_pc = S_masked * model_coeffs_at_opt[None, :]

        # 3b. Calculate the mixed Hessian d^2(y)/db_i/dpc_k for the linear toy model
        n_b = len(pop_coeff_w_bi_idx)
        n_pc = S_masked.shape[1]
        H_term = H_masked[:, pop_coeff_w_bi_idx, :] * scaling_factors_b[None, :, None] * model_coeffs_at_opt[None, None, :]
        selection_matrix = jnp.zeros((n_b, n_pc)).at[jnp.arange(n_b), pop_coeff_w_bi_idx].set(1)
        S_term = jnp.einsum('tk,jk->tjk', S_wrt_pc, selection_matrix)
        H_wrt_b_pc = H_term + S_term

        # 3c. Calculate the mixed-partial J_cross_pc with the corrected sign
        term1_final = jnp.einsum('tij,t->ij', H_wrt_b_pc, residuals_masked)
        term2_final = S_wrt_b.T @ S_wrt_pc
        J_cross_pc = (2 / sigma2[0]) * (term2_final - term1_final)
        
        # 3d. Calculate implicit and explicit gradients
        implicit_grad_pc = -v @ J_cross_pc
        
        # The explicit gradient MUST match the outer loss. For our debug tests,
        # the outer loss has no explicit dependency on pop_coeff, so this is 0.0.
        explicit_grad_pc = 0.0
        grad_pop_coeff = implicit_grad_pc + explicit_grad_pc
        grad_data_contrib = grad_pop_coeff
        
        # --- Gradient w.r.t. Sigma ---

        # 4a. Calculate J_cross_s2
        J_cross_s2 = 2 * (S_wrt_b.T @ residuals_masked) / sigma2[0] ** 2
        
        # 4b. Calculate implicit and explicit gradients
        implicit_grad_s2 = -v @ J_cross_s2

        # The explicit gradient MUST match the outer loss. For our debug tests,
        # the outer loss has no explicit dependency on sigma2, so this is 0.0.
        explicit_grad_s2 = 0.0
        total_grad_s2 = jnp.array([implicit_grad_s2 + explicit_grad_s2])

        # --- Gradient w.r.t. Omega ---

        # 5a. Calculate J_cross_o2 for the prior
        def prior_grad_fn(b_in, o2):
            return jax.grad(prior_penalty_chol, argnums=0)(b_in, o2)
        J_cross_o2 = jax.jacobian(prior_grad_fn, argnums=1)(estimated_b_i, omega2)

        # 5b. Calculate implicit and explicit gradients
        implicit_grad_o2 = -v @ J_cross_o2

        # The explicit gradient MUST match the outer loss, which it does here.
        def explicit_omega_loss(o2_to_grad, b_i_fixed):
            inv_o2 = jnp.linalg.inv(o2_to_grad)
            return b_i_fixed @ inv_o2 @ b_i_fixed
        explicit_grad_o2 = jax.grad(explicit_omega_loss, argnums=0)(omega2, estimated_b_i)
        
        total_grad_o2 = implicit_grad_o2 + explicit_grad_o2

        # 6. Return all gradients in the correct order
        return (None, None, grad_data_contrib, None, None,
                grad_pop_coeff, total_grad_s2, total_grad_o2, None)
    
    @jax.custom_vjp
    def estimate_single_b_i(
        initial_b_i, padded_y_i, data_contrib_i, ode_t0_i, time_mask_y_i,
        pop_coeff, sigma2, omega2, pop_coeff_w_bi_idx
    ):
        (estimated_b_i,
         H_foce,
         final_inner_loss_value,
         S_masked,
         H_masked,
         residuals_masked,
         scaling_factors_b, 
         model_coeffs_at_opt
         ) = estimate_single_b_i_impl(
            initial_b_i, padded_y_i, data_contrib_i, ode_t0_i, time_mask_y_i,
            pop_coeff, sigma2, omega2, pop_coeff_w_bi_idx
        )
        
        return estimated_b_i, H_foce, final_inner_loss_value,
        
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

#inter

# 1. Create a new wrapper function specifically for the IFT/VJP version
def debug_loss_ift(opt_params):
    """
    This function is identical to debug_loss, but it passes the IFT estimator
    to the outer loss function.
    """
    p = unpack_partial(opt_params = opt_params)
    
    # Static data for the test
    data_contribution_batch= jnp.array([[0., 0., 0.]])
    p['data_contribution'] = data_contribution_batch
    
    # The only difference is on the `compiled_estimate_b_i_foce` line
    loss_out = debug_omega_loss_only(
        padded_y=padded_y_batch,
        time_mask_y=jnp.array([jnp.repeat(True, padded_y_batch.shape[1])]),
        compiled_augdyn_ivp_solver_novmap_arr = None,
        compiled_2ndorder_augdyn_ivp_solver_novmap_arr = None,
        compiled_ivp_solver_novmap_arr = None,
        ode_t0_vals=None,
        pop_coeff_for_J_idx=jnp.array([0,1,2]), 
        
        # --- KEY CHANGE ---
        # Wire in the custom VJP estimator here
        compiled_estimate_b_i_foce=DEBUG_OMEGA_estimate_b_i_vmapped_ift,
        # --- END KEY CHANGE ---

        use_surrogate_neg2ll=True, 
        inner_optimizer_tol=None,
        inner_optimizer_maxiter=None,
        **p
    )
    
    return loss_out

#%%

print("--- Calculating Ground Truth with finitediffx ---")
fdx_grad = fdx.fgrad(debug_loss)(opt_params)

print("\n--- Calculating Gradient with Custom VJP ---")
jax_grad = jax.grad(debug_loss_ift)(opt_params)


# 3. Compare the final results
print("\n--- MRE Gradient Comparison ---")
print("finitediffx:\n", fdx_grad)
print("\njax.grad (VJP):\n", jax_grad)

#%%