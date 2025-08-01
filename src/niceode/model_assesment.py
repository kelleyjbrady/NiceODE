import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize, brentq
from copy import deepcopy
import numdifftools as nd
import jax
import jax.numpy as jnp
from scipy.stats import t as scipy_t, norm as scipy_norm
from functools import partial

def construct_profile_ci(model_obj,df, param_index,init_bounds_factor = 1.01, profile_bounds = None, profile_bounds_factor = 2.0, ci_level = 0.95, result_list = None ):
    print(f"Starting profiling CI construction for parameter {param_index}")
    if result_list is None:
        result_list = []
    orig_stiff_state = deepcopy(model_obj.stiff_ode)
    model_obj.stiff_ode = True
    
    fit_result = model_obj.fit_result_
    best_fit_params = fit_result.x.copy()
    target_loss = model_obj.fit_result_.fun
    _, _, _, theta_data = model_obj._assemble_pred_matrices(df)
    best_fit_neg2_log_likelihood = fit_result.fun
    param_range = [
        np.log((1/init_bounds_factor)* np.exp(best_fit_params[param_index]))
        ,np.log(init_bounds_factor * np.exp(best_fit_params[param_index]))
                   ]
    chi2_quantile = chi2.ppf(ci_level, 1)
    
    def objective_for_profiling(other_params, fixed_param_index, fixed_param_val):
        # Create a new parameter vector with the profiled parameter fixed
        #print(other_params)
        profiled_params = other_params.copy()
        profiled_params = np.insert(other_params,
                                    fixed_param_index,
                                    fixed_param_val)
        loss = model_obj._objective_function2(profiled_params, theta_data)
        return loss
    
    
    lower_bound = find_profile_bound(objective_func=objective_for_profiling,
                                 param_index = param_index, best_fit_params = best_fit_params,
                                 best_nll = best_fit_neg2_log_likelihood,
                                 chi2_quantile=chi2_quantile, start = param_range[0],
                                 end = best_fit_params[param_index],
                                 profile_bounds=profile_bounds, 
                                 profile_bounds_factor=profile_bounds_factor,
                                 lower=True)

    upper_bound = find_profile_bound(objective_func = objective_for_profiling, param_index = param_index,
                                     best_fit_params=best_fit_params,
                                    best_nll = best_fit_neg2_log_likelihood, chi2_quantile=chi2_quantile,
                                    start = best_fit_params[param_index],
                                    end = param_range[1],
                                    profile_bounds=profile_bounds, 
                                 profile_bounds_factor=profile_bounds_factor, lower=False)
    ci_label =    int(ci_level*100) 
    result_list.append( {f'ci{ci_label}_lower':lower_bound,
                                f'ci{ci_label}_upper':upper_bound}
                        )
    
    model_obj.stiff_ode = deepcopy(orig_stiff_state)
    print(f"Completed profiling CI construction for parameter {param_index}")
    return result_list
    
    


def find_profile_bound(objective_func, param_index,
                       best_fit_params, best_nll, chi2_quantile,
                       start, end, 
                       profile_bounds = None, profile_bounds_factor = 2.0, lower=True):
    """
    Finds the lower or upper bound of the profile likelihood confidence interval using a search algorithm.
    """
    
    other_params = np.delete(best_fit_params, param_index)
    initial_tr_radius = np.min([np.exp(np.max(np.abs(other_params)))*.07, 1.0])
    if profile_bounds is not None:
        other_p_bounds = profile_bounds
    else:
        other_p_bounds = [(np.log((1/profile_bounds_factor)*i), np.log(profile_bounds_factor*i)) for i in np.exp(other_params)]
    
    def root_function(param_value):
        try:
            # --- Existing minimize call ---
            result = minimize(
                objective_func,  # This is objective_for_profiling
                other_params,
                args=(param_index, param_value),
                bounds=other_p_bounds,
                method='COBYQA', 
                options = {'disp':False, 
                           "initial_tr_radius":initial_tr_radius,
                           #this final tr radius is quite high, however . . . 
                           #the point of this is to find the bounds of the region 
                           #within which the brentq solver will be applied
                           #so it probably doesnt matter if this is noisy as the point 
                           #of the increment decrement procedure to is find imprecise bounds. 
                           "final_tr_radius":1e-2 
                           }# Or your chosen method
                # Consider adding options like {'maxiter': some_val, 'eps': some_tol}
            )

            # Check if optimizer was successful AND if result.fun is valid
            if not result.success or not np.isfinite(result.fun):
                print(f"Optimizer failed or produced invalid NLL for {param_value=}")
                return 1e12 # Large penalty

            val_to_root = (result.fun - best_nll) - chi2_quantile
            if not np.isfinite(val_to_root): # Check for NaN/inf from subtraction
                print(f"Non-finite value for root function at {param_value=}")
                return 1e12 # Large penalty
            return val_to_root

        except Exception as e: # Catch any error, including from ODE solver via objective_func
            print(f"Exception in root_function for {param_value=}: {e}")
            return 1e12 # Return a large penalty to indicate a bad region 
    
    if lower:
        a = start
        decrement = .05*end
        #decrement = .01
        while root_function(a) < 0:
            a -= decrement
            if a < start - 10:
                bound = None

        a, b = a, end
    elif not lower:
        b = end
        increment = .05*start
        #Search for the upper bound using the bisection method
        while root_function(b) < 0:
            b += increment
            if b > end + 10:
                bound = None
        a, b = start, b
    use_brentq = True
    if use_brentq:
        bound = brentq(root_function, a, b)
    return bound


def estimate_per_subject_loss(params_jax, use_surrogate_neg2ll = True, _jax_objective_function_predict_ = None, loss_bundle = None):
    if loss_bundle is None:
        loss_bundle = _jax_objective_function_predict_(params_jax)
    per_subject_loss_components = loss_bundle[1]['per_subject_loss']
    log_det_i = per_subject_loss_components[0][0]
    quadratic_i = per_subject_loss_components[0][1]
    inner_loss_i = per_subject_loss_components[1] #this is the interaction term, not the inner loss
    n_t_i = per_subject_loss_components[2]
    if use_surrogate_neg2ll:
        n_t_scaler_i = 0.0
    else:
        n_t_scaler_i = n_t_i.reshape(-1,1) @ jnp.array([jnp.log(2 * jnp.pi)])
    per_subject_loss = log_det_i + quadratic_i + n_t_scaler_i + inner_loss_i
    return per_subject_loss

def get_robust_ci_foce_ndt(
    scipy_result,
    _jax_objective_function_predict_,
    predict_unpack,
    #static_args_for_vmap,
    init_params_for_scaling,
    n_subjects, 
    ci_level, 
    n_boot = 5000, 
    ci_dist = 't',
):
    """
    Calculates robust CIs for a FOCE/FOCEi model using a sandwich estimator
    with gradients from the numdifftools library.
    """
    
    ci_dist = scipy_t if ci_dist == 't' else scipy_norm
    
    final_params_np = scipy_result.x
    
    # 1. Extract the Hessian ("Bread") from the optimizer result
    #H_inv_scipy = scipy_result.hess_inv.todense()
    #H = np.linalg.inv(H_inv) # Keep as numpy for now

    # --- Calculate the "Filling" (G) using numdifftools ---
    
    # 2. Define the JIT-compiled function that returns the vector of per-subject losses.
    per_sub_loss_partial = jax.jit(partial(estimate_per_subject_loss,
                                    _jax_objective_function_predict_ = _jax_objective_function_predict_,
                                    ))
        
    # 3. Create a wrapper for numdifftools.
    #    It must take a NumPy array and return a NumPy array.
    def loss_vector_func_for_ndt(params_np: np.ndarray) -> np.ndarray:
        # a. Convert NumPy input to JAX array
        params_jax = jnp.asarray(params_np)
        # b. Call the fast, JIT-compiled JAX function
        losses_jax = per_sub_loss_partial(params_jax)
        # c. Convert JAX output back to NumPy array
        return np.asarray(losses_jax)
    
    def total_loss_vector_func_for_ndt(params_np: np.ndarray) -> np.ndarray:
        # a. Convert NumPy input to JAX array
        params_jax = jnp.asarray(params_np)
        # b. Call the fast, JIT-compiled JAX function
        losses_jax = per_sub_loss_partial(params_jax)
        # c. Convert JAX output back to NumPy array
        return np.asarray(jnp.sum(losses_jax))
    
    print("Calculating Hessian of total loss (H)...")
    H_calculator = nd.Hessian(total_loss_vector_func_for_ndt)
    H = H_calculator(final_params_np)
    H_inv = np.linalg.inv(H)
    print("Hessian calculation complete.")

    # 4. Use numdifftools.Jacobian to calculate the per-subject gradients.
    #    The result, J_scores, will have shape (n_subjects, n_params).
    print("Calculating Jacobian of per-subject losses with numdifftools...")
    J_calculator = nd.Jacobian(loss_vector_func_for_ndt)
    J_scores = J_calculator(final_params_np)
    print("Jacobian calculation complete.")

    # 5. Calculate the cross-product of the gradients (G).
    G = J_scores.T @ J_scores
    
    # --- Assemble the Sandwich ---
    
    # 6. Calculate the robust covariance matrix: Cov = H_inv @ G @ H_inv
    robust_cov_matrix = H_inv @ G @ H_inv

    # 7. Calculate Standard Errors and Confidence Intervals
    std_errors = np.sqrt(np.diag(robust_cov_matrix))
    #std_errors_log_centered = std_errors
    #std_errors_log_scale = std_errors_log_centered 
    
    alpha = (1 - ci_level)
    ci_label = int(100*ci_level) 
    t_crit = ci_dist.ppf(1 - alpha / 2, df=n_subjects - len(final_params_np))
    
    lower_ci_centered = final_params_np - t_crit * std_errors
    upper_ci_centered = final_params_np + t_crit * std_errors
    
    params_log_scale = final_params_np + init_params_for_scaling
    lower_ci_log = lower_ci_centered + init_params_for_scaling
    upper_ci_log = upper_ci_centered + init_params_for_scaling
    
    params_true_scale = np.exp(params_log_scale)
    lower_ci_true_scale = np.exp(lower_ci_log)
    upper_ci_true_scale = np.exp(upper_ci_log)
    
    #lower_ci_log = params_log_scale - t_crit * std_errors_log_scale
    #upper_ci_log = params_log_scale + t_crit * std_errors_log_scale
    
    
    
    std_errors_true_scale = params_true_scale * std_errors
    
    #This is very slow and should be rewritted.
    # Also probably better to just use MCMC if you want these
    boot_additional_ci = False
    if boot_additional_ci:
        param_samples = np.random.multivariate_normal(
            mean=final_params_np,
            cov=robust_cov_matrix,
            size=n_boot
        )
        
        sigma2_samples = []
        omega2_samples = []
        correlation_samples = []
        omega_diag_samples = []
        
        for i in range(n_boot):
            sample = param_samples[i, :]
            boot_params = predict_unpack(sample)
            
            sigma2_sample = boot_params['sigma2']
            omega2_sample = boot_params['omega2']
            omegas1_diag = np.diag(omega2_sample)
            omegas1_diag = np.sqrt(omegas1_diag)
            sd_matrix = np.outer(omegas1_diag, omegas1_diag)
            corr_matrix_sample = np.copy(omega2_sample / (sd_matrix + 1e-9))

            sigma2_samples.append(sigma2_sample)
            omega2_samples.append(omega2_sample)
            correlation_samples.append(corr_matrix_sample)
            omega_diag_samples.append(omegas1_diag)
        
        lower_p = alpha / 2 * 100
        upper_p = (1 - alpha / 2) * 100
        
        sigma2_ci = np.percentile(np.array(sigma2_samples), [lower_p, upper_p])
        omega2_ci = np.percentile(np.array(omega2_samples), [lower_p, upper_p], axis=0)
        correlation_ci = np.percentile(np.array(correlation_samples), [lower_p, upper_p], axis=0)
        
        # Also get the mean of the simulations as the point estimate
        final_omega2 = np.mean(np.array(omega2_samples), axis=0)
        final_sigma2 = np.mean(np.array(sigma2_samples))
        final_correlations = np.mean(np.array(correlation_samples), axis=0)
    
    return {
        "opt_std_errors": std_errors,
        f"opt_lower_ci{ci_label}": lower_ci_centered,
        f"opt_upper_ci{ci_label}": lower_ci_centered,
        "opt_robust_cov_matrix": robust_cov_matrix, 
        
        "log_params":params_log_scale,
        "log_std_errors": std_errors,
        f"log_lower_ci{ci_label}": lower_ci_log,
        f"log_upper_ci{ci_label}": upper_ci_log,
        
        "true_params":params_true_scale,
        "true_std_errors": std_errors_true_scale,
        f"true_lower_ci{ci_label}": lower_ci_true_scale,
        f"true_upper_ci{ci_label}": upper_ci_true_scale,
    }, (ci_level, ci_label)