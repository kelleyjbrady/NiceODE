import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize, brentq
from copy import deepcopy

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
                options = {'disp':True if param_index == 1 else False, 
                           "initial_tr_radius":initial_tr_radius,
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