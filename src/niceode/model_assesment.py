import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize, brentq

def construct_profile_ci(model_obj,df, param_index,init_bounds_factor = 1.01, profile_bounds = None, profile_bounds_factor = 2.0, ci_level = 0.95, result_list = None ):
    if result_list is None:
        result_list = []
    
    fit_result = model_obj.fit_result_
    best_fit_params = fit_result.x.copy()
    _, _, _, theta_data = model_obj._assemble_pred_matrices(df)
    best_fit_neg2_log_likelihood = fit_result.fun
    param_range = [
        np.log((1/init_bounds_factor)* np.exp(best_fit_params[param_index]))
        ,np.log(init_bounds_factor * np.exp(best_fit_params[param_index]))
                   ]
    chi2_quantile = chi2.ppf(ci_level, 1)
    
    def objective_for_profiling(other_params, fixed_param_index, fixed_param_val):
        # Create a new parameter vector with the profiled parameter fixed
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
                                    end = param_range[1],profile_bounds=profile_bounds, 
                                 profile_bounds_factor=profile_bounds_factor, lower=False)
    ci_label =    int(ci_level*100) 
    result_list.append( {f'ci{ci_label}_lower':lower_bound,
                                f'ci{ci_label}_upper':upper_bound}
                        )
    return result_list
    
    


def find_profile_bound(objective_func, param_index,
                       best_fit_params, best_nll, chi2_quantile,
                       start, end, profile_bounds = None, profile_bounds_factor = 2.0, lower=True):
    """
    Finds the lower or upper bound of the profile likelihood confidence interval using a search algorithm.
    """
    
    other_params = np.delete(best_fit_params, param_index)

    if profile_bounds is not None:
        other_p_bounds = profile_bounds
    else:
        other_p_bounds = [(np.log((1/profile_bounds_factor)*i), np.log(profile_bounds_factor*i)) for i in np.exp(other_params)]
    def root_function(param_value):
        result = minimize(
            objective_func,
            other_params,
            args=(param_index, param_value),
            bounds=other_p_bounds,
            method='L-BFGS-B'
        )
        return (result.fun - best_nll) - chi2_quantile
    
    
    if lower:
        a = start
        decrement = .005*end
        #decrement = .01
        while root_function(a) < 0:
            a -= decrement
            if a < start - 10:
                bound = None

        a, b = a, end
    elif not lower:
        b = end
        increment = .005*start
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