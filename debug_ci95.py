import joblib as jb
import numpy as np
from scipy.stats import chi2
from utils import CompartmentalModel, PopulationCoeffcient, ODEInitVals, neg2_log_likelihood_loss
from scipy.optimize import minimize, brentq
from diffeqs import( 
                    first_order_one_compartment_model, #dy/dt = -k * C
                    first_order_one_compartment_model2, #dy/dt = -cl/vd * C

                    )

with open(r'/workspaces/miniconda/PK-Analysis/debug_scale_df.jb', 'rb') as f:
    df = jb.load(f)
    

no_me_mod_k=  CompartmentalModel(
     ode_t0_cols=[ODEInitVals('DV')],
     population_coeff=[PopulationCoeffcient('k', 0.3, ),
                       #PopulationCoeffcient('vd', 20, ),
                       ],
     model_error_sigma=PopulationCoeffcient('sigma',
                                            optimization_init_val=4, 
                                            optimization_lower_bound=0.000001, 
                                            optimization_upper_bound=20),
     dep_vars= None, 
                              no_me_loss_function=neg2_log_likelihood_loss, 
                              optimizer_tol=None, 
                              pk_model_function=first_order_one_compartment_model, 
                              #ode_solver_method='BDF'
                              )

no_me_mod =  CompartmentalModel(
     ode_t0_cols=[ODEInitVals('DV')],
     population_coeff=[PopulationCoeffcient('cl', 25, ),
                       PopulationCoeffcient('vd', 80
                                            , optimization_lower_bound = np.log(70)
                                            , optimization_upper_bound = np.log(90)
                                            ),
                       ],
     dep_vars= None, 
     model_error_sigma=PopulationCoeffcient('sigma',
                                            optimization_init_val=4, 
                                            optimization_lower_bound=0.000001, 
                                            optimization_upper_bound=20),
                              no_me_loss_function=neg2_log_likelihood_loss, 
                              optimizer_tol=None, 
                              pk_model_function=first_order_one_compartment_model2, 
                              #ode_solver_method='BDF'
                              )

no_me_mod = no_me_mod.fit2(df,checkpoint_filename=f'mod_abs_test_nome.jb', parallel=False, parallel_n_jobs=4)

fit_result = no_me_mod.fit_result_
param_index = 0
ci_level = .95
best_fit_params = fit_result.x.copy()
_, _, _, beta_data = no_me_mod._assemble_pred_matrices(df)
best_fit_neg_log_likelihood = fit_result.fun
param_range = [np.log(0.99* np.exp(best_fit_params[param_index])), np.log(1.01 * np.exp(best_fit_params[param_index]))] # Initial range

# critical value of the chi-squared distribution
chi2_quantile = chi2.ppf(ci_level, 1)

def objective_for_profiling(other_params, fixed_param_index, fixed_param_val):
      
    # Create a new parameter vector with the profiled parameter fixed
    profiled_params = other_params.copy()
    profiled_params = np.insert(other_params,
                                fixed_param_index,
                                fixed_param_val)
    loss = no_me_mod._objective_function2(profiled_params, beta_data)
    return loss

def find_profile_bound(objective_func, param_index, best_fit_params, best_nll, chi2_quantile, start, end, lower=True):
    """
    Finds the lower or upper bound of the profile likelihood confidence interval using a search algorithm.
    """
    
    tolerance = 1e-4  # Set a suitable tolerance
    max_iterations = 25
    other_params = np.delete(best_fit_params, param_index)
    #These are VERY IMPORTANT
    #would be even better to use the 'CI' derived from NCA for at least Vd
    bounds_factor = 2.0
    other_p_bounds = [(np.max([np.log((1/bounds_factor)*i), 1e-3]), np.log(bounds_factor*i)) for i in np.exp(other_params)]
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
                return None

        a, b = a, end
    elif not lower:
        b = end
        increment = .005*start
        #Search for the upper bound using the bisection method
        while root_function(b) < 0:
            b += increment
            if b > end + 10:
                return None
        a, b = start, b
    use_brentq = True
    if use_brentq:
        bound = brentq(root_function, a, b)
        return bound
    else:
        for _ in range(max_iterations):
            mid = np.log((np.exp(a)+np.exp(b))/2)
            fval = root_function(mid)
            if fval > 0:
                if lower:
                    b = mid
                else:
                    a = mid
            else:
                if lower:
                    a= mid
                else:
                    b = mid
            if abs(b - a) < tolerance:
                return mid

lower_bound = find_profile_bound(objective_for_profiling,
                                 param_index, best_fit_params,
                                 best_fit_neg_log_likelihood,
                                 chi2_quantile, param_range[0],
                                 best_fit_params[param_index], lower=True)

upper_bound = find_profile_bound(objective_for_profiling, param_index, best_fit_params,
                                 best_fit_neg_log_likelihood, chi2_quantile, best_fit_params[param_index],
                                 param_range[1], lower=False)

# If bounds found, create finer profile
profile_parameter_values = []
profile_nll_values = []

if lower_bound is not None and upper_bound is not None:
    profile_parameter_values = np.linspace(lower_bound, upper_bound, 20)
    for val in profile_parameter_values:
        # Optimize other parameters with the current parameter fixed
        result = minimize(
            objective_for_profiling,
            np.delete(best_fit_params, param_index),
            args=(param_index, val),
            method='L-BFGS-B'
        )
        profile_nll_values.append({'neg2_ll':result.fun, 'x':result.x})