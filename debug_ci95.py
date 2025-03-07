import joblib as jb
import numpy as np
from scipy.stats import chi2
from utils import CompartmentalModel, PopulationCoeffcient, ODEInitVals, neg2_log_likelihood_loss
from scipy.optimize import minimize, brentq
from diffeqs import( 
                    first_order_one_compartment_model, #dy/dt = -k * C
                    first_order_one_compartment_model2, #dy/dt = -cl/vd * C

                    )
from model_assesment import construct_profile_ci


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
     population_coeff=[PopulationCoeffcient('cl', 15, ),
                       PopulationCoeffcient('vd', 45
                                            , optimization_lower_bound = np.log(35)
                                            , optimization_upper_bound = np.log(55)
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
        


res_dict = {}
for param_idx, param_val in enumerate(no_me_mod.fit_result_.x):
    print(f"Profiling parameter:{param_idx}")
    res_dict = construct_profile_ci(model_obj = no_me_mod, df = df, param_index=param_idx)

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