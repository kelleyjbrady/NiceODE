
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64']='True'
import jax
import pandas as pd
from niceode.diffeqs import OneCompartmentAbsorption, TwoCompartmentAbsorption
from niceode.utils import CompartmentalModel, ObjectiveFunctionColumn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from niceode.utils import plot_subject_levels, neg2_log_likelihood_loss, FOCE_approx_ll_loss
import joblib as jb
from niceode.utils import sum_of_squares_loss, PopulationCoeffcient, ODEInitVals, mean_squared_error_loss, huber_loss
import cProfile
from datetime import datetime
import uuid
import numpy as np

from niceode.nca import estimate_subject_slope_cv
from niceode.nca import identify_low_conc_zones, estimate_k_halflife
from niceode.nca import calculate_mrt
from niceode.nca import prepare_section_aucs, calculate_auc_from_sections
from copy import deepcopy

#%%
with open(r'/workspaces/PK-Analysis/debug/cp1805_prep.jb', 'rb') as f:
    df = jb.load(f)

#%%
res_df = pd.DataFrame()
res_df[['SUBJID', 'TIME',  'DV_scale']] = df[['ID_x', 'TIME_x', 'DV_scale']].copy()
piv_cols = ['DV_scale']
#%%

me_mod_fo =  CompartmentalModel(
        model_name = "debug_cp1805_abs_ka-clME-vd_sse_nodep_dermal",
            ode_t0_cols=[ ODEInitVals('DV_scale'), ODEInitVals('AMT_scale'),],
            conc_at_time_col = 'DV_scale',
            subject_id_col = 'ID_x', 
            time_col = 'TIME_x',
            population_coeff=[
                                PopulationCoeffcient('ka', 
                                                    optimization_init_val=2, 
                                                    #subject_level_intercept=True,
                                                    optimization_lower_bound = np.log(1e-6),
                                                    #optimization_upper_bound = np.log(3),
                                                    #subject_level_intercept_sd_init_val = 0.2, 
                                                    #subject_level_intercept_sd_upper_bound = 20,
                                                    #subject_level_intercept_sd_lower_bound=1e-6
                                                    ),
                                PopulationCoeffcient('cl',
                                                    optimization_init_val = .1,
                                                    optimization_lower_bound = np.log(1e-4),
                                                    optimization_upper_bound=np.log(1)
                                                    #optimization_upper_bound = np.log(.005),
                                                    #subject_level_intercept=True, 
                                                    #subject_level_intercept_sd_init_val = 0.3, 
                                                    #subject_level_intercept_sd_upper_bound = 5,
                                                    #subject_level_intercept_sd_lower_bound=1e-6
                                                    ),
                                PopulationCoeffcient('vd', optimization_init_val = 1.2
                                                    , optimization_lower_bound = np.log(.1)
                                                    ,optimization_upper_bound=np.log(5)
                                                    
                                                    #, optimization_upper_bound = np.log(.05)
                                                    ),
                            ],
            dep_vars= None, 
                                    no_me_loss_function=neg2_log_likelihood_loss, 
                                    no_me_loss_needs_sigma=True,
                                    #optimizer_tol=None, 
                                    pk_model_class=OneCompartmentAbsorption, 
                                    model_error_sigma=PopulationCoeffcient('sigma'
                                                                            ,log_transform_init_val=False
                                                                            , optimization_init_val=.5
                                                                            ,optimization_lower_bound=0.00001
                                                                            ,optimization_upper_bound=3
                                                                            ),
                                    #ode_solver_method='BDF'
                                    batch_id='mlflow_test_batch9',
                                    minimize_method = 'COBYQA'
                                    )

fit_model = True
if fit_model:
    me_mod_fo = me_mod_fo.fit2(df, )
else:
    with open(f"logs/fitted_model_{me_mod_fo.model_name}.jb", 'rb') as f:
        me_mod_fo = jb.load(f)
res_df[me_mod_fo.model_name] = me_mod_fo.predict2(df)
piv_cols.append(me_mod_fo.model_name)
me_mod_fo.save_fitted_model(jb_file_name = me_mod_fo.model_name)
#%%
import jax.numpy as jnp
from copy import deepcopy
import jax
ode_solver_jittable = deepcopy(me_mod_fo.jax_ivp_stiff_jittable_)
ode_t0_vals = jnp.array(me_mod_fo.ode_t0_vals, dtype = jnp.float64)
flat_time_map = jnp.array(me_mod_fo.time_mask.flatten(), dtype = jnp.float64)
y = jnp.array(me_mod_fo.y, dtype = np.float64)
# --- Assume these are defined using JAX ---
# def jax_ode_system(y, t, ode_params_vector): ...
# def jax_rk4_solver_step(y, t, dt, ode_params_vector): ... (or use a library like Diffrax)
# def jax_run_ivp_solver(ode_params_vector, ic, t_eval): ...
# def jax_calculate_profile_loss(y_pred_ivp, y_true_profile): ...
# --- End JAX definitions ---

# Global current best estimates for other ODE parameters
# (as in the previous iterative LightGBM example)
# other_ode_params_values = [...]
# param_index_being_optimized = ... # e.g., 0 for alpha
#%%
def neg2_log_likelihood_loss(y_true, y_pred, sigma = None):
    
    residuals = y_true - y_pred
    ss = jnp.sum(residuals**2)
    n = len(y_true)
    if sigma is None:
        sigma = jnp.sqrt(ss/n)
    neg2_log_likelihood = n * jnp.log(2 * jnp.pi * sigma**2) + ss / sigma**2
    return neg2_log_likelihood



# This is the function JAX will differentiate
@jax.jit
def calculate_total_system_loss_for_jax(param_to_optimize_val, # This is what LightGBM predicts (preds[0])
                                        other_fixed_params,
                                        param_idx,                  # JAX arrays
                                        ):
            
    
    current_ode_params_vector = other_fixed_params.at[param_idx].set(param_to_optimize_val)
    
    masses, concs = ode_solver_jittable(
        ode_t0_vals,
        current_ode_params_vector
        )
    sol_full = masses
    
    sol_full = sol_full[:,:,0].set(concs)
    sol_dep_var = concs
    sol_dep_var = sol_dep_var.flatten()
    
    sol = sol_dep_var[flat_time_map]
    # Construct the full ODE parameter vector
    
    
    
    loss = neg2_log_likelihood_loss(y, sol)
    
    return loss

# Get JAX functions for gradient and value+gradient
loss_and_grad_fn = jax.jit(jax.value_and_grad(calculate_total_system_loss_for_jax))
# hessian_fn = jax.jit(jax.hessian(calculate_total_system_loss_for_jax)) # Full Hessian can be expensive
#%%
def make_jax_custom_objective(
    other_fixed_params_np, # Numpy array
    param_idx_to_optimize,
             # Numpy array
    ):

    # Convert to JAX arrays once when the objective is created (or per call if they change)
    # For efficiency, these could be pre-converted and passed if static across calls for a given model fit
    other_fixed_params_jax = jnp.array(other_fixed_params_np)

    def actual_custom_objective_for_lgbm(preds, train_data): # preds is from LightGBM
        # Assuming X_ones, preds[0] is the current value of the parameter from this LGBM
        param_val_from_lgbm = preds[0] 
        
        # JAX computes loss and gradient
        loss_value, grad_value = loss_and_grad_fn(
            param_val_from_lgbm,
            other_fixed_params_jax,
            param_idx_to_optimize,
        )
        
        # LightGBM expects grad and hess for each sample.
        # If X_ones, preds are same for all "samples", so grad is same.
        grad_for_lgbm = np.full_like(preds, grad_value)
        
        # Hessian: Often simplified to a constant for stability/speed in LightGBM
        # e.g., hess_for_lgbm = np.full_like(preds, 1.0)
        # Or, if you compute it with JAX:
        # hess_value = hessian_fn(param_val_from_lgbm, ...)
        # hess_for_lgbm = np.full_like(preds, hess_value) # This assumes scalar hessian if 1 param
        hess_for_lgbm = np.full_like(preds, 1.0) # Placeholder

        # print(f"JAX Obj for P{param_idx_to_optimize}: val={param_val_from_lgbm:.4f}, loss={loss_value:.4f}, grad={grad_value:.4f}")
        return grad_for_lgbm, hess_for_lgbm
        
    return actual_custom_objective_for_lgbm

ode_param_models = []
for i in range(M_ode_params):
    model = lgb.LGBMRegressor(**lgbm_params_common)
    # "Fit" them initially to establish a starting prediction (e.g., our initial guess)
    model.fit(X_dummy_train, np.full(n_profiles_train, current_ode_param_estimates[i]))
    ode_param_models.append(model)

num_outer_iterations = 20 # Total outer loops
for outer_iter in range(num_outer_iterations):
    print(f"Outer Iteration: {outer_iter + 1}/{num_outer_iterations}")
    # Store the params predicted at the start of this iteration to pass to objectives
    params_at_iter_start = [model.predict(X_dummy_train)[0] for model in ode_param_models]

    for j in range(M_ode_params): # Iterate over each ODE parameter
        print(f" Optimizing Param {j} (Current val: {params_at_iter_start[j]:.4f})")
        
        # Create the specific objective for the j-th parameter
        # It uses the most recent estimates of other parameters
        # Note: This uses params_at_iter_start. For true coordinate descent,
        # you might use the *very latest* predictions if parameters are updated sequentially within the loop.
        # Using params_at_iter_start makes each parameter update based on the state at the start of the outer iter.
        
        other_params_for_obj = list(params_at_iter_start) # Copy

        custom_obj_for_j = make_custom_objective(
            param_index_to_optimize=j,
            all_current_params=other_params_for_obj, # Pass the current state of all params
            y_true_profiles=y_profiles_data[:n_profiles_train], # Use training subset
            initial_conditions_profiles=y_ics_data[:n_profiles_train],
            t_eval_points=t_eval_points
        )
        
        # Get the current model for parameter j
        current_model_j = ode_param_models[j]
        
        # Train/update this model for a few estimators
        # `init_model` allows for continued training (boosting)
        current_model_j.set_params(fobj=custom_obj_for_j, n_estimators=lgbm_params_common['n_estimators'])
        current_model_j.fit(
            X_dummy_train,
            y_dummy_targets_train, # Targets are not used by fobj directly
            init_model=current_model_j if outer_iter > 0 or j > 0 else None # Warm start
        )
        
        # Update the global estimate with the new prediction from this model
        current_ode_param_estimates[j] = current_model_j.predict(X_dummy_train)[0]
        print(f"  Updated Param {j} to: {current_ode_param_estimates[j]:.4f}")

    print(f" End of Outer Iter {outer_iter+1}: Current Pop Params: Alpha={current_ode_param_estimates[0]:.4f}, Beta={current_ode_param_estimates[1]:.4f}")
    # Optionally, evaluate overall loss on a validation set here

final_population_params = [model.predict(X_dummy_train)[0] for model in ode_param_models]
print(f"\nFinal Estimated Population Parameters: Alpha={final_population_params[0]:.4f}, Beta={final_population_params[1]:.4f}")
print(f"True population parameters were: Alpha={TRUE_ALPHA:.4f}, Beta={TRUE_BETA:.4f}")

# Final evaluation
final_avg_loss = 0
for i in range(n_profiles_total):
    y_pred_final = run_ivp_solver(final_population_params, y_ics_data[i], 
                                  (t_eval_points[0], t_eval_points[-1]), t_eval_points)
    final_avg_loss += calculate_profile_loss(y_pred_final, y_profiles_data[i])
print(f"Final average loss on all profiles: {final_avg_loss/n_profiles_total:.6f}")


# %%
