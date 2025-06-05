
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
from jax.experimental import host_callback as hcb
import jax.numpy as jnp
from copy import deepcopy
import jax


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
    me_mod_fo = me_mod_fo.fit2(df, ci_level = None)
else:
    with open(f"logs/fitted_model_{me_mod_fo.model_name}.jb", 'rb') as f:
        me_mod_fo = jb.load(f)
res_df[me_mod_fo.model_name] = me_mod_fo.predict2(df)
piv_cols.append(me_mod_fo.model_name)
me_mod_fo.save_fitted_model(jb_file_name = me_mod_fo.model_name)
#%%
def neg2_log_likelihood_loss2(y_true, y_pred, sigma = None):
    
    residuals = y_true - y_pred
    ss = jnp.sum(residuals**2)
    n = len(y_true)
    if sigma is None:
        sigma = jnp.sqrt(ss/n)
    neg2_log_likelihood = n * jnp.log(2 * jnp.pi * sigma**2) + ss / sigma**2
    return neg2_log_likelihood

nlme_loss = neg2_log_likelihood_loss2(me_mod_fo.y, me_mod_fo.predict2(df))
#%%

ode_solver_jitted = me_mod_fo.jax_ivp_stiff_compiled_solver_
ode_t0_vals = jnp.array(me_mod_fo.ode_t0_vals, dtype = jnp.float64)
flat_time_map = pd.Series(me_mod_fo.time_mask.flatten())
#%%
flat_time_map = jnp.array(flat_time_map[flat_time_map].index)
y_loss = jnp.array(me_mod_fo.y, dtype = np.float64)
df_tmp = me_mod_fo.data.copy()
df_tmp['y'] = me_mod_fo.y
timepoints = me_mod_fo.global_tp_eval
#%%
plot_subject_levels(df, y = 'DV_scale')
#%%

subs = df_tmp['SUBJID'].unique()

#this interpolation does not end up getting used
y_interp = np.empty((len(subs), len(timepoints)))
ys = []
times = []
for sub_idx, sub in enumerate(subs):
    df_loop = df_tmp.loc[df_tmp['SUBJID'] == sub, :]
    y_loop = np.array(df_loop['y'])
    time_loop = np.array(df_loop['TIME'])
    ys.append(y_loop)
    times.append(time_loop)
    y_interp_loop = np.interp(timepoints, time_loop, y_loop)

    y_interp[sub_idx] = y_interp_loop
    


#%%
def neg2_log_likelihood_loss(y_true, y_pred, sigma = None):
    
    residuals = y_true - y_pred
    ss = jnp.sum(residuals**2)
    n = len(y_true)
    if sigma is None:
        sigma = jnp.sqrt(ss/n)
    neg2_log_likelihood = n * jnp.log(2 * jnp.pi * sigma**2) + ss / sigma**2
    return neg2_log_likelihood

def neg2_log_likelihood_loss_np(y_true, y_pred, sigma = None):
    
    residuals = y_true - y_pred
    ss = np.sum(residuals**2)
    n = len(y_true)
    if sigma is None:
        sigma = np.sqrt(ss/n)
    neg2_log_likelihood = n * np.log(2 * np.pi * sigma**2) + ss / sigma**2
    return neg2_log_likelihood


@jax.jit 
def calculate_total_system_loss_for_jax(param_to_optimize_val, 
                                        other_fixed_params,
                                        param_idx,        
                                        ):
            
    
    current_ode_params_vector = other_fixed_params.at[param_idx].set(param_to_optimize_val)
    
    params_in = jnp.exp(current_ode_params_vector)
    params_in = jnp.tile(params_in, (ode_t0_vals.shape[0], 1))
    masses, concs = ode_solver_jitted(
        ode_t0_vals,
        params_in
        )
    #sol_full = masses
    
    #sol_full = sol_full[:,:,0].set(concs)
    sol_dep_var = concs
    sol_dep_var = sol_dep_var.flatten()
    
    sol = sol_dep_var[flat_time_map]
    
    
    
    loss = neg2_log_likelihood_loss(y_loss, sol)
    

    debug_print = False
    if debug_print:
        jax.debug.print(f"y_true: {y_loss}")
        jax.debug.print("Value of sol: {sol}", sol = sol)
        jax.debug.print("Value of loss: {loss}", loss = loss)

    
    return loss

#%%
#test calculate_total_system_loss_for_jax
tmp_df = me_mod_fo.fit_result_summary_
#use the best fit values generated by nlme analysis as the current best ode params for testing
other_params_tmp = tmp_df.loc[tmp_df['population_coeff'], 'best_fit_param_val'].to_numpy().flatten()
#test the loss function jitted on its own
calculate_total_system_loss_for_jax(1.0, jnp.array(other_params_tmp), 0)

#%%
#test the value and grad function without jitting it
jax.value_and_grad(calculate_total_system_loss_for_jax)(1.0, jnp.array(other_params_tmp), 0)
#%%
#%%

loss_and_grad_fn = jax.jit(jax.value_and_grad(calculate_total_system_loss_for_jax))
# test the value and grad function after jitting it
loss_and_grad_fn(1.0, jnp.array(other_params_tmp), 0)
# hessian_fn = jax.jit(jax.hessian(calculate_total_system_loss_for_jax))
#%%
def make_jax_custom_objective(
    other_fixed_params_np, 
    param_idx_to_optimize,
        
    ):


    other_fixed_params_jax = jnp.array(other_fixed_params_np)
    # this function below has the signature output during the training loop when the sklearn api is utilized
    def actual_custom_objective_for_lgbm(y_true, y_pred): 
        # Assuming X_ones, preds[0] is the current value of the parameter from this LGBM
        #print(f"pred coeff {y_pred}")
        param_val_from_lgbm = y_pred[0] 
        
        # JAX computes loss and gradient
        fun_out, grad_value = loss_and_grad_fn(
            param_val_from_lgbm,
            other_fixed_params_jax,
            param_idx_to_optimize,
        )
        loss_value = fun_out

        
        # LightGBM expects grad and hess for each sample.
        # If X_ones, preds are same for all "samples", so grad is same.
        grad_for_lgbm = np.full_like(y_pred, grad_value)
        
        # Hessian: Often simplified to a constant for stability/speed in LightGBM
        hess_for_lgbm = np.full_like(y_pred, 1.0)

        print(f"JAX Obj for P{param_idx_to_optimize}: val={param_val_from_lgbm:.10f}, loss={loss_value:.10f}, grad={grad_value:.10f}")
        return grad_for_lgbm, hess_for_lgbm
        
    return actual_custom_objective_for_lgbm

#%%
M_ode_params = me_mod_fo.pk_args_diffeq
#use the best fit values generated by nlme analysis as the current best ode params for fitting as well
#repeated for clarity
tmp_df = me_mod_fo.fit_result_summary_
current_ode_param_estimates = tmp_df.loc[tmp_df['population_coeff'], 'best_fit_param_val'].to_numpy().flatten()
#%%
import lightgbm as lgb
#%%
sub_data = me_mod_fo.subject_data.copy()
#X_train = pd.DataFrame(np.ones((sub_data.shape[0], 1)), columns = ['x'])
X_train = np.random.random_sample((sub_data.shape[0], 150))
#%%
start_n_estimators = 5
ode_param_models = []
for p_idx, p in enumerate(M_ode_params):
    model = lgb.LGBMRegressor(min_child_samples = 1, 
                                   verbose = 3, 
                                   feature_pre_filter = False,
                                   n_estimators = start_n_estimators
                                   )
    # "Fit" them initially to establish a starting prediction (e.g., our initial guess)
    model.fit(X_train, np.full(X_train.shape[0], current_ode_param_estimates[p_idx]))
    ode_param_models.append(deepcopy(model))

model_j_n_estimators = np.repeat(1, len(M_ode_params), )
#%%

num_outer_iterations = 500 # Total outer loops
N_ode_params = pd.Series(range(len(M_ode_params)))
min = N_ode_params.min()
max = N_ode_params.max()

#plan to perform coordinate descent for the parameters in a random or
#Once this is working, might be better to adjust the params  
#in a dynamic order determined by the magnitude of the gradient 
#on each of the params
iter_param_plan = [N_ode_params.sample(frac = 1, replace = False).to_numpy()
                   for n in range(num_outer_iterations)]
#%%
lgbm_param_plan = pd.DataFrame()
#stochastic learning rate 
learning_rate_plan = np.random.uniform(.0001, .001, num_outer_iterations)
lgbm_param_plan['learning_rate'] = learning_rate_plan
#number of trees to add to the base models each iteration
n_trees_plan = np.random.randint(2,10,  num_outer_iterations)
lgbm_param_plan['n_estimators'] = n_trees_plan
lgbm_param_plan['opt_order'] = iter_param_plan
y_dummy_targets_train = np.zeros(X_train.shape[0])


base_lgbm_params = {
    'num_leaves': 2,
    'min_split_gain': -np.inf, # Allow any leaf formation
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'min_child_samples': 1, # Be permissive
    'feature_pre_filter': False, # You already have this
    #'verbose': -1,
    # 'min_sum_hessian_in_leaf': 1e-5 # If you want to be extra sure
}

#%%
current_ode_param_estimates = current_ode_param_estimates.astype(np.float64)
#current_ode_param_estimates[1] = current_ode_param_estimates[1] * .8
param_history = np.empty((num_outer_iterations, current_ode_param_estimates.shape[0]))
param_history[0, :] = current_ode_param_estimates
for outer_iter_idx, outer_row in lgbm_param_plan.iterrows():
    print(f"Outer Iteration: {outer_iter_idx + 1}/{num_outer_iterations}====================== OUTER ====================")
    # Store the params predicted at the start of this iteration to pass to objectives
    #params_at_iter_start = [model.predict(X_train)[0] for model in ode_param_models]
    params_at_iter_start = current_ode_param_estimates
    print(f"Before: {params_at_iter_start}")
    try:
        alt_before = [model.predict(X_train)[0] for model in ode_param_models]
        print(f"Before alt: {alt_before}")
    except:
        1==1
    for j_idx, j in enumerate(outer_row['opt_order']): # Iterate over each ODE parameter
        print(f" Optimizing Param {j} (Current val: {params_at_iter_start[j]:.10f})")
        
        
        other_params_for_obj = jnp.array(params_at_iter_start)

        custom_obj_for_j = make_jax_custom_objective(
            other_fixed_params_np=other_params_for_obj, # Pass the current state of all params
            param_idx_to_optimize=j,
        )
        
        # Get the current model for parameter j
        
        current_model_j = deepcopy(ode_param_models[j])
        try:
            current_n_estimators = current_model_j.booster_.num_trees()
            print(f'N Estimators in: {current_n_estimators}')
            model_j_n_estimators[j] = current_n_estimators + outer_row['n_estimators']
            print(f'Planned N Estimators out: {model_j_n_estimators[j]}')
            print(f"Estimator iters in: {current_model_j.booster_.current_iteration()}")
            first_iter = False
        except:
            model_j_n_estimators[j] = 1
            first_iter = True
            print(f'Planned N Estimators out: {model_j_n_estimators[j]}')
        
        
        #update the params
        current_model_j.set_params(objective=custom_obj_for_j,
                                   n_estimators=model_j_n_estimators[j],
                                   learning_rate = outer_row['learning_rate'],
                                   verbose = 2, 
                                   num_leaves= 25,
                                    min_split_gain= 0.0, # Allow any leaf formation
                                    reg_alpha= 0.0,
                                    reg_lambda= 0.0,
                                    min_child_samples= 1, # Be permissive
                                    min_child_weight = 1e-5,
                                    feature_pre_filter= False,
                                   )
        #fit the model for a bit
        current_model_j = current_model_j.fit(
            X_train,
            y_dummy_targets_train, # Targets are not used by fobj directly
            init_model=None if first_iter else current_model_j  # Warm start
        )
        
        print(f'Actual n estimators out: {current_model_j.booster_.num_trees()}')
        print(f"Estimator iters out: {current_model_j.booster_.current_iteration()}")
        ode_param_models[j] = deepcopy(current_model_j)
        
        # Update the global estimate with the new prediction from this model
        current_ode_param_estimates[j] = current_model_j.predict(X_train)[0]
        print(f"  Updated Param {j} to: {current_ode_param_estimates[j]:.10f}")
    insert_idx = outer_iter_idx + 1
    param_history[insert_idx, :] = current_ode_param_estimates
    
    print(f"After: {current_ode_param_estimates}")
    print(f" End of Outer Iter {outer_iter_idx+1}")
    
    if insert_idx - 5 >= 0:
        check_diff = 0
        for i in range(5):
            tmp_idx = i + 1
            check_idx = insert_idx - tmp_idx
            param_abs_diff = np.abs(param_history[insert_idx, :] - param_history[check_idx, :])
            check_diff = check_diff + np.sum(param_abs_diff)
        print(f"Early Stop Diff: {check_diff}")
        if check_diff < 1e-5:
            print('Parameter Estimates have stabilized, early stopping')
            break
    # Optionally, evaluate overall loss on a validation set here

final_population_params = [model.predict(X_train)[0] for model in ode_param_models]
print(f"\nFinal Estimated Population Parameters: Alpha={final_population_params[0]:.10f}, Beta={final_population_params[1]:.10f}")
#print(f"True population parameters were: Alpha={TRUE_ALPHA:.10f}, Beta={TRUE_BETA:.10f}")
#%%



# %%
