#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_subject_levels
import joblib as jb
from utils import CompartmentalModel, PopulationCoeffcient, ODEInitVals, neg2_log_likelihood_loss
from diffeqs import( 
                    first_order_one_compartment_model, #dy/dt = -k * C
                    first_order_one_compartment_model2, #dy/dt = -cl/vd * C

                    )
import numpy as np
from pymc_utils import make_pymc_model
import pymc as pm
from datetime import datetime

import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

now_str = datetime.now().strftime("%d%m%Y-%H%M%S")
with open(r'/workspaces/miniconda/PK-Analysis/debug_scale_df.jb', 'rb') as f:
    df = jb.load(f)
base_p = "/workspaces/miniconda/PK-Analysis"
logs_path = os.path.join(base_p, 'logs')
if not os.path.exists(logs_path):
    os.makedirs(logs_path)


fit_model = True
if fit_model:
    dump_path = os.path.join(logs_path, f'no_me_mod_test_debug_obj_{now_str}.jb')
    no_me_mod =  CompartmentalModel(
        ode_t0_cols=[ODEInitVals('DV')],
        population_coeff=[PopulationCoeffcient('cl', 15,
                                                subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.2, 
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                ),
                        PopulationCoeffcient('vd', 45
                                                , optimization_lower_bound = np.log(35)
                                                , optimization_upper_bound = np.log(55)
                                                ),
                        ],
        dep_vars= None, 
        model_error_sigma=PopulationCoeffcient('sigma',
                                                log_transform_init_val=False,
                                                optimization_init_val=.5, 
                                                optimization_lower_bound=0.000001, 
                                                optimization_upper_bound=20),
                                no_me_loss_function=neg2_log_likelihood_loss, 
                                optimizer_tol=None, 
                                pk_model_function=first_order_one_compartment_model2, 
                                #ode_solver_method='BDF'
                                )
    write_path = os.path.join(logs_path, f'no_me_mod_test_opt_log_{now_str}.jb')
    no_me_mod = no_me_mod.fit2(df,checkpoint_filename=write_path, parallel=False, parallel_n_jobs=4)
    
    with open(dump_path, 'wb') as f:
        jb.dump(no_me_mod, f)
else:
    debug_obj_v = '18032025-142607' 
    dump_path = os.path.join(logs_path, f'no_me_mod_test_debug_obj_{debug_obj_v}.jb')
    with open(dump_path, 'rb') as f:
        no_me_mod = jb.load(f)
#%%
coords = {'subject':list(no_me_mod.subject_data['SUBJID'].values), 
          'obs_id': list(no_me_mod.data.index.values)
          }
init_summary = no_me_mod.init_vals_pd.copy()
model_params = init_summary.loc[init_summary['population_coeff'], :]
model_param_dep_vars = init_summary.loc[(init_summary['population_coeff'] == False)
                                             & (init_summary['model_error'] == False), :]
model_error = init_summary.loc[init_summary['model_error'], :]
#%%
best_fit_df = no_me_mod.fit_result_summary_.reset_index().copy()
#%%
pop_coeff_f1 = best_fit_df['population_coeff']

best_model_params = best_fit_df.loc[pop_coeff_f1, 
                                    ['model_coeff', 'best_fit_param_val']]
model_params = model_params.merge(best_model_params, how = 'left', on = 'model_coeff')
model_params['init_val'] = model_params['best_fit_param_val'].copy()
#Assume 20% CV for a nice wide but informative prior 
model_params['sigma'] = np.log(np.exp(model_params['init_val']) * .2)
#%%
b_i_approx = no_me_mod.b_i_approx
b_i_approx.columns = [i[0] for i in b_i_approx.columns.to_flat_index()]
np.exp(np.mean(np.exp(b_i_approx), axis = 1))
best_me_params = best_fit_df.loc[best_fit_df['subject_level_intercept']]
best_me_params = best_me_params.rename(columns = {'best_fit_param_val':'best_fit_param_val_me'})
best_me_params = best_me_params[['model_coeff',
                                 'subject_level_intercept_name',
                                 'best_fit_param_val_me']]
model_params = model_params.merge(best_me_params,
                                  how = 'left',
                                  on = ['model_coeff',
                                        'subject_level_intercept_name'])
model_params['subject_level_intercept_sd_init_val'] = model_params['best_fit_param_val_me'].copy()
#Assume 20% CV for a nice wide but informative prior
tmp_c = 'subject_level_intercept_sd_init_val'
model_params['sigma_subject_level_intercept_sd_init_val'] = np.log(np.exp(model_params[tmp_c]) * .2
                                                                   
                                                                   )
#%%


model_error = best_fit_df.loc[best_fit_df['model_error']
                              , 'best_fit_param_val'].to_numpy()[0]

#find a nice reference for this approxmiation
sigma_log_approx = model_error / np.mean(no_me_mod.data['DV'])

model = make_pymc_model(no_me_mod, no_me_mod.subject_data,
                        no_me_mod.data, model_params,  
                        model_param_dep_vars, model_error = sigma_log_approx,
                        ode_method='scipy')
#%%                       
make_graph_viz = True
if make_graph_viz:
    pm.model_to_graphviz(model)
    
#%%
vars_list = list(model.values_to_rvs.keys())[:-1]

#sampler = "DEMetropolisZ"
chains = 4
tune = 10000
total_draws = 10000
draws = np.round(total_draws/chains, 0).astype(int)
with model:
    trace_DEMZ = pm.sample(step=[pm.DEMetropolisZ(vars_list)], cores = 1, tune = tune, draws = draws, chains = chains,)
    #trace_NUTS = pm.sample(step=[pm.NUTS(vars_list)], cores = 1, tune = tune, draws = draws, chains = chains)
