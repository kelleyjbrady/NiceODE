#%%
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64']='True'

import numpyro
numpyro.set_host_device_count(4)

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jb
from niceode.utils import CompartmentalModel, PopulationCoeffcient, ODEInitVals, neg2_log_likelihood_loss
from niceode.diffeqs import( 
                    first_order_one_compartment_model, #dy/dt = -k * C
                    first_order_one_compartment_model2, #dy/dt = -cl/vd * C
                    OneCompartmentAbsorption
                    )
import numpy as np
from niceode.pymc_utils import make_pymc_model
import pymc as pm
from datetime import datetime

import os
import arviz as az
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

now_str = datetime.now().strftime("%d%m%Y-%H%M%S")

with open(r'/workspaces/PK-Analysis/debug/cp1805_prep.jb', 'rb') as f:
    df = jb.load(f)
base_p = "/workspaces/PK-Analysis/"
logs_path = os.path.join(base_p, 'logs')
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

#%%
me_mod_fo =  CompartmentalModel(
    model_name = "debug_cp1805_abs_kaME-clME-vdME_FO_nodep_dermal",
          ode_t0_cols=[ ODEInitVals('DV_scale'), ODEInitVals('AMT_scale'),],
          conc_at_time_col = 'DV_scale',
          dose_col='AMT_scale',
          subject_id_col = 'ID_x', 
          time_col = 'TIME_x',
          population_coeff=[
                            PopulationCoeffcient('ka', optimization_init_val = .7, 
                                                 subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(1e-6),
                                                 #optimization_upper_bound = np.log(3),
                                                 subject_level_intercept_sd_init_val = 0.2, 
                                                 subject_level_intercept_sd_upper_bound = 20,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('cl',
                                                 optimization_init_val = .1,
                                                  optimization_lower_bound = np.log(1e-4),
                                                  optimization_upper_bound=np.log(1),
                                                 #optimization_upper_bound = np.log(.005),
                                                subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.3, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('vd', optimization_init_val = 1.2
                                                , optimization_lower_bound = np.log(.1)
                                                ,optimization_upper_bound=np.log(5),
                                                subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.3, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                #, optimization_upper_bound = np.log(.05)
                                                ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=neg2_log_likelihood_loss, 
                                   no_me_loss_needs_sigma=True,
                                   optimizer_tol=None, 
                                   pk_model_class=OneCompartmentAbsorption, 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.2
                                                                          ,optimization_lower_bound=0.00001
                                                                          ,optimization_upper_bound=3
                                                                          ),
                                   #ode_solver_method='BDF'
                                   batch_id='mlflow_test_batch9',
                                   minimize_method = 'COBYQA'
                                   )



no_me_mod = me_mod_fo.fit2(df, ci_level = None)
#%%

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
model_params['init_val_log_scale'] = model_params['init_val'].copy()
model_params['init_val_true_scale'] = np.exp(model_params['init_val'])
model_params['init_val_softplus'] = np.log(np.exp(model_params['init_val_true_scale']) - 1)
#%%
#Assume 20% CV for a nice wide but informative prior 
prior_cv = 0.3
softplus = False
if softplus:
    f = model_params['init_val_true_scale'] < 0.5
    model_params.loc[f, 'sigma'] = np.sqrt(np.log( 1 +  prior_cv**2))
    f = model_params['init_val_true_scale'] > 10
    model_params.loc[f, 'sigma'] = model_params.loc[f, 'init_val_true_scale'] * prior_cv
    f1 = model_params['init_val_true_scale'] <= 10
    f2 = model_params['init_val_true_scale'] >= 0.5
    model_params.loc[f1 & f2, 'sigma'] = model_params.loc[f1 & f2, 'init_val_softplus'] * prior_cv
    model_params['init_val'] = model_params['init_val_softplus'].copy()
else:
    model_params['sigma'] = np.sqrt(np.log( 1 +  prior_cv**2))
#%%
b_i_approx = no_me_mod.b_i_approx.copy()
b_i_approx.columns = [i[0] for i in b_i_approx.columns.to_flat_index()]
best_me_params = best_fit_df.loc[best_fit_df['subject_level_intercept']]
best_me_params = best_me_params.rename(columns = {'best_fit_param_val':'best_fit_param_val_me'})
best_me_params = best_me_params[['model_coeff',
                                 'subject_level_intercept_name',
                                 'best_fit_param_val_me']]
model_params = model_params.merge(best_me_params,
                                  how = 'left',
                                  on = ['model_coeff',
                                        'subject_level_intercept_name'])
if softplus:
    f = model_params['init_val_true_scale'] < 0.5
    model_params.loc[f, 'subject_level_intercept_sd_init_val'] = model_params.loc[f, 'best_fit_param_val_me'].copy()
    f = model_params['init_val_true_scale'] > 10
    model_params.loc[f, 'subject_level_intercept_sd_init_val'] = (model_params.loc[f, 'best_fit_param_val_me'] 
                                    * model_params.loc[f, 'init_val_true_scale'])
    f1 = model_params['init_val_true_scale'] <= 10
    f2 = model_params['init_val_true_scale'] >= 0.5
    model_params.loc[f1 & f2, 'subject_level_intercept_sd_init_val'] = (model_params.loc[f1 & f2, 'init_val_softplus'] 
                                          * model_params.loc[f1 & f2, 'best_fit_param_val_me'])
else:
    model_params['subject_level_intercept_sd_init_val'] = model_params['best_fit_param_val_me'].copy()
#Assume 20% CV for a nice wide but informative prior
#tmp_c = 'subject_level_intercept_sd_init_val'
#model_params['sigma_subject_level_intercept_sd_init_val'] = np.log(np.exp(model_params[tmp_c]) * .2
#                                                                   
#                                                                   )
#%%


model_error = best_fit_df.loc[best_fit_df['model_error']
                              , 'best_fit_param_val'].to_numpy()[0]

#find a nice reference for this approxmiationFalse
error_model = 'additive'
if error_model == 'proportional':
    pm_error = model_error / np.mean(no_me_mod.data['DV_scale'])
if error_model == 'additive':
    pm_error = model_error
#%%
model = make_pymc_model(no_me_mod, no_me_mod.subject_data,
                        no_me_mod.data, model_params,  
                        model_param_dep_vars, model_error = pm_error,
                        link_function = 'exp'
                        )
#%%                       
make_graph_viz = True
if make_graph_viz:
    pm.model_to_graphviz(model)
    
#%%
vars_list = list(model.values_to_rvs.keys())[:-1]

#sampler = "DEMetropolisZ"
chains = 4
tune = 2000
total_draws = 6000
draws = np.round(total_draws/chains, 0).astype(int)
with model:
    #trace_DEMZ = pm.sample(step=[pm.DEMetropolisZ(vars_list)], cores = 1, tune = tune, draws = draws, chains = chains,)
    trace_NUTS = pm.sample( tune = tune, draws = draws, chains = chains, nuts_sampler = 'numpyro', target_accept = 0.92 )
    #trace_bj_nuts = pm.sampling.jax.sample_blackjax_nuts(tune = tune,
    #                                                     draws = draws, chains=chains,
    #                                                     chain_method='vectorized')
# %%
with open(f'trace_nuts_{now_str}.jb', 'wb') as f:
    jb.dump(trace_NUTS, f)
# %%
