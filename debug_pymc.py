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

with open(r'/workspaces/miniconda/PK-Analysis/debug_scale_df.jb', 'rb') as f:
    df = jb.load(f)
    
    
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

coords = {'subject':list(no_me_mod.subject_data['SUBJID'].values), 
          'obs_id': list(no_me_mod.data.index.values)
          }
init_summary = no_me_mod.init_vals_pd.copy()
model_params = init_summary.loc[init_summary['population_coeff'], :]
model_param_dep_vars = init_summary.loc[init_summary['population_coeff'] == False, :]

best_fit_df = no_me_mod.fit_result_summary_.reset_index().rename(columns = {'index':'model_coeff', 0:'best_fit_param_val'})
model_params = model_params.merge(best_fit_df, how = 'left', on = 'model_coeff')
model_params['init_val'] = model_params['best_fit_param_val'].copy()
model_params['sigma'] = model_params['init_val'] * .05
model_error = best_fit_df.loc[best_fit_df['model_coeff'] == 'sigma2'
                              , 'best_fit_param_val'].to_numpy()[0]


model = make_pymc_model(no_me_mod.subject_data,
                        no_me_mod.data, model_params,  
                        model_param_dep_vars, model_error = model_error,)