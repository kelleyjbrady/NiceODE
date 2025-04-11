# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_subject_levels
import joblib as jb
from utils import sum_of_squares_loss, numba_one_compartment_model, PopulationCoeffcient, ODEInitVals, mean_squared_error_loss, huber_loss
import cProfile
from datetime import datetime


# %%
from utils import CompartmentalModel, FOCE_approx_ll_loss, FO_approx_ll_loss
from diffeqs import( OneCompartmentFODiffEq,
                    mm_one_compartment_model,
                    first_order_one_compartment_model,
                    first_order_one_compartment_model2,
                    parallel_elim_one_compartment_model, 
                    one_compartment_absorption
                    )
import numpy as np

# %%
diffeq_obj = OneCompartmentFODiffEq()
pk_model_function = diffeq_obj.diff_eq()

# %%


# %%



now_str = datetime.now().strftime("_%d%m%Y-%H%M%S")
with open(r'/workspaces/PK-Analysis/absorbtion_debug_scale_df.jb', 'rb') as f:
    scale_df = jb.load(f)
#%%
scale_df['dose_ng'] = scale_df['AMT']*1000
scale_df['DV_ng/L'] = scale_df['DV'] * 1000
# %%
me_mod_fo =  CompartmentalModel(
          ode_t0_cols=[ ODEInitVals('DV_ng/L'), ODEInitVals('dose_ng'),],
          population_coeff=[
                            PopulationCoeffcient('ka', .7, 
                                                 subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(.07),
                                                 optimization_upper_bound = np.log(2),
                                                 subject_level_intercept_sd_init_val = 0.2, 
                                                 subject_level_intercept_sd_upper_bound = 20,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('cl',
                                                 15,
                                                  optimization_lower_bound = np.log(5),
                                                 optimization_upper_bound = np.log(25),
                                                subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.3, 
                                                subject_level_intercept_sd_upper_bound = 20,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('vd', 35
                                                , optimization_lower_bound = np.log(25)
                                                , optimization_upper_bound = np.log(35)
                                                ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   no_me_loss_needs_sigma=False,
                                   optimizer_tol=None, 
                                   pk_model_function=one_compartment_absorption, 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.005
                                                                          ,optimization_lower_bound=0.000001
                                                                          ,optimization_upper_bound=.5
                                                                          ),
                                   #ode_solver_method='BDF'
                                   )
#%%
me_mod_fo = me_mod_fo.fit2(scale_df,checkpoint_filename=f'mod_abs_test_me_fo_abs_{now_str}.jb', n_iters_per_checkpoint=1, parallel=False, parallel_n_jobs=4)
scale_df['me_fo_preds'] = me_mod_fo.predict2(scale_df)
stack_cols = ['DV', 'me_fo_preds',]
long_df = scale_df.melt(id_vars = ['SUBJID', 'TIME'], value_vars = stack_cols, value_name='Conc', var_name = 'pred_method')

#%%
b_i_apprx_df = pd.DataFrame( dtype = pd.Float64Dtype())
b_i_apprx_df['b_i_fo_cl'] = me_mod_fo.b_i_approx[('cl', 'omega2_cl')].to_numpy()
b_i_apprx_df['SUBJID'] = scale_df['SUBJID'].drop_duplicates().values
scale_df = (scale_df.merge(b_i_apprx_df, how = 'left', on = 'SUBJID') 
            if 'b_i_fo_cl' not in scale_df.columns else scale_df.copy())

me_mod_foce =  CompartmentalModel(
          ode_t0_cols=[ODEInitVals('DV')],
          population_coeff=[PopulationCoeffcient('cl', 25, subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(15), 
                                                 optimization_upper_bound = np.log(40),
                                                 subject_level_intercept_sd_init_val = 0.38, 
                                                 subject_level_intercept_sd_lower_bound = .001, 
                                                 subject_level_intercept_sd_upper_bound = 2,
                                                 subject_level_intercept_init_vals_column_name='b_i_fo_cl',
                                                 ),
                            PopulationCoeffcient('vd', 80
                                                 , optimization_lower_bound = np.log(70)
                                                 , optimization_upper_bound = np.log(90)
                                                 
                                                 ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   #optimizer_tol=.00001, 
                                   pk_model_function=first_order_one_compartment_model2, 
                                   me_loss_function=FOCE_approx_ll_loss, 
                                   model_error_sigma=PopulationCoeffcient('sigma', .18
                                                                       ,log_transform_init_val=False
                                                                       , optimization_lower_bound=.001, 
                                                                       optimization_upper_bound=4
                                                                       )
                                   #ode_solver_method='BDF'
                                   )



# %%
me_mod_foce.fit2(scale_df,checkpoint_filename=f'mod_abs_test_me_foce_{now_str}.jb', n_iters_per_checkpoint=1, parallel=False, parallel_n_jobs=4)


with open('me_mod_debug_foce.jb', 'wb') as f:
    jb.dump(me_mod_foce, f)

