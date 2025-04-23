# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_subject_levels
import joblib as jb


# %%
from utils import CompartmentalModel, FOCE_approx_ll_loss, FO_approx_ll_loss
from diffeqs import( OneCompartmentFODiffEq,
                    mm_one_compartment_model,
                    first_order_one_compartment_model,
                    first_order_one_compartment_model2,
                    parallel_elim_one_compartment_model, 
                    one_compartment_absorption, 
                    OneCompartmentConc,
                    OneCompartmentBolus_CL
                    )
import numpy as np

# %%
diffeq_obj = OneCompartmentFODiffEq()
pk_model_function = diffeq_obj.diff_eq()

# %%


# %%

from utils import sum_of_squares_loss, numba_one_compartment_model, PopulationCoeffcient, ODEInitVals, mean_squared_error_loss, huber_loss
import cProfile
from datetime import datetime

now_str = datetime.now().strftime("_%d%m%Y-%H%M%S")
#%%
with open(r'/workspaces/PK-Analysis/absorbtion_debug_scale_df.jb', 'rb') as f:
    scale_df = jb.load(f)
#%%
scale_df['dose_ng'] = scale_df['AMT']*1000
scale_df['DV_ng/L'] = (scale_df['DV'] * 1000)
scale_df['dose_scale'] = scale_df['dose_ng'] / 1e5
scale_df['DV_scale'] = scale_df['DV_ng/L'] / 1e5
#scale_df['DV_scale']= scale_df['DV_ng/L']/scale_df['dose_ng'].max()
#scale_df['dose_scale'] = 1.0

zero_out_abs = True
if zero_out_abs:
    dfs = []
    for c in scale_df['SUBJID'].unique():
        work_df = scale_df.loc[scale_df['SUBJID'] == c, :].reset_index(drop = True)
        tmax = work_df.loc[work_df['DV'] == work_df['DV'].max(), 'TIME'].to_numpy()[0]
        gte_max_f = work_df['TIME'] >= tmax
        t0_f = work_df['TIME'] == 0
        work_df = work_df.loc[gte_max_f | t0_f, :]
        #max_idx = work_df.loc[work_df['DV'] == work_df['DV'].max(), :].index[0]
        #work_df = work_df.iloc[max_idx:, :]
        #work_df['TIME'] = work_df['TIME'] - work_df['TIME'].min()
        dfs.append(work_df.copy())
    work_df = pd.concat(dfs)
else:
    work_df = scale_df.copy()

work_df.loc[work_df['TIME'] == 0, 'solve_ode_at_TIME'] = False
work_df.loc[work_df['TIME'] > 0, 'solve_ode_at_TIME'] = True

scale_df = work_df.copy()
pred_df = scale_df.copy()
pred_df['solve_ode_at_TIME'] = True


piv_cols = []
res_df = pd.DataFrame()
#%%
no_me_mod_fo =  CompartmentalModel(
          ode_t0_cols=[ODEInitVals('dose_scale')],
          conc_at_time_col = 'DV_scale',
          solve_ode_at_time_col = 'solve_ode_at_TIME',
          population_coeff=[PopulationCoeffcient('cl', 18, subject_level_intercept=False,
                                                 #subject_level_intercept_sd_init_val = 0.2, 
                                                 #subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('vd', 40, ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   no_me_loss_needs_sigma=False,
                                   #optimizer_tol=.001, 
                                   pk_model_class=OneCompartmentBolus_CL(), 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.5
                                                                          ,optimization_lower_bound=0.000001
                                                                          ,optimization_upper_bound=5
                                                                          ),
                                   #ode_solver_method='BDF'
                                  # optimizer_tol = .1
                                  #minimize_method = 'COBYQA'
                                   )
res_df['DV_scale'] = scale_df['DV_scale']
#%%
no_me_mod_fo = no_me_mod_fo.fit2(scale_df,)
res_df['no_me_fo_sse'] = no_me_mod_fo.predict2(pred_df)
piv_cols.extend(['DV_scale', 'no_me_fo_sse'])
# %%
me_mod_fo =  CompartmentalModel(
          ode_t0_cols=[ODEInitVals('DV')],
          conc_at_time_col = 'DV',
          population_coeff=[PopulationCoeffcient('cl', 18, subject_level_intercept=True,
                                                 subject_level_intercept_sd_init_val = 0.2, 
                                                 subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('vd', 30, 
                                                 
                                                 ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   no_me_loss_needs_sigma=False,
                                   #optimizer_tol=.001, 
                                   pk_model_function=first_order_one_compartment_model2, 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.2
                                                                          ,optimization_lower_bound=0.000001
                                                                          ,optimization_upper_bound=5
                                                                          ),
                                   #ode_solver_method='BDF'
                                  # optimizer_tol = .1
                                  #minimize_method = 'COBYQA'
                                   )



me_mod_fo = me_mod_fo.fit2(scale_df,checkpoint_filename=f'mod_abs_test_me_fo{now_str}.jb', n_iters_per_checkpoint=1, parallel=False, parallel_n_jobs=4)

res_df['me_fo_preds'] = me_mod_fo.predict2(scale_df)
piv_cols.extend(['me_fo_preds', ])
# %%
me_mod_fo_q =  CompartmentalModel(
          ode_t0_cols=[ODEInitVals('DV')],
          conc_at_time_col = 'DV',
          population_coeff=[PopulationCoeffcient('cl', 18, subject_level_intercept=True,
                                                 subject_level_intercept_sd_init_val = 0.2, 
                                                 subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('vd', 30, 
                                                 
                                                 ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   no_me_loss_needs_sigma=False,
                                   #optimizer_tol=.001, 
                                   pk_model_function=first_order_one_compartment_model2, 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.2
                                                                          ,optimization_lower_bound=0.000001
                                                                          ,optimization_upper_bound=5
                                                                          ),
                                   #ode_solver_method='BDF'
                                  # optimizer_tol = .1
                                  minimize_method = 'COBYQA'
                                   )



me_mod_fo_q = me_mod_fo_q.fit2(scale_df,checkpoint_filename=f'mod_abs_test_me_fo{now_str}.jb', n_iters_per_checkpoint=1, parallel=False, parallel_n_jobs=4)
res_df['me_fo_preds'] = me_mod_fo_q.predict2(scale_df)
piv_cols.extend(['me_fo_preds', ])
#%%
test_df = scale_df.drop(columns = 'DV').copy()
me_mod_fo_ln_q =  CompartmentalModel(
          ode_t0_cols=[ODEInitVals('DV_ln')],
          conc_at_time_col = 'DV_ln',
          population_coeff=[PopulationCoeffcient('cl', 18, subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(5), 
                                                 optimization_upper_bound = np.log(30),
                                                 subject_level_intercept_sd_init_val = 0.2, 
                                                 subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('vd', 20, 
                                                 optimization_lower_bound = np.log(30), 
                                                 optimization_upper_bound = np.log(55),
                                                 ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   no_me_loss_needs_sigma=False,
                                   #optimizer_tol=.001, 
                                   pk_model_function=first_order_one_compartment_model2, 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.2
                                                                          ,optimization_lower_bound=0.000001
                                                                          ,optimization_upper_bound=5
                                                                          ),
                                   #ode_solver_method='BDF'
                                  # optimizer_tol = .1
                                  minimize_method = 'COBYQA'
                                   )



#me_mod_fo_ln_q = me_mod_fo_ln_q.fit2(test_df,checkpoint_filename=f'mod_abs_test_me_fo{now_str}.jb', n_iters_per_checkpoint=1, parallel=False, parallel_n_jobs=4)
#res_df['me_fo_preds_ln_cobyqa'] = me_mod_fo_ln_q.predict2(test_df)
#res_df['DV_ln'] = test_df['DV_ln']
#piv_cols.extend(['me_fo_preds_ln_cobyqa', 'DV_ln'])
#%%
me_mod_fo_ln =  CompartmentalModel(
          ode_t0_cols=[ODEInitVals('DV_ln')],
          conc_at_time_col = 'DV_ln',
          population_coeff=[PopulationCoeffcient('cl', 18, subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(5), 
                                                 optimization_upper_bound = np.log(30),
                                                 subject_level_intercept_sd_init_val = 0.2, 
                                                 subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('vd', 20, 
                                                 #optimization_upper_bound=np.log(60),
                                                 optimization_lower_bound = np.log(30), 
                                                 optimization_upper_bound = np.log(55),
                                                 ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   no_me_loss_needs_sigma=False,
                                   #optimizer_tol=.001, 
                                   pk_model_function=first_order_one_compartment_model2, 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.5
                                                                          ,optimization_lower_bound=0.000001
                                                                          ,optimization_upper_bound=5
                                                                          ),
                                   #ode_solver_method='BDF'
                                  # optimizer_tol = .1
                                  #minimize_method = 'COBYQA'
                                   )



#me_mod_fo_ln = me_mod_fo_ln.fit2(scale_df,checkpoint_filename=f'mod_abs_test_me_fo{now_str}.jb', n_iters_per_checkpoint=1, parallel=False, parallel_n_jobs=4)
#res_df['me_fo_preds_ln'] = me_mod_fo_ln.predict2(scale_df)
#piv_cols.extend(['me_fo_preds_ln', ])
#%%
me_mod_fo_scale =  CompartmentalModel(
          ode_t0_cols=[ODEInitVals('DV_scale')],
          conc_at_time_col = 'DV_scale',
          population_coeff=[PopulationCoeffcient('cl', 18, subject_level_intercept=True,
                                                 subject_level_intercept_sd_init_val = 0.2, 
                                                 subject_level_intercept_sd_lower_bound=1e-6, 
                                                 #subject_level_intercept_sd_upper_bound=1
                                                 ),
                            PopulationCoeffcient('vd', 30, ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   no_me_loss_needs_sigma=False,
                                   optimizer_tol=None, 
                                   pk_model_function=first_order_one_compartment_model2, 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.5
                                                                          ,optimization_lower_bound=0.000001
                                                                          ,optimization_upper_bound=5
                                                                          ),
                                   #ode_solver_method='BDF'
                                   minimize_method = 'COBYQA'
                                   )



#me_mod_fo_scale = me_mod_fo_scale.fit2(scale_df,checkpoint_filename=f'mod_abs_test_me_fo{now_str}.jb', n_iters_per_checkpoint=1, parallel=False, parallel_n_jobs=4)
#scale_df['me_fo_preds_alt_scale'] = me_mod_fo_scale.predict2(scale_df)


#%%
b_i_apprx_df = pd.DataFrame( dtype = pd.Float64Dtype())
b_i_apprx_df['b_i_fo_cl'] = me_mod_fo.b_i_approx[('cl', 'omega2_cl')].to_numpy()
b_i_apprx_df['SUBJID'] = scale_df['SUBJID'].drop_duplicates().values
scale_df = (scale_df.merge(b_i_apprx_df, how = 'left', on = 'SUBJID') 
            if 'b_i_fo_cl' not in scale_df.columns else scale_df.copy())

me_mod_foce =  CompartmentalModel(
          ode_t0_cols=[ODEInitVals('DV')],
          conc_at_time_col = 'DV',
          population_coeff=[PopulationCoeffcient('cl', 15, subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(10), 
                                                 optimization_upper_bound = np.log(20),
                                                 subject_level_intercept_sd_init_val = 0.38, 
                                                 subject_level_intercept_sd_lower_bound = .001, 
                                                 subject_level_intercept_sd_upper_bound = 2,
                                                 subject_level_intercept_init_vals_column_name='b_i_fo_cl',
                                                 ),
                            PopulationCoeffcient('vd', 40
                                                 , optimization_lower_bound = np.log(30)
                                                 , optimization_upper_bound = np.log(60)
                                                 
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
                                                                       ),
                                   #ode_solver_method='BDF', 
                                   #minimize_method = 'COBYQA'
                                   )



me_mod_foce.fit2(scale_df,checkpoint_filename=f'mod_abs_test_me_foce_{now_str}.jb', n_iters_per_checkpoint=1, parallel=False, parallel_n_jobs=4)
res_df['me_foce_preds'] = me_mod_foce.predict2(scale_df)
piv_cols.append('me_foce_preds')
#stack_cols = ['DV', 'DV_ln', 'DV_scale', 'me_fo_preds', 'me_foce_preds', 'me_fo_preds_alt_scale', 'me_fo_preds_ln']
#long_df = scale_df.melt(id_vars = ['SUBJID', 'TIME'], value_vars = piv_cols, value_name='Conc', var_name = 'pred_method')
#%%
me_mod_foce_q =  CompartmentalModel(
          ode_t0_cols=[ODEInitVals('DV')],
          conc_at_time_col = 'DV',
          population_coeff=[PopulationCoeffcient('cl', 15, subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(10), 
                                                 optimization_upper_bound = np.log(20),
                                                 subject_level_intercept_sd_init_val = 0.38, 
                                                 subject_level_intercept_sd_lower_bound = .001, 
                                                 subject_level_intercept_sd_upper_bound = 2,
                                                 subject_level_intercept_init_vals_column_name='b_i_fo_cl',
                                                 ),
                            PopulationCoeffcient('vd', 40
                                                 , optimization_lower_bound = np.log(30)
                                                 , optimization_upper_bound = np.log(60)
                                                 
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
                                                                       ),
                                   #ode_solver_method='BDF', 
                                   minimize_method = 'COBYQA'
                                   )



me_mod_foce_q.fit2(scale_df,checkpoint_filename=f'mod_abs_test_me_foce_{now_str}.jb', n_iters_per_checkpoint=1, parallel=False, parallel_n_jobs=4)
res_df['me_foce_preds'] = me_mod_foce_q.predict2(scale_df)
piv_cols.append('me_foce_preds')


#%%

me_mod_foce_ln =  CompartmentalModel(
          ode_t0_cols=[ODEInitVals('DV_ln')],
          conc_at_time_col = 'DV_ln',
          population_coeff=[PopulationCoeffcient('cl', 15, subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(5), 
                                                 optimization_upper_bound = np.log(25),
                                                 subject_level_intercept_sd_init_val = 0.2, 
                                                 subject_level_intercept_sd_lower_bound = .001, 
                                                 subject_level_intercept_sd_upper_bound = 2,
                                                 subject_level_intercept_init_vals_column_name='b_i_fo_cl',
                                                 ),
                            PopulationCoeffcient('vd', 40
                                                 , optimization_lower_bound = np.log(30)
                                                 , optimization_upper_bound = np.log(50)
                                                 
                                                 ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   #optimizer_tol=.00001, 
                                   pk_model_function=first_order_one_compartment_model2, 
                                   me_loss_function=FOCE_approx_ll_loss, 
                                   model_error_sigma=PopulationCoeffcient('sigma', .4
                                                                       ,log_transform_init_val=False
                                                                       , optimization_lower_bound=.001, 
                                                                       optimization_upper_bound=1
                                                                       ),
                                   #ode_solver_method='BDF', 
                                   #minimize_method = 'COBYQA'
                                   )



#me_mod_foce_ln.fit2(scale_df,checkpoint_filename=f'mod_abs_test_me_foce_{now_str}.jb', n_iters_per_checkpoint=1, parallel=False, parallel_n_jobs=4)
#res_df['me_foce_preds_ln'] = me_mod_foce_ln.predict2(scale_df)
#piv_cols.append('me_foce_preds_ln')

#with open('me_mod_debug_foce.jb', 'wb') as f:
#    jb.dump(me_mod_foce, f)

#%%

me_mod_foce_ln_q =  CompartmentalModel(
          ode_t0_cols=[ODEInitVals('DV_ln')],
          conc_at_time_col = 'DV_ln',
          population_coeff=[PopulationCoeffcient('cl', 15, subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(10), 
                                                 optimization_upper_bound = np.log(60),
                                                 subject_level_intercept_sd_init_val = 0.38, 
                                                 subject_level_intercept_sd_lower_bound = .001, 
                                                 subject_level_intercept_sd_upper_bound = 2,
                                                 subject_level_intercept_init_vals_column_name='b_i_fo_cl',
                                                 ),
                            PopulationCoeffcient('vd', 40
                                                 , optimization_lower_bound = np.log(30)
                                                 , optimization_upper_bound = np.log(60)
                                                 
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
                                                                       optimization_upper_bound=1
                                                                       ),
                                   #ode_solver_method='BDF', 
                                   minimize_method = 'COBYQA'
                                   )



#me_mod_foce_ln_q.fit2(scale_df,checkpoint_filename=f'mod_abs_test_me_foce_{now_str}.jb', n_iters_per_checkpoint=1, parallel=False, parallel_n_jobs=4)
#res_df['me_foce_preds_ln_q'] = me_mod_foce_ln_q.predict2(scale_df)
#piv_cols.append('me_foce_preds_ln_q')


import plotly.express as px
res_df[['SUBJID', 'TIME']] = scale_df[['SUBJID', 'TIME']].values
long_df = res_df.melt(id_vars = ['SUBJID', 'TIME'], value_vars = piv_cols, value_name='Conc', var_name = 'pred_method')
px.line(data_frame=long_df, x = 'TIME', y = 'Conc', color = 'pred_method', line_group='SUBJID', animation_frame='SUBJID')

# %%
