# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_subject_levels
import joblib as jb


# %%
from utils import CompartmentalModel, FOCE_approx_ll_loss, FO_approx_ll_loss, neg2_log_likelihood_loss
from diffeqs import( OneCompartmentFODiffEq,
                    mm_one_compartment_model,
                    first_order_one_compartment_model,
                    first_order_one_compartment_model2,
                    parallel_elim_one_compartment_model, 
                    one_compartment_absorption,
                    one_compartment_absorption2, 
                    TwoCompartmentBolus
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
    
scale_df = work_df.copy()

work_df.loc[work_df['TIME'] == 0, 'solve_ode_at_TIME'] = False
work_df.loc[work_df['TIME'] > 0, 'solve_ode_at_TIME'] = True

piv_cols = []
res_df = pd.DataFrame()

piv_cols.append('DV_scale')

scale_df = work_df.copy()
scale_df['c2_init'] = 0.0
pred_df = scale_df.copy()
#pred_df = pred_df.loc[pred_df['solve_ode_at_TIME'], :].copy()
pred_df['solve_ode_at_TIME'] = True
res_df[['SUBJID', 'TIME',  'DV_scale']] = pred_df[['SUBJID', 'TIME', 'DV_scale']].copy()

# %%
me_mod_fo =  CompartmentalModel(
    model_name = "debug_hydrocortisone_2cmptbolus_cl-v1ME-q-v2_fo_3",
          ode_t0_cols=[ ODEInitVals('dose_scale'), ODEInitVals('c2_init'),],
          conc_at_time_col = 'DV_scale',
          solve_ode_at_time_col = 'solve_ode_at_TIME',
          population_coeff=[
                            PopulationCoeffcient('cl', 14, 
                                                 #subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(1e-6),
                                                 optimization_upper_bound = np.log(40),
                                                 #subject_level_intercept_sd_init_val = 0.2, 
                                                 #subject_level_intercept_sd_upper_bound = 20,
                                                #subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('v1',
                                                 40,
                                                  optimization_lower_bound = np.log(1e-6),
                                                 #optimization_upper_bound = np.log(45),
                                                subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.3, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('q', 40
                                                , optimization_lower_bound = np.log(1e-6)
                                                , optimization_upper_bound = np.log(80)
                                                ),
                            PopulationCoeffcient('v2', 3
                                                , optimization_lower_bound = np.log(1e-6)
                                                , optimization_upper_bound = np.log(50)
                                                ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   no_me_loss_needs_sigma=False,
                                   optimizer_tol=None, 
                                   pk_model_class=TwoCompartmentBolus(), 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.2
                                                                          ,optimization_lower_bound=0.00001
                                                                          ,optimization_upper_bound=1
                                                                          ),
                                   #ode_solver_method='BDF'
                                   minimize_method = 'COBYQA',
                                   )
fit_model = True
if fit_model:
    me_mod_fo = me_mod_fo.fit2(scale_df, )
else:
    with open(f"logs/fitted_model_{me_mod_fo.model_name}.jb", 'rb') as f:
        me_mod_fo = jb.load(f)
#Something about the way predict2 works in the context of this ME model makes it such 
#that the preds output by predict2 do not match those generated during fitting, possibly only 
#for ME models in this context, but other context should be checked. Specifically here, the neg2ll for this 
#me model is lower than the neg2ll for the non mixed effects model below, but the preds generated 
#by this model are much worse than the model below. 
#%%
res_df[me_mod_fo.model_name] = me_mod_fo.predict2(pred_df, )
res_df[f"avg_effect_{me_mod_fo.model_name}"] = me_mod_fo.predict2(pred_df, subject_level_prediction = False, )
piv_cols.extend([me_mod_fo.model_name, f"avg_effect_{me_mod_fo.model_name}"])
me_mod_fo.save_fitted_model(jb_file_name = me_mod_fo.model_name)
#%%
me_mod_fo2 =  CompartmentalModel(
    model_name = "debug_hydrocortisone_2cmptbolus_cl-v1-q-v2_sse",
          ode_t0_cols=[ ODEInitVals('dose_scale'), ODEInitVals('c2_init'),],
          conc_at_time_col = 'DV_scale',
          solve_ode_at_time_col = 'solve_ode_at_TIME',
          population_coeff=[
                                                     PopulationCoeffcient('cl', 14, 
                                                 #subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(1e-6),
                                                 #optimization_upper_bound = np.log(20),
                                                 #subject_level_intercept_sd_init_val = 0.2, 
                                                 #subject_level_intercept_sd_upper_bound = 20,
                                                #subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('v1',
                                                 40,
                                                  optimization_lower_bound = np.log(1e-6),
                                                 #optimization_upper_bound = np.log(50),
                                                #subject_level_intercept=True, 
                                                #subject_level_intercept_sd_init_val = 0.3, 
                                                #subject_level_intercept_sd_upper_bound = 5,
                                                #subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('q', 14,
                                                optimization_lower_bound = np.log(1e-6),
                                                #, optimization_upper_bound = np.log(30)
                                                ),
                            PopulationCoeffcient('v2', 3,
                                                optimization_lower_bound = np.log(1e-6),
                                                #, optimization_upper_bound = np.log(50)
                                                ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   no_me_loss_needs_sigma=False,
                                   optimizer_tol=None, 
                                   pk_model_class=TwoCompartmentBolus(), 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.5
                                                                          ,optimization_lower_bound=0.00001
                                                                          ,optimization_upper_bound=1
                                                                          ),
                                   #ode_solver_method='BDF'
                                   minimize_method = 'COBYQA',
                                   #ode_solver_method = 'Radau'
                                   )
#%%
fits = []
fit_res_dfs = []
for sub in scale_df['SUBJID'].unique():
    fit_df = scale_df.loc[scale_df['SUBJID'] == sub, :].copy()
    fit = me_mod_fo2.fit2(fit_df,)
    inner_pred_df = pred_df.loc[scale_df['SUBJID'] == sub, :]
    fit_df['pred_y'] = fit.predict2(inner_pred_df)
    fits.append(fit.fit_result_)
    fit_res_dfs.append(fit_df.copy())
fit_res_df = pd.concat(fit_res_dfs)
fit_readout = [np.exp(i['x']) for i in fits]
res_df['indiv_fit_preds'] = fit_res_df['pred_y'].copy()
piv_cols.append('indiv_fit_preds')

#%%
fit_model = True
if fit_model:
    me_mod_fo2 = me_mod_fo2.fit2(scale_df, )
else:
    with open(f"logs/fitted_model_{me_mod_fo2.model_name}.jb", 'rb') as f:
        me_mod_fo2 = jb.load(f)
#issue is that the pred_df does not have the correct value in the AMT col for the initial condition
#need to pass the same scale_df and instead alter how res_df is constructed so that it has the t0 
#possibly make the predict function always start at zero for the solving or let tspan be passed to the 
#predict function, which would be even better. 
res_df[me_mod_fo2.model_name] = me_mod_fo2.predict2(pred_df)
piv_cols.append(me_mod_fo2.model_name)
me_mod_fo2.save_fitted_model(jb_file_name = me_mod_fo2.model_name)
#%%

me_mod_fo2 =  CompartmentalModel(
    model_name = "debug_hydrocortisone_2cmptbolus_cl-v1-q-v2_n2ll",
          ode_t0_cols=[ ODEInitVals('dose_scale'), ODEInitVals('c2_init'),],
          conc_at_time_col = 'DV_scale',
          solve_ode_at_time_col = 'solve_ode_at_TIME',
          population_coeff=[
                                                     PopulationCoeffcient('cl', 14, 
                                                 #subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(1e-6),
                                                 #optimization_upper_bound = np.log(20),
                                                 #subject_level_intercept_sd_init_val = 0.2, 
                                                 #subject_level_intercept_sd_upper_bound = 20,
                                                #subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('v1',
                                                 40,
                                                  optimization_lower_bound = np.log(1e-6),
                                                 #optimization_upper_bound = np.log(50),
                                                #subject_level_intercept=True, 
                                                #subject_level_intercept_sd_init_val = 0.3, 
                                                #subject_level_intercept_sd_upper_bound = 5,
                                                #subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('q', 14,
                                                optimization_lower_bound = np.log(1e-6),
                                                #, optimization_upper_bound = np.log(30)
                                                ),
                            PopulationCoeffcient('v2', 3,
                                                optimization_lower_bound = np.log(1e-6),
                                                #, optimization_upper_bound = np.log(50)
                                                ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=neg2_log_likelihood_loss, 
                                   no_me_loss_needs_sigma=True,
                                   optimizer_tol=None, 
                                   pk_model_class=TwoCompartmentBolus(), 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.2
                                                                          ,optimization_lower_bound=0.00001
                                                                          ,optimization_upper_bound=1
                                                                          ),
                                   #ode_solver_method='BDF'
                                   minimize_method = 'COBYQA',
                                   #ode_solver_method = 'Radau'
                                   )

#%%
fit_model = True
if fit_model:
    me_mod_fo2 = me_mod_fo2.fit2(scale_df, )
else:
    with open(f"logs/fitted_model_{me_mod_fo2.model_name}.jb", 'rb') as f:
        me_mod_fo2 = jb.load(f)
#issue is that the pred_df does not have the correct value in the AMT col for the initial condition
#need to pass the same scale_df and instead alter how res_df is constructed so that it has the t0 
#possibly make the predict function always start at zero for the solving or let tspan be passed to the 
#predict function, which would be even better. 
res_df[me_mod_fo2.model_name] = me_mod_fo2.predict2(pred_df)
piv_cols.append(me_mod_fo2.model_name)
me_mod_fo2.save_fitted_model(jb_file_name = me_mod_fo2.model_name)

#%%

me4_mod_fo =  CompartmentalModel(
    model_name = "debug_hydrocortisone_2cmptbolus_clME-v1ME-qME-v2ME_fo_2",
          ode_t0_cols=[ ODEInitVals('dose_scale'), ODEInitVals('c2_init'),],
          conc_at_time_col = 'DV_scale',
          solve_ode_at_time_col = 'solve_ode_at_TIME',
          population_coeff=[
                            PopulationCoeffcient('cl', 14, 
                                                 #subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(1e-6),
                                                 optimization_upper_bound = np.log(40),
                                                 subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.2, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('v1',
                                                 40,
                                                  optimization_lower_bound = np.log(1e-6),
                                                 #optimization_upper_bound = np.log(60),
                                                subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.3, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('q', 14,
                                                 optimization_lower_bound = np.log(1e-6),
                                                 optimization_upper_bound = np.log(80),
                                                 subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.35, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                ),
                            PopulationCoeffcient('v2', 3,
                                                 optimization_lower_bound = np.log(1e-6),
                                                 optimization_upper_bound = np.log(50),
                                                 subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.1, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   no_me_loss_needs_sigma=False,
                                   optimizer_tol=None, 
                                   pk_model_class=TwoCompartmentBolus(), 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.2
                                                                          ,optimization_lower_bound=0.00001
                                                                          ,optimization_upper_bound=1
                                                                          ),
                                   #ode_solver_method='BDF'
                                   minimize_method = 'COBYQA',
                                   )
fit_model = True
if fit_model:
    me4_mod_fo = me4_mod_fo.fit2(scale_df, )
else:
    with open(f"logs/fitted_model_{me4_mod_fo.model_name}.jb", 'rb') as f:
        me4_mod_fo = jb.load(f)
#issue is that the pred_df does not have the correct value in the AMT col for the initial condition
#need to pass the same scale_df and instead alter how res_df is constructed so that it has the t0 
#possibly make the predict function always start at zero for the solving or let tspan be passed to the 
#predict function, which would be even better. 
res_df[me4_mod_fo.model_name] = me4_mod_fo.predict2(pred_df)
piv_cols.append(me4_mod_fo.model_name)
me4_mod_fo.save_fitted_model(jb_file_name = me4_mod_fo.model_name)
#%%
me3_mod_fo =  CompartmentalModel(
    model_name = "debug_hydrocortisone_2cmptbolus_clME-v1ME-qME-v2_fo_2",
          ode_t0_cols=[ ODEInitVals('dose_scale'), ODEInitVals('c2_init'),],
          conc_at_time_col = 'DV_scale',
          solve_ode_at_time_col = 'solve_ode_at_TIME',
          population_coeff=[
                            PopulationCoeffcient('cl', 14, 
                                                 #subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(1e-6),
                                                 optimization_upper_bound = np.log(40),
                                                 subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.2, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('v1',
                                                 40,
                                                  optimization_lower_bound = np.log(1e-6),
                                                 #optimization_upper_bound = np.log(60),
                                                subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.3, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('q', 14,
                                                 optimization_lower_bound = np.log(1e-6),
                                                 optimization_upper_bound = np.log(80),
                                                 subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.35, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                ),
                            PopulationCoeffcient('v2', 3,
                                                 optimization_lower_bound = np.log(1e-6),
                                                 optimization_upper_bound = np.log(50),
                                                 #subject_level_intercept=True, 
                                                #subject_level_intercept_sd_init_val = 0.1, 
                                                #subject_level_intercept_sd_upper_bound = 5,
                                                #subject_level_intercept_sd_lower_bound=1e-6
                                                ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   no_me_loss_needs_sigma=False,
                                   optimizer_tol=None, 
                                   pk_model_class=TwoCompartmentBolus(), 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.2
                                                                          ,optimization_lower_bound=0.00001
                                                                          ,optimization_upper_bound=1
                                                                          ),
                                   #ode_solver_method='BDF'
                                   minimize_method = 'COBYQA',
                                   )
fit_model = True
if fit_model:
    me3_mod_fo = me3_mod_fo.fit2(scale_df, )
else:
    with open(f"logs/fitted_model_{me3_mod_fo.model_name}.jb", 'rb') as f:
        me3_mod_fo = jb.load(f)
#issue is that the pred_df does not have the correct value in the AMT col for the initial condition
#need to pass the same scale_df and instead alter how res_df is constructed so that it has the t0 
#possibly make the predict function always start at zero for the solving or let tspan be passed to the 
#predict function, which would be even better. 
res_df[me3_mod_fo.model_name] = me3_mod_fo.predict2(pred_df)
piv_cols.append(me3_mod_fo.model_name)
me3_mod_fo.save_fitted_model(jb_file_name = me3_mod_fo.model_name)
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

