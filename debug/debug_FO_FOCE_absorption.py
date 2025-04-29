# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from niceode.utils import plot_subject_levels
import joblib as jb
from niceode.utils import sum_of_squares_loss, numba_one_compartment_model, PopulationCoeffcient, ODEInitVals, mean_squared_error_loss, huber_loss
import cProfile
from datetime import datetime
import uuid


# %%
from niceode.utils import CompartmentalModel, FOCE_approx_ll_loss, FO_approx_ll_loss
from niceode.diffeqs import( OneCompartmentFODiffEq,
                    mm_one_compartment_model,
                    first_order_one_compartment_model,
                    first_order_one_compartment_model2,
                    parallel_elim_one_compartment_model, 
                    one_compartment_absorption, 
                    one_compartment_absorption2, 
                    OneCompartmentAbsorption, 
                    OneCompartmentAbsorption2
                    )
import numpy as np

# %%
diffeq_obj = OneCompartmentFODiffEq()
pk_model_function = diffeq_obj.diff_eq()

# %%


# %%



#now_str = datetime.now().strftime("_%d%m%Y-%H%M%S")
batch_id = uuid.uuid4()
with open(r'/workspaces/PK-Analysis/data/absorbtion_debug_scale_df.jb', 'rb') as f:
    scale_df = jb.load(f)
#%%
scale_df['dose_ng'] = scale_df['AMT']*1000
scale_df['DV_ng/L'] = (scale_df['DV'] * 1000)
scale_df['dose_scale'] = scale_df['dose_ng'] / 1e5
scale_df['DV_scale'] = scale_df['DV_ng/L'] / 1e5
scale_df['solve_ode_at_TIME'] = True
#scale_df['DV_scale']= scale_df['DV_ng/L']/scale_df['dose_ng'].max()
#scale_df['dose_scale'] = 1.0
piv_cols = []
res_df = pd.DataFrame()
res_df[['SUBJID', 'TIME',  'DV_scale']] = scale_df[['SUBJID', 'TIME', 'DV_scale']].copy()
piv_cols.append('DV_scale')

# %%
me_mod_fo =  CompartmentalModel(
    model_name = "debug_hydrocortisone_abs_ka-clME-vd_fo_2",
          ode_t0_cols=[ ODEInitVals('DV_scale'), ODEInitVals('dose_scale'),],
          conc_at_time_col = 'DV_scale',
          population_coeff=[
                            PopulationCoeffcient('ka', 2, 
                                                 #subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(.1),
                                                 optimization_upper_bound = np.log(3),
                                                 #subject_level_intercept_sd_init_val = 0.2, 
                                                 #subject_level_intercept_sd_upper_bound = 20,
                                                #subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('cl',
                                                 14,
                                                  optimization_lower_bound = np.log(5),
                                                 optimization_upper_bound = np.log(25),
                                                subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.3, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('vd', 40
                                                , optimization_lower_bound = np.log(30)
                                                , optimization_upper_bound = np.log(50)
                                                ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   no_me_loss_needs_sigma=False,
                                   optimizer_tol=None, 
                                   pk_model_class=OneCompartmentAbsorption(), 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.5
                                                                          ,optimization_lower_bound=0.00001
                                                                          ,optimization_upper_bound=1
                                                                          ),
                                   #ode_solver_method='BDF'
                                   )
fit_model = True
if fit_model:
    me_mod_fo = me_mod_fo.fit2(scale_df, )
else:
    with open(f"logs/fitted_model_{me_mod_fo.model_name}.jb", 'rb') as f:
        me_mod_fo = jb.load(f)
res_df[me_mod_fo.model_name] = me_mod_fo.predict2(scale_df)
piv_cols.append(me_mod_fo.model_name)
me_mod_fo.save_fitted_model(jb_file_name = me_mod_fo.model_name)

#%%
me2_mod_fo =  CompartmentalModel(
    model_name = "debug_hydrocortisone_abs_kaME-clME-vd_fo_2",
          ode_t0_cols=[ ODEInitVals('DV_scale'), ODEInitVals('dose_scale'),],
          conc_at_time_col = 'DV_scale',
          population_coeff=[
                            PopulationCoeffcient('ka', 2, 
                                                 subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(.1),
                                                 optimization_upper_bound = np.log(3),
                                                 subject_level_intercept_sd_init_val = 0.2, 
                                                 subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('cl',
                                                 14,
                                                  optimization_lower_bound = np.log(5),
                                                 optimization_upper_bound = np.log(25),
                                                subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.3, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('vd', 40
                                                , optimization_lower_bound = np.log(30)
                                                , optimization_upper_bound = np.log(50)
                                                ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   no_me_loss_needs_sigma=False,
                                   optimizer_tol=None, 
                                   pk_model_class=OneCompartmentAbsorption(), 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.5
                                                                          ,optimization_lower_bound=0.00001
                                                                          ,optimization_upper_bound=1
                                                                          ),
                                   #ode_solver_method='BDF'
                                   )

fit_model = True
if fit_model:
    me2_mod_fo = me2_mod_fo.fit2(scale_df, )
else:
    with open(f"logs/fitted_model_{me2_mod_fo.model_name}.jb", 'rb') as f:
        me2_mod_fo = jb.load(f)
res_df[me2_mod_fo.model_name] = me2_mod_fo.predict2(scale_df)
piv_cols.append(me2_mod_fo.model_name)
me2_mod_fo.save_fitted_model(jb_file_name = me2_mod_fo.model_name)


#%%
me_mod_fo2 =  CompartmentalModel(
    model_name = "debug_hydrocortisone_abs_ka-ke-vd_sse_2",
          ode_t0_cols=[ ODEInitVals('DV_scale'), ODEInitVals('dose_scale'),],
          conc_at_time_col = 'DV_scale',
          population_coeff=[
                            PopulationCoeffcient('ka', 1.4, 
                                                 #subject_level_intercept=True,
                                                 #optimization_lower_bound = np.log(.5),
                                                 #optimization_upper_bound = np.log(3),
                                                 #subject_level_intercept_sd_init_val = 0.2, 
                                                 #subject_level_intercept_sd_upper_bound = 20,
                                                #subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('ke',
                                                 .4,
                                                  #optimization_lower_bound = np.log(.01),
                                                 #optimization_upper_bound = np.log(5),
                                                #subject_level_intercept=True, 
                                                #subject_level_intercept_sd_init_val = 0.08, 
                                                #subject_level_intercept_sd_upper_bound = 5,
                                                #subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('vd',
                                                 35,
                                                  #optimization_lower_bound = np.log(.01),
                                                 #optimization_upper_bound = np.log(5),
                                                #subject_level_intercept=True, 
                                                #subject_level_intercept_sd_init_val = 0.08, 
                                                #subject_level_intercept_sd_upper_bound = 5,
                                                #subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   no_me_loss_needs_sigma=False,
                                   optimizer_tol=None, 
                                   pk_model_class=OneCompartmentAbsorption2(), 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.5
                                                                          ,optimization_lower_bound=0.00001
                                                                          ,optimization_upper_bound=1
                                                                          ),
                                   #ode_solver_method='BDF'
                                   #minimize_method = 'COBYQA'
                                   ode_solver_method = 'Radau'
                                   )
#%%
fits = []
fit_res_dfs = []
for sub in scale_df['SUBJID'].unique():
    fit_df = scale_df.loc[scale_df['SUBJID'] == sub, :].copy()
    fit = me_mod_fo2.fit2(fit_df,)
    fit_df['pred_y'] = fit.predict2(fit_df)
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
res_df[me_mod_fo2.model_name] = me_mod_fo2.predict2(scale_df)
piv_cols.append(me_mod_fo2.model_name)
me_mod_fo2.save_fitted_model(jb_file_name = me_mod_fo2.model_name)
#long_df = scale_df.melt(id_vars = ['SUBJID', 'TIME'], value_vars = piv_cols, value_name='Conc', var_name = 'pred_method')

#%%
import plotly.express as px

long_df = res_df.melt(id_vars = ['SUBJID', 'TIME'], value_vars = piv_cols, value_name='Conc', var_name = 'pred_method')
px.line(data_frame=long_df, x = 'TIME', y = 'Conc', color = 'pred_method', line_group='SUBJID', animation_frame='SUBJID')


#%%
b_i_apprx_df = pd.DataFrame( dtype = pd.Float64Dtype())
b_i_apprx_df['b_i_fo_cl'] = me_mod_fo.b_i_approx[('cl', 'omega2_cl')].to_numpy()
b_i_apprx_df['SUBJID'] = scale_df['SUBJID'].drop_duplicates().values
scale_df = (scale_df.merge(b_i_apprx_df, how = 'left', on = 'SUBJID') 
            if 'b_i_fo_cl' not in scale_df.columns else scale_df.copy())
#%%
fit_foce = False 
if fit_foce:
    me_mod_foce =  CompartmentalModel(
        model_name = "debug_hydrocortisone_abs_ka-clME-vd_foce",
            ode_t0_cols=[ ODEInitVals('DV_scale'), ODEInitVals('dose_scale'),],
            conc_at_time_col = 'DV_scale',
            population_coeff=[
                                PopulationCoeffcient('ka', 2, 
                                                    #subject_level_intercept=True,
                                                    optimization_lower_bound = np.log(.1),
                                                    optimization_upper_bound = np.log(3),
                                                    #subject_level_intercept_sd_init_val = 0.2, 
                                                    #subject_level_intercept_sd_upper_bound = 20,
                                                    #subject_level_intercept_sd_lower_bound=1e-6
                                                    ),
                                PopulationCoeffcient('cl',
                                                    14,
                                                    optimization_lower_bound = np.log(5),
                                                    optimization_upper_bound = np.log(25),
                                                    subject_level_intercept=True, 
                                                    subject_level_intercept_sd_init_val = 0.3, 
                                                    subject_level_intercept_sd_upper_bound = 5,
                                                    subject_level_intercept_sd_lower_bound=1e-6,
                                                    subject_level_intercept_init_vals_column_name='b_i_fo_cl'
                                                    ),
                                PopulationCoeffcient('vd', 40
                                                    , optimization_lower_bound = np.log(30)
                                                    , optimization_upper_bound = np.log(50)
                                                    ),
                            ],
            dep_vars= None, 
                                    no_me_loss_function=sum_of_squares_loss, 
                                    no_me_loss_needs_sigma=False,
                                    me_loss_function = FOCE_approx_ll_loss,
                                    optimizer_tol=None, 
                                    pk_model_class=OneCompartmentAbsorption(), 
                                    model_error_sigma=PopulationCoeffcient('sigma'
                                                                            ,log_transform_init_val=False
                                                                            , optimization_init_val=.5
                                                                            ,optimization_lower_bound=0.00001
                                                                            ,optimization_upper_bound=1
                                                                            ),
                                    #ode_solver_method='BDF'
                                    )




    fit_model = False
    if fit_model:
        me_mod_foce = me_mod_foce.fit2(scale_df, )
    else:
        with open(f"logs/fitted_model_{me_mod_foce.model_name}.jb", 'rb') as f:
            me_mod_foce = jb.load(f)
    res_df[me_mod_foce.model_name] = me_mod_foce.predict2(scale_df)
    piv_cols.append(me_mod_foce.model_name)
    me_mod_foce.save_fitted_model(jb_file_name=me_mod_foce.model_name)

#%%
b_i_apprx_df = pd.DataFrame( dtype = pd.Float64Dtype())
b_i_apprx_df['b_i_fo_cl2'] = me2_mod_fo.b_i_approx[('cl', 'omega2_cl')].to_numpy()
b_i_apprx_df['b_i_fo_ka2'] = me2_mod_fo.b_i_approx[('ka', 'omega2_ka')].to_numpy()
b_i_apprx_df['SUBJID'] = scale_df['SUBJID'].drop_duplicates().values
scale_df = (scale_df.merge(b_i_apprx_df, how = 'left', on = 'SUBJID') 
            if 'b_i_fo_cl2' not in scale_df.columns else scale_df.copy())
#%%
me2_mod_foce =  CompartmentalModel(
           ode_t0_cols=[ ODEInitVals('DV_scale'), ODEInitVals('dose_scale'),],
          conc_at_time_col = 'DV_scale',
          population_coeff=[
                            PopulationCoeffcient('ka', 2, 
                                                 subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(.1),
                                                 optimization_upper_bound = np.log(3),
                                                 subject_level_intercept_sd_init_val = 0.2, 
                                                 subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6, 
                                                 subject_level_intercept_init_vals_column_name='b_i_fo_ka2'
                                                 ),
                            PopulationCoeffcient('cl',
                                                 14,
                                                  optimization_lower_bound = np.log(5),
                                                 optimization_upper_bound = np.log(25),
                                                subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.3, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6,
                                                subject_level_intercept_init_vals_column_name='b_i_fo_cl2'
                                                 ),
                            PopulationCoeffcient('vd', 40
                                                , optimization_lower_bound = np.log(30)
                                                , optimization_upper_bound = np.log(50)
                                                ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   no_me_loss_needs_sigma=False,
                                   me_loss_function = FOCE_approx_ll_loss,
                                   optimizer_tol=None, 
                                   pk_model_class=OneCompartmentAbsorption(), 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.5
                                                                          ,optimization_lower_bound=0.00001
                                                                          ,optimization_upper_bound=1
                                                                          ),
                                   #ode_solver_method='BDF'
                                   )




me2_mod_foce.fit2(scale_df,)
res_df['me2_foce_preds'] = me2_mod_foce.predict2(scale_df)
piv_cols.append('me2_foce_preds')

#with open('me_mod_debug_foce.jb', 'wb') as f:
#    jb.dump(me_mod_foce, f)


# %%
