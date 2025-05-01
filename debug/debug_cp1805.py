# %%
import pandas as pd
from niceode.diffeqs import OneCompartmentAbsorption, TwoCompartmentAbsorption
from niceode.utils import CompartmentalModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from niceode.utils import plot_subject_levels, neg2_log_likelihood_loss, FOCE_approx_ll_loss
import joblib as jb
from niceode.utils import sum_of_squares_loss, numba_one_compartment_model, PopulationCoeffcient, ODEInitVals, mean_squared_error_loss, huber_loss
import cProfile
from datetime import datetime
import uuid
import numpy as np

from niceode.nca import estimate_subject_slope_cv
from niceode.nca import identify_low_conc_zones, estimate_k_halflife
from niceode.nca import calculate_mrt
from niceode.nca import prepare_section_aucs, calculate_auc_from_sections
#%%

df_c = pd.read_csv('/workspaces/PK-Analysis/data/CP1805/CP1805_conc.csv')
df_dem = pd.read_csv('/workspaces/PK-Analysis/data/CP1805/CP1805_demog.csv')
df_dem = df_dem.drop(columns = ['Unnamed: 0'])
df_dose = pd.read_csv('/workspaces/PK-Analysis/data/CP1805/CP1805_dose.csv')
#df_dose = df_dose.drop(columns = ['Unnamed: 0']).drop_duplicates()
#%%

df = df_c.merge(df_dem, how = 'left', on = 'ID')#.merge(df_c, how = 'left',on = "Unnamed: 0")

df = df.merge(df_dose, how = 'left', on = 'Unnamed: 0')
df = df.loc[df['DAY_x'] == 1, :]
df.loc[df['CONC'] == 500, 'CONC'] = np.nan
df = df.loc[df['CONC'].isnull() == False, :]
# %%

df['SUBJID'] = df['ID_x']
df['TIME'] = df['TIME_x']
df['CONC_ng/mL'] = df['CONC'].copy()
df['DOSE_ug'] = df['DOSE']
df['DOSE_ng'] = df['DOSE_ug'] * 1e3
df['CONC_ng/L'] = df['CONC_ng/mL'] * 1e3
df['DV_scale'] = df['CONC_ng/L'] / 1e6
df['AMT_scale'] = df['DOSE_ng'] / 1e6
#df['DV_scale'] = df['CONC'] / 1000.0 #ng/ml = mg/L, then scale down near
#df['AMT_scale'] = df['DOSE'] / 1000 #mg, scale down
#%%
df['solve_ode_at_TIME'] = True
df_oral = df.loc[df['ROUTE'] == 'Dermal', :].copy()
df_allroute = df.copy()
res_df = pd.DataFrame()
res_df_all = pd.DataFrame()
piv_cols = []
res_df[['SUBJID', 'TIME',  'DV_scale']] = df_oral[['ID_x', 'TIME_x', 'DV_scale']].copy()
res_df_all[['SUBJID', 'TIME',  'DV_scale']] = df_allroute[['ID_x', 'TIME_x', 'DV_scale']].copy()
piv_cols = ['DV_scale']
df_nca = df_oral.copy()
#%%

dfs = []
work_dfs = []
for sub in df_oral['SUBJID'].unique():
    work_df = df_oral.loc[df_oral['SUBJID'] == sub, :]
    work_dfs.append(work_df.copy())
    dfs.append(estimate_subject_slope_cv(work_df,
                                         conc_col='CONC_ng/L', id_col='SUBJID'))
    



zero_starts = identify_low_conc_zones(dfs, low_frac=.01, id_col = 'SUBJID')
ks = estimate_k_halflife(dfs, zero_zone_df=zero_starts, id_col = 'SUBJID')
ks[['window_halflife_est', 'window_k_est']].describe()
# %%

auc_calc_df = (df_oral
               .merge(zero_starts, how = 'left', on = 'SUBJID')
               .merge(ks, how = 'left', on = 'SUBJID')
               .copy())
auc_sections_df, aumc_sections_df = prepare_section_aucs(auc_calc_df, 
                                                         id_col = 'SUBJID', 
                                                         conc_col='CONC_ng/L', 
                                                         time_col = 'TIME'
                                                         )
aumcs_res = calculate_auc_from_sections(aumc_sections_df, id_col = 'SUBJID')
auc_res = calculate_auc_from_sections(auc_sections_df, id_col = 'SUBJID')
#auc_res.describe()

#%%
clf_df = df_oral.merge(auc_res, how = 'left', on = 'SUBJID')
#%%
clf_df = clf_df.loc[clf_df['TIME'] == 0, :].copy()
#clf_df['dose_ng'] = clf_df['AMT']*1000 #defaults to ug, dose is known to be 20mg
clf_df['cl/F_L_per_h'] = (clf_df['DOSE_ng'] #ng
                   /clf_df['linup_logdown_auc'] #ng*h/L
                   )
clf_df['cl/F__L/hr'] = clf_df['cl/F_L_per_h'] #/ 1000
clf_df['cl/F__L/hr'].describe()

#%%


mrt_res = calculate_mrt(aumcs_res, auc_res, id_col = 'SUBJID')
vss_res = mrt_res.merge(clf_df, how = 'left', on = 'SUBJID')
vss_res['vss'] = (vss_res['mrt'] * #hr
                  vss_res['cl/F__L/hr'] #L/R
                  )
vss_res['vss'].describe()

#%%
from niceode.nca import NCA
nca_obj = NCA(
    subject_id_col='SUBJID', 
    conc_col='CONC_ng/L',
    time_col='TIME', 
    dose_col='DOSE_ng',
    data = df_nca
)
#%%
noboot_tmp = nca_obj.estimate_all_nca_params(terminal_phase_adj_r2_thresh=0.85)
#tmp = nca_obj.estimate_all_nca_params(terminal_phase_adj_r2_thresh=0.85, n_boots=10)
#%%
me_mod_fo =  CompartmentalModel(
    model_name = "debug_cp1805_abs_ka-clME-vd_sse_nodep_dermal",
          ode_t0_cols=[ ODEInitVals('DV_scale'), ODEInitVals('AMT_scale'),],
          conc_at_time_col = 'DV_scale',
          subject_id_col = 'ID_x', 
          time_col = 'TIME_x',
          population_coeff=[
                            PopulationCoeffcient('ka', 2, 
                                                 #subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(1e-6),
                                                 #optimization_upper_bound = np.log(3),
                                                 #subject_level_intercept_sd_init_val = 0.2, 
                                                 #subject_level_intercept_sd_upper_bound = 20,
                                                #subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('cl',
                                                 .1,
                                                  optimization_lower_bound = np.log(1e-4),
                                                  optimization_upper_bound=np.log(1)
                                                 #optimization_upper_bound = np.log(.005),
                                                #subject_level_intercept=True, 
                                                #subject_level_intercept_sd_init_val = 0.3, 
                                                #subject_level_intercept_sd_upper_bound = 5,
                                                #subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('vd', 1.2
                                                , optimization_lower_bound = np.log(.1)
                                                ,optimization_upper_bound=np.log(5)
                                                
                                                #, optimization_upper_bound = np.log(.05)
                                                ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=neg2_log_likelihood_loss, 
                                   no_me_loss_needs_sigma=True,
                                   optimizer_tol=None, 
                                   pk_model_class=OneCompartmentAbsorption(), 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.5
                                                                          ,optimization_lower_bound=0.00001
                                                                          ,optimization_upper_bound=3
                                                                          ),
                                   #ode_solver_method='BDF'
                                   minimize_method = 'COBYQA'
                                   )
fit_model = True
if fit_model:
    me_mod_fo = me_mod_fo.fit2(df_oral, )
else:
    with open(f"logs/fitted_model_{me_mod_fo.model_name}.jb", 'rb') as f:
        me_mod_fo = jb.load(f)
res_df[me_mod_fo.model_name] = me_mod_fo.predict2(df_oral)
piv_cols.append(me_mod_fo.model_name)
me_mod_fo.save_fitted_model(jb_file_name = me_mod_fo.model_name)

#%%

fits = []
fit_res_dfs = []
for sub in df_oral['ID_x'].unique():
    fit_df = df_oral.loc[df_oral['ID_x'] == sub, :].copy()
    fit = me_mod_fo.fit2(fit_df,)
    fit_df['pred_y'] = fit.predict2(fit_df)
    fits.append(fit.fit_result_)
    fit_res_dfs.append(fit_df.copy())
fit_res_df = pd.concat(fit_res_dfs)
fit_readout = [np.exp(i['x']) for i in fits]
res_df['indiv_fit_preds'] = fit_res_df['pred_y'].copy()
piv_cols.append('indiv_fit_preds')
# %%
me_mod_fo =  CompartmentalModel(
    model_name = "debug_cp1805_abs_kaME-clME-vdME_FO_nodep_dermal",
          ode_t0_cols=[ ODEInitVals('DV_scale'), ODEInitVals('AMT_scale'),],
          conc_at_time_col = 'DV_scale',
          subject_id_col = 'ID_x', 
          time_col = 'TIME_x',
          population_coeff=[
                            PopulationCoeffcient('ka', .7, 
                                                 subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(1e-6),
                                                 #optimization_upper_bound = np.log(3),
                                                 subject_level_intercept_sd_init_val = 0.2, 
                                                 subject_level_intercept_sd_upper_bound = 20,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('cl',
                                                 .1,
                                                  optimization_lower_bound = np.log(1e-4),
                                                  optimization_upper_bound=np.log(1),
                                                 #optimization_upper_bound = np.log(.005),
                                                subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.3, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('vd', 1.2
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
                                   pk_model_class=OneCompartmentAbsorption(), 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.2
                                                                          ,optimization_lower_bound=0.00001
                                                                          ,optimization_upper_bound=3
                                                                          ),
                                   #ode_solver_method='BDF'
                                   minimize_method = 'COBYQA'
                                   )
fit_model = True
if fit_model:
    me_mod_fo = me_mod_fo.fit2(df_oral, )
else:
    with open(f"logs/fitted_model_{me_mod_fo.model_name}.jb", 'rb') as f:
        me_mod_fo = jb.load(f)
res_df[me_mod_fo.model_name] = me_mod_fo.predict2(df_oral)
piv_cols.append(me_mod_fo.model_name)
me_mod_fo.save_fitted_model(jb_file_name = me_mod_fo.model_name)

#%%
b_i_apprx_df = pd.DataFrame( dtype = pd.Float64Dtype())
b_i_apprx_df['b_i_fo_ka'] = me_mod_fo.b_i_approx[('ka', 'omega2_ka')].to_numpy()
b_i_apprx_df['b_i_fo_cl'] = me_mod_fo.b_i_approx[('cl', 'omega2_cl')].to_numpy()
b_i_apprx_df['b_i_fo_vd'] = me_mod_fo.b_i_approx[('vd', 'omega2_vd')].to_numpy()
b_i_apprx_df['SUBJID'] = df_oral['SUBJID'].drop_duplicates().values
scale_df = (df_oral.merge(b_i_apprx_df, how = 'left', on = 'SUBJID') 
            if 'b_i_fo_cl' not in df_oral.columns else df_oral.copy())

me_mod_foce =  CompartmentalModel(
    model_name = "debug_cp1805_abs_kaME-clME-vdME_FOCE_nodep_dermal",
          ode_t0_cols=[ ODEInitVals('DV_scale'), ODEInitVals('AMT_scale'),],
          conc_at_time_col = 'DV_scale',
          subject_id_col = 'ID_x', 
          time_col = 'TIME_x',
          population_coeff=[
                            PopulationCoeffcient('ka', .7, 
                                                 subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(1e-6),
                                                 #optimization_upper_bound = np.log(3),
                                                 subject_level_intercept_sd_init_val = 0.4, 
                                                 subject_level_intercept_sd_upper_bound = 2,
                                                subject_level_intercept_sd_lower_bound=1e-2,
                                                subject_level_intercept_init_vals_column_name='b_i_fo_ka'
                                                 ),
                            PopulationCoeffcient('cl',
                                                 .1,
                                                  optimization_lower_bound = np.log(1e-4),
                                                  optimization_upper_bound=np.log(1),
                                                 #optimization_upper_bound = np.log(.005),
                                                #subject_level_intercept=True, 
                                                #subject_level_intercept_sd_init_val = 0.3, 
                                                #subject_level_intercept_sd_upper_bound = 5,
                                                #subject_level_intercept_sd_lower_bound=1e-6,
                                                #subject_level_intercept_init_vals_column_name='b_i_fo_cl'
                                                
                                                 ),
                            PopulationCoeffcient('vd', 1.2
                                                , optimization_lower_bound = np.log(.1)
                                                ,optimization_upper_bound=np.log(5),
                                                subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.25, 
                                                subject_level_intercept_sd_upper_bound = 2,
                                                subject_level_intercept_sd_lower_bound=1e-2,
                                                #, optimization_upper_bound = np.log(.05)
                                                subject_level_intercept_init_vals_column_name='b_i_fo_vd'
                                                
                                                ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=neg2_log_likelihood_loss, 
                                   me_loss_function = FOCE_approx_ll_loss,
                                   no_me_loss_needs_sigma=True,
                                   optimizer_tol=None, 
                                   pk_model_class=OneCompartmentAbsorption(), 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.9
                                                                          ,optimization_lower_bound=0.01
                                                                          ,optimization_upper_bound=1.5
                                                                          ),
                                   #ode_solver_method='BDF'
                                   minimize_method = 'COBYQA'
                                   )
fit_model = True
if fit_model:
    me_mod_foce = me_mod_foce.fit2(df_oral, )
else:
    with open(f"logs/fitted_model_{me_mod_foce.model_name}.jb", 'rb') as f:
        me_mod_foce = jb.load(f)
res_df[me_mod_foce.model_name] = me_mod_foce.predict2(df_oral)
piv_cols.append(me_mod_foce.model_name)
me_mod_foce.save_fitted_model(jb_file_name = me_mod_foce.model_name)


#%%
df_oral['c2_init'] = 0.0
me_mod_fo =  CompartmentalModel(
    model_name = "debug_cp1805_abs_ka-cl-v1-q-v2_sse_nodep",
          ode_t0_cols=[ ODEInitVals('DV_scale'),ODEInitVals('c2_init'), ODEInitVals('AMT_scale'),],
          conc_at_time_col = 'DV_scale',
          subject_id_col = 'ID_x', 
          time_col = 'TIME_x',
          population_coeff=[
                            PopulationCoeffcient('ka', 1, 
                                                 #subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(.1),
                                                 optimization_upper_bound = np.log(3),
                                                 #subject_level_intercept_sd_init_val = 0.2, 
                                                 #subject_level_intercept_sd_upper_bound = 20,
                                               # subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('cl',
                                                 .08,
                                                  optimization_lower_bound = np.log(.01),
                                                 optimization_upper_bound = np.log(1),
                                                #subject_level_intercept=True, 
                                                #subject_level_intercept_sd_init_val = 0.3, 
                                                #subject_level_intercept_sd_upper_bound = 5,
                                                #subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('v1', .85
                                                , optimization_lower_bound = np.log(1e-3)
                                                , optimization_upper_bound = np.log(5)
                                                #,subject_level_intercept=True, 
                                                #subject_level_intercept_sd_init_val = 0.3, 
                                                #subject_level_intercept_sd_upper_bound = 5,
                                                #subject_level_intercept_sd_lower_bound=1e-6
                                                ),
                            PopulationCoeffcient('q', .5
                                                , optimization_lower_bound = np.log(1e-6)
                                                , optimization_upper_bound = np.log(10)
                                                ),
                            PopulationCoeffcient('v2', .2
                                                , optimization_lower_bound = np.log(1e-3)
                                                , optimization_upper_bound = np.log(5)
                                                ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=neg2_log_likelihood_loss, 
                                   no_me_loss_needs_sigma=True,
                                   optimizer_tol=None, 
                                   pk_model_class=TwoCompartmentAbsorption(), 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.5
                                                                          ,optimization_lower_bound=0.00001
                                                                          ,optimization_upper_bound=3
                                                                          ),
                                   #ode_solver_method='BDF'
                                   minimize_method = 'COBYQA'
                                   )

#%%
fits = []
fit_res_dfs = []
for sub in df_oral['ID_x'].unique():
    fit_df = df_oral.loc[df_oral['ID_x'] == sub, :].copy()
    fit = me_mod_fo.fit2(fit_df,)
    fit_df['pred_y'] = fit.predict2(fit_df)
    fits.append(fit.fit_result_)
    fit_res_dfs.append(fit_df.copy())
fit_res_df = pd.concat(fit_res_dfs)
fit_readout = [np.exp(i['x']) for i in fits]
res_df['indiv_fit_preds_2cmptabs'] = fit_res_df['pred_y'].copy()
piv_cols.append('indiv_fit_preds_2cmptabs')
#%%
df_oral['c2_init'] = 0.0
me_mod_fo =  CompartmentalModel(
    model_name = "debug_cp1805_2cmpabs_kaME-clME-v1ME-qME-v2ME_neg2ll_nodep",
          ode_t0_cols=[ ODEInitVals('DV_scale'),ODEInitVals('c2_init'), ODEInitVals('AMT_scale'),],
          conc_at_time_col = 'DV_scale',
          subject_id_col = 'ID_x', 
          time_col = 'TIME_x',
          population_coeff=[
                            PopulationCoeffcient('ka', 1, 
                                                 subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(.1),
                                                 optimization_upper_bound = np.log(3),
                                                 subject_level_intercept_sd_init_val = 0.2, 
                                                 subject_level_intercept_sd_upper_bound = 20,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('cl',
                                                 .08,
                                                  optimization_lower_bound = np.log(.01),
                                                 optimization_upper_bound = np.log(1),
                                                subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.3, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('v1', .15
                                                , optimization_lower_bound = np.log(1e-2)
                                                , optimization_upper_bound = np.log(1)
                                                ,subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.3, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                ),
                            PopulationCoeffcient('q', 1
                                                , optimization_lower_bound = np.log(.1)
                                                , optimization_upper_bound = np.log(4)
                                                ,subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.3, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                ),
                            PopulationCoeffcient('v2', .5
                                                , optimization_lower_bound = np.log(1e-2)
                                                , optimization_upper_bound = np.log(5)
                                                ,subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.3, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6
                                                ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   no_me_loss_needs_sigma=False,
                                   optimizer_tol=None, 
                                   pk_model_class=TwoCompartmentAbsorption(), 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.5
                                                                          ,optimization_lower_bound=0.00001
                                                                          ,optimization_upper_bound=3
                                                                          ),
                                   #ode_solver_method='BDF'
                                   minimize_method = 'COBYQA'
                                   )


#%%
fit_model = True
if fit_model:
    me_mod_fo = me_mod_fo.fit2(df_oral, )
else:
    with open(f"logs/fitted_model_{me_mod_fo.model_name}.jb", 'rb') as f:
        me_mod_fo = jb.load(f)
res_df[me_mod_fo.model_name] = me_mod_fo.predict2(df_oral)
piv_cols.append(me_mod_fo.model_name)
me_mod_fo.save_fitted_model(jb_file_name = me_mod_fo.model_name)
# %%
import plotly.express as px

long_df = res_df.melt(id_vars = ['SUBJID', 'TIME'], value_vars = piv_cols, value_name='Conc', var_name = 'pred_method')
px.line(data_frame=long_df, x = 'TIME', y = 'Conc', color = 'pred_method', line_group='SUBJID', animation_frame='SUBJID')

#test
# %%
