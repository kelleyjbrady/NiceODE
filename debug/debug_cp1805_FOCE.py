#%%
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64']='True'
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import numpyro
numpyro.set_host_device_count(4)

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jb
from niceode.utils import (CompartmentalModel, 
                           PopulationCoeffcient,
                           ODEInitVals,
                           neg2_log_likelihood_loss,
                           FOCE_approx_ll_loss, 
                           FOCEi_approx_ll_loss
)
from niceode.diffeqs import( 
                    first_order_one_compartment_model, #dy/dt = -k * C
                    first_order_one_compartment_model2, #dy/dt = -cl/vd * C
                    OneCompartmentAbsorption
                    )
import numpy as np
from niceode.pymc_utils import make_pymc_model
import pymc as pm
from datetime import datetime



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
                                   batch_id='focei_debug1',
                                   minimize_method = 'COBYQA'
                                   )
fit_model = False
if fit_model:
    me_mod_fo = me_mod_fo.fit2(df, ci_level=None)
else:
    with open(f"/workspaces/PK-Analysis/logs/fitted_model_{me_mod_fo.model_name}.jb", 'rb') as f:
        me_mod_fo = jb.load(f)
#res_df[me_mod_fo.model_name] = me_mod_fo.predict2(df)
#piv_cols.append(me_mod_fo.model_name)
me_mod_fo.save_fitted_model(jb_file_name = me_mod_fo.model_name)


#%%
b_i_apprx_df = pd.DataFrame( dtype = pd.Float64Dtype())
b_i_apprx_df['b_i_fo_ka'] = me_mod_fo.b_i_approx[('ka', 'omega2_ka')].to_numpy()
b_i_apprx_df['b_i_fo_cl'] = me_mod_fo.b_i_approx[('cl', 'omega2_cl')].to_numpy()
b_i_apprx_df['b_i_fo_vd'] = me_mod_fo.b_i_approx[('vd', 'omega2_vd')].to_numpy()
b_i_apprx_df['SUBJID'] = df['SUBJID'].drop_duplicates().values
scale_df = (df.merge(b_i_apprx_df, how = 'left', on = 'SUBJID') 
            if 'b_i_fo_cl' not in df.columns else df.copy())

me_mod_foce =  CompartmentalModel(
    model_name = "debug_cp1805_abs_kaME-cl-vdME_FOCE_nodep_dermal",
          ode_t0_cols=[ ODEInitVals('DV_scale'), ODEInitVals('AMT_scale'),],
          conc_at_time_col = 'DV_scale',
          subject_id_col = 'ID_x', 
          time_col = 'TIME_x',
          population_coeff=[
                            PopulationCoeffcient('ka', optimization_init_val = .7, 
                                                 subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(1e-6),
                                                 #optimization_upper_bound = np.log(3),
                                                 subject_level_intercept_sd_init_val = 0.4, 
                                                 subject_level_intercept_sd_upper_bound = 2,
                                                subject_level_intercept_sd_lower_bound=1e-2,
                                                subject_level_intercept_init_vals_column_name='b_i_fo_ka'
                                                 ),
                            PopulationCoeffcient('cl',
                                                 optimization_init_val = .1,
                                                  optimization_lower_bound = np.log(1e-4),
                                                  optimization_upper_bound=np.log(1),
                                                 #optimization_upper_bound = np.log(.005),
                                                subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.3, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6,
                                                subject_level_intercept_init_vals_column_name='b_i_fo_cl'
                                                
                                                 ),
                            PopulationCoeffcient('vd', optimization_init_val = 1.2
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
                                   pk_model_class=OneCompartmentAbsorption, 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.9
                                                                          ,optimization_lower_bound=0.01
                                                                          ,optimization_upper_bound=1.5
                                                                          ),
                                   #ode_solver_method='BDF'
                                   batch_id='focei_debug1',
                                   minimize_method = 'COBYQA'
                                   )
fit_model = True
if fit_model:
    me_mod_foce = me_mod_foce.fit2(scale_df, ci_level = None, stiff_ode = True )
    
#%%
me_mod_focei =  CompartmentalModel(
    model_name = "debug_cp1805_abs_kaME-cl-vdME_FOCEi_nodep_dermal",
          ode_t0_cols=[ ODEInitVals('DV_scale'), ODEInitVals('AMT_scale'),],
          conc_at_time_col = 'DV_scale',
          subject_id_col = 'ID_x', 
          time_col = 'TIME_x',
          population_coeff=[
                            PopulationCoeffcient('ka', optimization_init_val = .7, 
                                                 subject_level_intercept=True,
                                                 optimization_lower_bound = np.log(1e-6),
                                                 #optimization_upper_bound = np.log(3),
                                                 subject_level_intercept_sd_init_val = 0.4, 
                                                 subject_level_intercept_sd_upper_bound = 2,
                                                subject_level_intercept_sd_lower_bound=1e-2,
                                                subject_level_intercept_init_vals_column_name='b_i_fo_ka'
                                                 ),
                            PopulationCoeffcient('cl',
                                                 optimization_init_val = .1,
                                                  optimization_lower_bound = np.log(1e-4),
                                                  optimization_upper_bound=np.log(1),
                                                subject_level_intercept=True, 
                                                subject_level_intercept_sd_init_val = 0.3, 
                                                subject_level_intercept_sd_upper_bound = 5,
                                                subject_level_intercept_sd_lower_bound=1e-6,
                                                subject_level_intercept_init_vals_column_name='b_i_fo_cl'
                                                
                                                 ),
                            PopulationCoeffcient('vd', optimization_init_val = 1.2
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
                                   me_loss_function = FOCEi_approx_ll_loss,
                                   no_me_loss_needs_sigma=True,
                                   optimizer_tol=None, 
                                   pk_model_class=OneCompartmentAbsorption, 
                                   model_error_sigma=PopulationCoeffcient('sigma'
                                                                          ,log_transform_init_val=False
                                                                          , optimization_init_val=.9
                                                                          ,optimization_lower_bound=0.01
                                                                          ,optimization_upper_bound=1.5
                                                                          ),
                                   #ode_solver_method='BDF'
                                   batch_id='focei_debug1',
                                   minimize_method = 'COBYQA'
                                   )
fit_model = True
if fit_model:
    me_mod_focei = me_mod_focei.fit2(scale_df, ci_level = None )