#%%
import pandas as pd
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64']='True'

df = pd.read_table(r"/workspaces/PK-Analysis/data/theoph.txt", ) 
# %%


df.loc[df['CONC'] == '.', 'CONC'] = 0
df.loc[df['AMT'] == '.', 'AMT'] = pd.NA
df[['CONC', 'TIME', 'AMT']] = df[['CONC', 'TIME', 'AMT']].astype(pd.Float64Dtype())
# %%

df['AMT'] = df['AMT'].ffill()
df['solve_ode_at_t'] = True
df['AMT'] = 320
#%%
from niceode.nca import NCA
nca_obj = NCA(
    subject_id_col='ID', 
    conc_col='CONC',
    time_col='TIME', 
    dose_col='AMT',
    data = df
)
# %%
noboot_tmp = nca_obj.estimate_all_nca_params(terminal_phase_adj_r2_thresh=0.85)
# %%
from niceode.utils import (CompartmentalModel, 
                           ODEInitVals,
                           PopulationCoeffcient,
                           neg2_log_likelihood_loss,
                           ObjectiveFunctionColumn
                           
                           )
from niceode.diffeqs import OneCompartmentAbsorption
import numpy as np

#%%
me_mod_fo =  CompartmentalModel(
        model_name = "debug_theoph_abs_ka-clME-vd_sse_nodep_dermal",
            ode_t0_cols=[ ODEInitVals('CONC'), ODEInitVals('AMT'),],
            conc_at_time_col = 'CONC',
            subject_id_col = 'ID', 
            time_col = 'TIME',
            population_coeff=[
                                PopulationCoeffcient('ka', 
                                                    optimization_init_val=2, 
                                                    subject_level_intercept=True,
                                                    optimization_lower_bound = np.log(1e-6),
                                                    optimization_upper_bound = np.log(15),
                                                    subject_level_intercept_sd_init_val = 0.2, 
                                                    subject_level_intercept_sd_upper_bound = 20,
                                                    subject_level_intercept_sd_lower_bound=1e-6
                                                    ),
                                PopulationCoeffcient('cl',
                                                    optimization_init_val = 1,
                                                    optimization_lower_bound = np.log(1e-4),
                                                    optimization_upper_bound=np.log(4),
                                                    subject_level_intercept=True, 
                                                    subject_level_intercept_sd_init_val = 0.3, 
                                                    subject_level_intercept_sd_upper_bound = 5,
                                                    subject_level_intercept_sd_lower_bound=1e-6
                                                    ),
                                PopulationCoeffcient('vd', optimization_init_val = 3
                                                    , optimization_lower_bound = np.log(.1)
                                                    ,optimization_upper_bound=np.log(6), 
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
                                    #optimizer_tol=None, 
                                    pk_model_class=OneCompartmentAbsorption, 
                                    model_error_sigma=PopulationCoeffcient('sigma'
                                                                            ,log_transform_init_val=False
                                                                            , optimization_init_val=.5
                                                                            ,optimization_lower_bound=0.00001
                                                                            ,optimization_upper_bound=3
                                                                            ),
                                    #ode_solver_method='BDF'
                                    batch_id='theoph_test1',
                                    minimize_method = 'COBYQA'
                                    )

fit_model = True
if fit_model:
    me_mod_fo = me_mod_fo.fit2(df, ci_level = None)

# %%
me_mod_fo2 =  CompartmentalModel(
        model_name = "debug_theoph_abs_ka-clME-vd_sse_nodep_dermal",
            ode_t0_cols=[ ODEInitVals('CONC'), ODEInitVals('AMT'),],
            conc_at_time_col = 'CONC',
            subject_id_col = 'ID', 
            time_col = 'TIME',
            population_coeff=[
                                PopulationCoeffcient('ka', 
                                                    optimization_init_val=2, 
                                                    subject_level_intercept=True,
                                                    optimization_lower_bound = np.log(1e-6),
                                                    optimization_upper_bound = np.log(15),
                                                    subject_level_intercept_sd_init_val = 0.2, 
                                                    subject_level_intercept_sd_upper_bound = 20,
                                                    subject_level_intercept_sd_lower_bound=1e-6
                                                    ),
                                PopulationCoeffcient('cl',
                                                    optimization_init_val = 1,
                                                    optimization_lower_bound = np.log(1e-4),
                                                    optimization_upper_bound=np.log(4),
                                                    subject_level_intercept=True, 
                                                    subject_level_intercept_sd_init_val = 0.3, 
                                                    subject_level_intercept_sd_upper_bound = 5,
                                                    subject_level_intercept_sd_lower_bound=1e-6
                                                    ),
                                PopulationCoeffcient('vd', optimization_init_val = 3
                                                    , optimization_lower_bound = np.log(.1)
                                                    ,optimization_upper_bound=np.log(6), 
                                                    subject_level_intercept=True, 
                                                    subject_level_intercept_sd_init_val = 0.3, 
                                                    subject_level_intercept_sd_upper_bound = 5,
                                                    subject_level_intercept_sd_lower_bound=1e-6
                                                    
                                                    #, optimization_upper_bound = np.log(.05)
                                                    ),
                            ],
            dep_vars= {'vd':[ObjectiveFunctionColumn(coeff_name = 'vd',column_name='WEIGHT', optimization_lower_bound=-5,
                                                   optimization_upper_bound = 5),
                           ]},
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
                                    batch_id='theoph_test1',
                                    minimize_method = 'COBYQA'
                                    )

fit_model = True
if fit_model:
    me_mod_fo2 = me_mod_fo2.fit2(df, ci_level = None)
# %%
