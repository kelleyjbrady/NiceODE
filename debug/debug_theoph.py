#%%
import pandas as pd
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64']='True'

import numpyro
numpyro.set_host_device_count(4)

#%%
from niceode.utils import (CompartmentalModel, 
                           ODEInitVals,
                           PopulationCoeffcient,
                           neg2_log_likelihood_loss,
                           ObjectiveFunctionColumn,
                           FOCE_approx_ll_loss
                           )
from niceode.diffeqs import OneCompartmentAbsorption
import numpy as np

#%%

df = pd.read_csv(r"/workspaces/PK-Analysis/data/theo_nlmixr2.csv", ) 
df.loc[df['AMT'] == 0.0, 'AMT'] = pd.NA
df['AMT'] = df['AMT'].ffill()
df = df.loc[df['EVID'] == 0, :].copy()
# %%

#%%
from niceode.nca import NCA
nca_obj = NCA(
    subject_id_col='ID', 
    conc_col='DV',
    time_col='TIME', 
    dose_col='AMT',
    data = df
)
# %%
noboot_tmp = nca_obj.estimate_all_nca_params(terminal_phase_adj_r2_thresh=0.85)
# %%

#%%
me_mod_fo =  CompartmentalModel(
        model_name = "debug_theoph_abs_ka-clME-vd_sse_nodep_dermal",
            ode_t0_cols=[ ODEInitVals('DV'), ODEInitVals('AMT'),],
            conc_at_time_col = 'DV',
            subject_id_col = 'ID', 
            time_col = 'TIME',
            population_coeff=[
                                PopulationCoeffcient('ka', 
                                                    optimization_init_val=1.6, 
                                                    subject_level_intercept=True,
                                                    optimization_lower_bound = np.log(1e-6),
                                                    optimization_upper_bound = np.log(15),
                                                    subject_level_intercept_sd_init_val = 0.6, 
                                                    subject_level_intercept_sd_upper_bound = 20,
                                                    subject_level_intercept_sd_lower_bound=1e-6
                                                    ),
                                PopulationCoeffcient('cl',
                                                    optimization_init_val = np.exp(2.75),
                                                    optimization_lower_bound = np.log(1e-4),
                                                    optimization_upper_bound=np.log(4),
                                                    subject_level_intercept=True, 
                                                    subject_level_intercept_sd_init_val = 0.3, 
                                                    subject_level_intercept_sd_upper_bound = 5,
                                                    subject_level_intercept_sd_lower_bound=1e-6
                                                    ),
                                PopulationCoeffcient('vd', optimization_init_val = np.exp(3.6)
                                                    , optimization_lower_bound = np.log(.1)
                                                    ,optimization_upper_bound=np.log(6), 
                                                    subject_level_intercept=True, 
                                                    subject_level_intercept_sd_init_val = 0.1, 
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

#%%
from niceode.pymc_utils import make_pymc_model
model = make_pymc_model(me_mod_fo,
                        link_function = 'exp',
                        use_existing_fit = False,
                        )
#%%
import pymc as pm
chains = 4
tune = 2000
total_draws = 6000
draws = np.round(total_draws/chains, 0).astype(int)
with model:
    #trace_DEMZ = pm.sample(step=[pm.DEMetropolisZ(vars_list)], cores = 1, tune = tune, draws = draws, chains = chains,)
    trace_NUTS = pm.sample( tune = tune, draws = draws, chains = chains, nuts_sampler = 'numpyro', target_accept = 0.92 )
 
# %%
me_mod_fo2 =  CompartmentalModel(
        model_name = "debug_theoph_abs_ka-clME-vd_FOCE_nodep_dermal",
            ode_t0_cols=[ ODEInitVals('DV'), ODEInitVals('AMT'),],
            conc_at_time_col = 'DV',
            subject_id_col = 'ID', 
            time_col = 'TIME',
            population_coeff=[
                                PopulationCoeffcient('ka', 
                                                    optimization_init_val=1.06, 
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
                                PopulationCoeffcient('vd', optimization_init_val = 3.5
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
                                    me_loss_function=FOCE_approx_ll_loss,
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
