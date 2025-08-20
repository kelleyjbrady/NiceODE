#%%

import os
import finitediffx as fdx
# --- Control Flag ---
USE_GPU = False
# --------------------

if USE_GPU:
    # Set JAX to use the GPU. The device number (0) is for the first GPU.
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
    # Optional: Pin JAX to a specific GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
else:
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import numpyro
    numpyro.set_host_device_count(4)
    


import jax
print(f"JAX is running on: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

#%%
import pandas as pd

import numpyro
#numpyro.set_host_device_count(4)
from niceode.pymc_utils import make_pymc_model
import pymc as pm
#%%
from niceode.utils import (CompartmentalModel, 
                           ODEInitVals,
                           PopulationCoeffcient,
                           neg2_log_likelihood_loss,
                           ObjectiveFunctionColumn,
                           FOCE_approx_ll_loss,
                           FOCEi_approx_ll_loss,
                           FO_approx_ll_loss
                           )
from niceode.diffeqs import OneCompartmentAbsorption
import numpy as np
import joblib as jb
from niceode.jax_utils import (FOCE_approx_neg2ll_loss_jax,
                               FO_approx_neg2ll_loss_jax, 
                               FOCEi_approx_neg2ll_loss_jax,
                               FOCE_approx_neg2ll_loss_jax_fdxINNER,
                               FOCE_approx_neg2ll_loss_jax_iftINNER_ALT,
                               FOCE_approx_neg2ll_loss_jax_iftINNER, 
                               FOCE_approx_neg2ll_loss_jax_fdxOUTER, 
                               FOCEi_approx_neg2ll_loss_jax_fdxOUTER
                               )
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


#%%
me_mod_fo =  CompartmentalModel(
        model_name = "debug_theoph_abs_ka-clME-vd_JAXFOCE_jaxoptspwrapLbfgsb_fdxouteriftinner_nodep_omegadiag_dermal",
            ode_t0_cols=[ ODEInitVals('DV'), ODEInitVals('AMT'),],
            conc_at_time_col = 'DV',
            subject_id_col = 'ID', 
            time_col = 'TIME',
            population_coeff=[
                                PopulationCoeffcient('ka', 
                                                    optimization_init_val=1.6, 
                                                    subject_level_intercept=True,
                                                    #optimization_lower_bound = np.log(1e-6),
                                                    #optimization_upper_bound = np.log(15),
                                                    subject_level_intercept_sd_init_val = 0.6, 
                                                    subject_level_intercept_sd_upper_bound = 20,
                                                    subject_level_intercept_sd_lower_bound=1e-6
                                                    ),
                                PopulationCoeffcient('cl',
                                                    optimization_init_val = 3,
                                                    #optimization_lower_bound = np.log(1e-4),
                                                    #optimization_upper_bound=np.log(25),
                                                    subject_level_intercept=True, 
                                                    subject_level_intercept_sd_init_val = 0.3, 
                                                    subject_level_intercept_sd_upper_bound = 5,
                                                    subject_level_intercept_sd_lower_bound=1e-6
                                                    ),
                                PopulationCoeffcient('vd', optimization_init_val = 35,
                                                    #, optimization_lower_bound = np.log(.1)
                                                    #,optimization_upper_bound=np.log(80), 
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
                                    #optimizer_tol=1e-4,
                                    pk_model_class=OneCompartmentAbsorption, 
                                    model_error_sigma=PopulationCoeffcient('sigma'
                                                                            ,log_transform_init_val=False
                                                                            , optimization_init_val=.5
                                                                            ,optimization_lower_bound=0.00001
                                                                            ,optimization_upper_bound=3
                                                                            ),
                                    #ode_solver_method='BDF'
                                    batch_id='theoph_test1',
                                    #minimize_method = 'COBYQA'
                                    #use_full_omega=False, 
                                    significant_digits=5,
                                    me_loss_function=FO_approx_ll_loss,
                                    jax_loss=FO_approx_neg2ll_loss_jax,
                                    use_full_omega=True, 
                                    use_surrogate_neg2ll=True, 
                                    fit_jax_objective=True,
                                    )
#%%
fit_model = True
load_jb = False
if fit_model:
    me_mod_fo = me_mod_fo.fit2(df, ci_level = None, debug_fit=False, )
if load_jb:
    me_mod_fo = jb.load(r"/workspaces/PK-Analysis/debug/logs/fitted_model_cb_debug.jb.jb")

#%%
fit_pymc = True
if fit_pymc:

    model = make_pymc_model(me_mod_fo,
                            fit_df = df,
                            link_function = 'exp',
                            use_existing_fit = False,
                            )
#%%
import pymc_extras as pmx
if fit_pymc:
    chains = 4
    tune = 2000
    total_draws = 6000
    draws = np.round(total_draws / chains, 0).astype(int)
    with model:
        #trace_NUTS = pm.sample(
        #    tune=tune,
        #    draws=draws,
        #    chains=chains,
        #    nuts_sampler="numpyro",
        #    target_accept=0.92,
        #)
        
        idata_path = pm.fit(
        #method="pathfinder",
        #jitter=12,
        n=1000,
        random_seed=123,
        )

# %%

