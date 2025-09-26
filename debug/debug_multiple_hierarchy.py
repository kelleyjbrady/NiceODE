#%%
#Set the devices which Jax should use. 
#This must be done before importing jax
import os
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

import finitediffx as fdx
import pandas as pd
import numpyro
from niceode.pymc_utils import make_pymc_model
import pymc as pm
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
from niceode.jax_utils import FO_approx_neg2ll_loss_jax

#%%

df =  pd.read_csv(r"/workspaces/PK-Analysis/data/CP1805/CP1805Data.csv").iloc[1:, :]
df = df.loc[~df['Conc'].isnull(), :]
df = df.loc[df['Day'] == 1.0, :].reset_index(drop = True)
#%%
me_mod_fo =  CompartmentalModel(
        model_name = "debug_cp1805_multiple_hierarchy",
            ode_t0_cols=[ ODEInitVals('Conc'), ODEInitVals('Dose'),],
            conc_at_time_col = 'Conc',
            subject_id_col = 'Subject', 
            time_col = 'Time',
            population_coeff=[
                                PopulationCoeffcient('ka', 
                                                    optimization_init_val=1.6, 
                                                    subject_level_intercept=True,
                                                    #optimization_lower_bound = np.log(1e-6),
                                                    #optimization_upper_bound = np.log(15),
                                                    subject_level_intercept_sd_init_val = 0.6, 
                                                    #subject_level_intercept_sd_upper_bound = 20,
                                                    #subject_level_intercept_sd_lower_bound=1e-6, 
                                                    aux_hierarchy_levels=True,
                                                    aux_hierarchy_names=['ka_route'], 
                                                    aux_hierarchy_init_vals=[0.0], 
                                                    aux_hierarchy_group_by_cols=['Route'], 
                                                    
                                                    
                                                    ),
                                PopulationCoeffcient('cl',
                                                    optimization_init_val = 3,
                                                    #optimization_lower_bound = np.log(1e-4),
                                                    #optimization_upper_bound=np.log(25),
                                                    subject_level_intercept=True, 
                                                    subject_level_intercept_sd_init_val = 0.3, 
                                                   # subject_level_intercept_sd_upper_bound = 5,
                                                   # subject_level_intercept_sd_lower_bound=1e-6
                                                    ),
                                PopulationCoeffcient('vd', optimization_init_val = 35,
                                                    #, optimization_lower_bound = np.log(.1)
                                                    #,optimization_upper_bound=np.log(80), 
                                                    subject_level_intercept=True, 
                                                    subject_level_intercept_sd_init_val = 0.1, 
                                                    #subject_level_intercept_sd_upper_bound = 5,
                                                    #subject_level_intercept_sd_lower_bound=1e-6
                                                    
                                                    #, optimization_upper_bound = np.log(.05)
                                                    ),
                            ],
            dep_vars= None, 

                                    pk_model_class=OneCompartmentAbsorption, 
                                    model_error_sigma=PopulationCoeffcient('sigma'
                                                                            ,log_transform_init_val=False
                                                                            , optimization_init_val=.5
                                                                            ,optimization_lower_bound=0.00001
                                                                            ,optimization_upper_bound=3
                                                                            ),
  
                                    batch_id='theoph_test1',

                                    significant_digits=3,
                                    #me_loss_function=FO_approx_ll_loss,
                                    jax_loss=FO_approx_neg2ll_loss_jax,
                                    use_full_omega=True, 
                                    use_surrogate_neg2ll=True, 
                                    fit_jax_objective=True,
                                    )

#%%
model = make_pymc_model(me_mod_fo,
                            fit_df = df,
                            link_function = 'exp',
                            use_existing_fit = False,
                            )
#%%