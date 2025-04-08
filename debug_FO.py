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
                    one_compartment_absorption
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
with open(r'/workspaces/PK-Analysis/debug_scale_df.jb', 'rb') as f:
    scale_df = jb.load(f)


# %%
me_mod_fo =  CompartmentalModel(
          ode_t0_cols=[ODEInitVals('DV')],
          population_coeff=[PopulationCoeffcient('cl', 25, subject_level_intercept=True,
                                                 subject_level_intercept_sd_init_val = 0.2, 
                                                 subject_level_intercept_sd_lower_bound=1e-6
                                                 ),
                            PopulationCoeffcient('vd', 80, ),
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
                                   )

me_mod_fo.fit2(scale_df,checkpoint_filename=f'mod_abs_test_me_fo{now_str}.jb', n_iters_per_checkpoint=1, parallel=False, parallel_n_jobs=4)


with open('me_mod_debug_fo.jb', 'wb') as f:
    jb.dump(me_mod_fo, f)

