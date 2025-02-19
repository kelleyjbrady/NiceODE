# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_subject_levels
import joblib as jb


# %%
from utils import OneCompartmentModel
from diffeqs import( OneCompartmentFODiffEq,
                    mm_one_compartment_model,
                    first_order_one_compartment_model,
                    first_order_one_compartment_model2,
                    parallel_elim_one_compartment_model, 
                    one_compartment_absorption
                    )

# %%
diffeq_obj = OneCompartmentFODiffEq()
pk_model_function = diffeq_obj.diff_eq()

# %%


# %%

from utils import sum_of_squares_loss, numba_one_compartment_model, PopulationCoeffcient, ODEInitVals, mean_squared_error_loss, huber_loss
import cProfile

with open(r'/workspaces/miniconda/PK-Analysis/debug_scale_df.jb', 'rb') as f:
    scale_df = jb.load(f)


# %%
me_mod =  OneCompartmentModel(
     ode_t0_cols=[ODEInitVals('DV')],
     population_coeff=[PopulationCoeffcient('k', .6, subject_level_intercept=True),
                       ],
     dep_vars= None, 
                              loss_function=sum_of_squares_loss, 
                              optimizer_tol=None, 
                              pk_model_function=first_order_one_compartment_model, 
                              #ode_solver_method='BDF'
                              )

# %%
me_mod.fit2(scale_df,checkpoint_filename=f'mod_abs_test_me.jb', parallel=False, parallel_n_jobs=4)
