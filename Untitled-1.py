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
me_mod =  CompartmentalModel(
          ode_t0_cols=[ODEInitVals('DV')],
          population_coeff=[PopulationCoeffcient('cl', 25, subject_level_intercept=True,
                                                 subject_level_intercept_init_val = 0.2),
                            PopulationCoeffcient('vd', 80, ),
                         ],
          dep_vars= None, 
                                   no_me_loss_function=sum_of_squares_loss, 
                                   optimizer_tol=None, 
                                   pk_model_function=first_order_one_compartment_model2, 
                                   me_loss_function=FOCE_approx_ll_loss,
                                   #ode_solver_method='BDF'
                                   )


# %%
me_mod.fit2(scale_df,checkpoint_filename=f'mod_abs_test_me.jb', n_iters_per_checkpoint=1, parallel=False, parallel_n_jobs=4)


with open('me_mod_debug.jb', 'wb') as f:
    jb.dump(me_mod, f)

