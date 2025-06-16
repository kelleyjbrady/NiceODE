# %%
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64']='True'
import jax

import pandas as pd
from niceode.diffeqs import OneCompartmentAbsorption, TwoCompartmentAbsorption
from niceode.utils import CompartmentalModel, ObjectiveFunctionColumn
from niceode.jax_utils import FO_approx_ll_loss_jax
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from niceode.utils import (
    plot_subject_levels,
    neg2_log_likelihood_loss,
    FOCE_approx_ll_loss,
)
import joblib as jb
from niceode.utils import (
    sum_of_squares_loss,
    PopulationCoeffcient,
    ODEInitVals,
    ModelError,
    mean_squared_error_loss,
    huber_loss,
)
import cProfile
from datetime import datetime
import uuid
import numpy as np

from niceode.nca import estimate_subject_slope_cv
from niceode.nca import identify_low_conc_zones, estimate_k_halflife
from niceode.nca import calculate_mrt
from niceode.nca import prepare_section_aucs, calculate_auc_from_sections
import jax.numpy as jnp
from niceode.jax_utils import make_jittable_pk_coeff
from functools import partial
# Your JAX code will now reliably use the CPU
print(f"JAX default backend: {jax.default_backend()}")
# %%

df_c = pd.read_csv("/workspaces/PK-Analysis/data/CP1805/CP1805_conc.csv")
df_dem = pd.read_csv("/workspaces/PK-Analysis/data/CP1805/CP1805_demog.csv")
df_dem = df_dem.drop(columns=["Unnamed: 0"])
df_dose = pd.read_csv("/workspaces/PK-Analysis/data/CP1805/CP1805_dose.csv")
# df_dose = df_dose.drop(columns = ['Unnamed: 0']).drop_duplicates()
# %%

df = df_c.merge(
    df_dem, how="left", on="ID"
)  # .merge(df_c, how = 'left',on = "Unnamed: 0")

df = df.merge(df_dose, how="left", on="Unnamed: 0")
df = df.loc[df["DAY_x"] == 1, :]
df.loc[df["CONC"] == 500, "CONC"] = np.nan
df = df.loc[df["CONC"].isnull() == False, :]
# %%

df["SUBJID"] = df["ID_x"]
df["TIME"] = df["TIME_x"]
df["CONC_ng/mL"] = df["CONC"].copy()
df["DOSE_ug"] = df["DOSE"]
df["DOSE_ng"] = df["DOSE_ug"] * 1e3
df["CONC_ng/L"] = df["CONC_ng/mL"] * 1e3
df["DV_scale"] = df["CONC_ng/L"] / 1e6
df["AMT_scale"] = df["DOSE_ng"] / 1e6
df["sex_cat"] = np.where(df["SEX"] == "Male", 0, 1)
df['sex_cat_tmp'] = df['sex_cat']
df['WEIGHT_tmp'] = df['WEIGHT']
# df['DV_scale'] = df['CONC'] / 1000.0 #ng/ml = mg/L, then scale down near
# df['AMT_scale'] = df['DOSE'] / 1000 #mg, scale down
# %%
df["solve_ode_at_TIME"] = True
df_oral = df.loc[df["ROUTE"] == "Dermal", :].copy()
df_allroute = df.copy()
res_df = pd.DataFrame()
res_df_all = pd.DataFrame()
piv_cols = []
res_df[["SUBJID", "TIME", "DV_scale"]] = df_oral[["ID_x", "TIME_x", "DV_scale"]].copy()
res_df_all[["SUBJID", "TIME", "DV_scale"]] = df_allroute[
    ["ID_x", "TIME_x", "DV_scale"]
].copy()
piv_cols = ["DV_scale"]
df_nca = df_oral.copy()


me_mod_fo = CompartmentalModel(
    model_name="debugpipe_cp1805_abs_kaME-clME-vdME_FO_vdDWEIGHT_dermal",
    ode_t0_cols=[
        ODEInitVals("DV_scale"),
        ODEInitVals("AMT_scale"),
    ],
    conc_at_time_col="DV_scale",
    subject_id_col="ID_x",
    time_col="TIME_x",
    population_coeff=[
        PopulationCoeffcient(
            "cl",
            optimization_init_val = 0.7,
            subject_level_intercept=True,
            optimization_lower_bound=np.log(1e-6),
            # optimization_upper_bound = np.log(3),
            subject_level_intercept_sd_init_val=0.2,
            subject_level_intercept_sd_upper_bound=20,
            subject_level_intercept_sd_lower_bound=1e-6,
        ),
        PopulationCoeffcient(
            "ka",
            optimization_init_val = 0.1,
            optimization_lower_bound=np.log(1e-4),
            optimization_upper_bound=np.log(1),
            # optimization_upper_bound = np.log(.005),
            subject_level_intercept=True,
            subject_level_intercept_sd_init_val = 0.3,
            subject_level_intercept_sd_upper_bound = 5,
            subject_level_intercept_sd_lower_bound=1e-6
        ),
        PopulationCoeffcient(
            "vd",
            optimization_init_val = 1.2,
            optimization_lower_bound=np.log(0.1),
            optimization_upper_bound=np.log(5),
            # subject_level_intercept=True,
            # subject_level_intercept_sd_init_val = 0.3,
            # subject_level_intercept_sd_upper_bound = 5,
            # subject_level_intercept_sd_lower_bound=1e-6
            # , optimization_upper_bound = np.log(.05)
        ),
    ],
    dep_vars={
        "vd": [
            ObjectiveFunctionColumn(
                coeff_name = 'tmp',
                column_name="WEIGHT",
                allometric_norm_value=70,
                model_method="allometric",
            ),
            ObjectiveFunctionColumn(
                coeff_name = 'tmp',
                column_name="sex_cat",
            )
        ], 
        "cl": [
            ObjectiveFunctionColumn(
                coeff_name = 'tmp',
                column_name="WEIGHT_tmp",
                allometric_norm_value=70,
                model_method="allometric",
            ),
            ObjectiveFunctionColumn(
                coeff_name = 'tmp',
                column_name="sex_cat_tmp",
            )
        ],
        
    },
    dep_vars2=[
        ObjectiveFunctionColumn(
            coeff_name="vd",
            column_name="WEIGHT",
            allometric_norm_value=70,
            model_method="allometric",
        )
    ],
    no_me_loss_function=neg2_log_likelihood_loss,
    no_me_loss_needs_sigma=True,
    optimizer_tol=None,
    pk_model_class=OneCompartmentAbsorption,
    model_error_sigma=PopulationCoeffcient(
        "sigma",
        log_transform_init_val=False,
        optimization_init_val=0.2,
        optimization_lower_bound=0.00001,
        optimization_upper_bound=3,
    ),
    model_error2=ModelError("sigma",
        log_transform_init_val=False,
        optimization_init_val=0.2,
        optimization_lower_bound=0.00001,
        optimization_upper_bound=3,),
    # ode_solver_method='BDF'
    batch_id="debug_datapipe",
    minimize_method="COBYQA",
)
tmp = me_mod_fo._assemble_pred_matrices(
        df_oral,
    )
#%%


init_params, theta_data = me_mod_fo.fit2(
    df_oral,
)
n_pop_e = me_mod_fo.n_population_coeff
n_subj_e = me_mod_fo.n_subject_level_intercept_sds
unique_groups = me_mod_fo.unique_groups
groups_idx = me_mod_fo.y_groups
y = me_mod_fo.y
unpadded_y_len = len(y)
expected_len_out =  len(unique_groups)
time_mask = me_mod_fo.time_mask
time_mask_y = jnp.array(time_mask)
time_mask_J = [time_mask.reshape((time_mask.shape[0], time_mask.shape[1], 1)) for n in range(n_subj_e)]
time_mask_J = jnp.array(np.concatenate(time_mask_J, axis = 2))
ode_t0_vals = jnp.array(me_mod_fo.ode_t0_vals.to_numpy())
me_mod_fo._compile_jax_ivp_solvers()
generate_pk_model_coeff_jax = make_jittable_pk_coeff(len(unique_groups))
params_order = (i for i in init_params)

#%%
mask_df = []
for row in range(time_mask.shape[0]):
    inner_df = pd.DataFrame()
    id = unique_groups[row]
    dat = time_mask[row]
    inner_df['mask'] = dat
    inner_df['id'] = id
    inner_df['time'] = me_mod_fo.global_tp_eval
    mask_df.append(inner_df)
mask_df = pd.concat(mask_df)

tmp_y = pd.DataFrame()
tmp_y['y'] = y
tmp_y['id'] = groups_idx
tmp_y['time'] = me_mod_fo.data['TIME']

masked_y = mask_df.merge(tmp_y, how = 'left', on = ['id', 'time'])

masked_y['y'] = masked_y['y'].fillna(0.0)
padded_y = masked_y['y'].to_numpy().reshape(len(unique_groups), len(me_mod_fo.global_tp_eval))
#%%
dfs = []
td_df = pd.DataFrame(theta_data)
theta_params = [i[0] for i in pd.DataFrame(td_df).columns]
unique_pop_params = [i[0] for idx, i in enumerate(init_params) if idx < n_pop_e ]
#test inserting in the middle and at the end in addtion to the front
#unique_pop_params = unique_pop_params[:1]  + ['test'] + unique_pop_params[1:] 
#unique_pop_params = unique_pop_params + ['test2']
#%%
td_df_alt = td_df.copy()
last_seen_param = None
#new_theta_full = np.copy(unique_theta_params)
new_td_df = td_df.copy()
for p_idx, p in enumerate(unique_pop_params):
    if p not in np.unique(theta_params):
        c_new = (p, f'{p}_zero_feature')
        if last_seen_param is None:
            insert_idx = 0
        else:
            insert_idx_tmp = [idx for idx, i in enumerate(theta_params) if i == last_seen_param ]
            insert_idx = insert_idx_tmp[-1] + 1
        d_new = np.zeros((td_df.shape[0], 1))
        old_cols = list(new_td_df.columns)
        oc_pre = old_cols[:insert_idx]
        oc_post = old_cols[insert_idx:]
        new_cols = oc_pre + [c_new] + oc_post 
        new_tmp_pre = new_td_df.copy().to_numpy()[:, :insert_idx]
        new_tmp_post = new_td_df.to_numpy()[:, insert_idx:]
        new_tmp = np.concatenate([new_tmp_pre, d_new, new_tmp_post], axis = 1)
        new_td_df = pd.DataFrame(new_tmp, columns = new_cols)
        
        theta_params = [i[0] for i in new_td_df.columns]
    last_seen_param = p
#%%
td_tensor = []    
for p_idx, p in enumerate(unique_pop_params):
    tmp_df = new_td_df.copy()
    other_cols = [i for i in tmp_df.columns if i[0] != p]
    tmp_df[other_cols] = 0
    td_tensor.append(tmp_df.to_numpy())

td_tensor = np.stack(td_tensor, axis = 2)            
            
        
        


#%%v

@partial(jax.jit, static_argnames = ("compiled_gen_ode_coeff", "compiled_ivp_solver_keys"))
def ivp_predictor(pop_coeffs, thetas, theta_data, ode_t0_vals, compiled_gen_ode_coeff, compiled_ivp_solver_keys):
    model_coeffs_i = compiled_gen_ode_coeff(
        pop_coeffs, thetas, theta_data,
    )
    #model_coeffs = {i:model_coeffs_i[i[0]] for i in pop_coeffs_order}
    
    #model_coeffs_a = jnp.vstack([model_coeffs[i] for i in model_coeffs]).T
    padded_full_preds, padded_pred_y = compiled_ivp_solver_keys(
        ode_t0_vals,
        model_coeffs_i
    )
    return padded_full_preds, padded_pred_y


#%%
@jax.jit
def loss_wrapper(p):
    
    f = FO_approx_ll_loss_jax(
    p,
    params_order=params_order, 
    theta_data=theta_data,
    padded_y=jnp.array(padded_y),
    unpadded_y_len=jnp.array(unpadded_y_len),
    y_groups_idx=jnp.array(groups_idx),
    y_groups_unique=jnp.array(unique_groups), 
    n_population_coeff=n_pop_e,
    n_subject_level_effects=n_subj_e,
    time_mask_y=jnp.array(time_mask_y),
    time_mask_J=jnp.array(time_mask_J),
    compiled_ivp_solver_keys = me_mod_fo.jax_ivp_keys_stiff_compiled_solver_,
    ode_t0_vals=jnp.array(ode_t0_vals),
    compiled_gen_ode_coeff=generate_pk_model_coeff_jax,
    #compiled_ivp_predictor = ivp_predictor,
    solve_for_omegas=False
    )
    
    return f

#%%
loss_p = partial(FO_approx_ll_loss_jax, 
                 params_order=params_order, #a static_argname
                theta_data=theta_data,
                padded_y=jnp.array(padded_y),
                unpadded_y_len=jnp.array(unpadded_y_len),
                y_groups_idx=jnp.array(groups_idx),
                y_groups_unique=jnp.array(unique_groups), 
                n_population_coeff=n_pop_e, #a static_argname
                n_subject_level_effects=n_subj_e, #a static_argname
                time_mask_y=jnp.array(time_mask_y),
                time_mask_J=jnp.array(time_mask_J),
                compiled_ivp_solver_keys = me_mod_fo.jax_ivp_keys_stiff_compiled_solver_, #a static_argname
                ode_t0_vals=jnp.array(ode_t0_vals),
                compiled_gen_ode_coeff=generate_pk_model_coeff_jax, #a static_argname
                #compiled_ivp_predictor = ivp_predictor,
                solve_for_omegas=False #a static_argname
                 
                 )

#%%
#calling the loss w/out value and grad works
loss_f = jax.jit(loss_p)

res = loss_f(init_params)


#%%
fo_value_and_grad = jax.value_and_grad(loss_p, argnums = 0, has_aux = True)
#this fails
(loss, aux_data), grads = fo_value_and_grad(init_params,)
                     
#%%
neg2_ll, b_i_approx, padded_preds = res
padded_pred_y, padded_full_preds = padded_preds


# %%
