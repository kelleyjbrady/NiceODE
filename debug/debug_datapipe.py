# %%
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax

import pandas as pd
from niceode.diffeqs import OneCompartmentAbsorption, TwoCompartmentAbsorption
from niceode.utils import CompartmentalModel, ObjectiveFunctionColumn
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
            "ka",
            optimization_init_val = 0.7,
            subject_level_intercept=True,
            optimization_lower_bound=np.log(1e-6),
            # optimization_upper_bound = np.log(3),
            subject_level_intercept_sd_init_val=0.2,
            subject_level_intercept_sd_upper_bound=20,
            subject_level_intercept_sd_lower_bound=1e-6,
        ),
        PopulationCoeffcient(
            "cl",
            optimization_init_val = 0.1,
            optimization_lower_bound=np.log(1e-4),
            optimization_upper_bound=np.log(1),
            # optimization_upper_bound = np.log(.005),
            # subject_level_intercept=True,
            # subject_level_intercept_sd_init_val = 0.3,
            # subject_level_intercept_sd_upper_bound = 5,
            # subject_level_intercept_sd_lower_bound=1e-6
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
                allometric_norm_value=70,
                model_method="allometric",
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
fit_model = True
if fit_model:
    me_mod_fo = me_mod_fo.fit2(
        df_oral,
    )
else:
    with open(f"logs/fitted_model_{me_mod_fo.model_name}.jb", "rb") as f:
        me_mod_fo = jb.load(f)
res_df[me_mod_fo.model_name] = me_mod_fo.predict2(df_oral)
piv_cols.append(me_mod_fo.model_name)
me_mod_fo.save_fitted_model(jb_file_name=me_mod_fo.model_name)

# %%
