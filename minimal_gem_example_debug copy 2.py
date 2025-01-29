import os
os.environ["PYTENSOR_FLAGS"] = "optimizer=None,exception_verbosity=high"
import pymc as pm
import numpy as np
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
import jax
import jax.numpy as jnp
import diffrax
import pandas as pd
import pytensor
from scipy.integrate import solve_ivp

# Dummy data for demonstration
n_subjects = 5
n_timepoints = 10
timepoints = np.linspace(0, 10, n_timepoints)
time_mask_data = np.random.choice([True, False], size=(n_subjects, n_timepoints))
# Create pm_subj_df with subject IDs
subject_ids = np.arange(n_subjects)
pm_subj_df = pd.DataFrame({"DV": np.random.rand(n_subjects), "SUBJID": subject_ids}).set_index("SUBJID")

# Create pm_df with appropriate structure
pm_df = pd.DataFrame({
    "DV": np.random.rand(n_subjects * n_timepoints),
    "SUBJID": np.repeat(subject_ids, n_timepoints),
    "TIME_MASK": time_mask_data.flatten()
})

# Filter pm_df based on TIME_MASK
pm_df = pm_df[pm_df["TIME_MASK"]].reset_index(drop=True)

# Model parameters
t0 = 0.0
t1 = timepoints[-1]
dt0 = 0.1


# Placeholder for your actual model_params and model_param_dep_vars
model_params = []
model_param_dep_vars = []

def one_compartment_diffrax(t, y, args):
    k, Vd = args
    C = y
    dCdt = -(k / Vd) * C
    return dCdt

def solve_ode_scipy(y0, theta, time_mask, timepoints):
    """Solves the ODE for a single subject using SciPy."""
    sol = solve_ivp(
        one_compartment_diffrax,
        [t0, t1],
        y0,
        args=(theta,),
        t_eval=timepoints,
        method="RK45"
    )
    return sol.y.T[time_mask]

def make_pymc_model_potential(pm_subj_df, pm_df, model_params, model_param_dep_vars, time_mask_data, timepoints):
    coords = {"subject": pm_subj_df.index.values,
              "obs_id": pm_df.index.values}

    with pm.Model(coords=coords) as model:
        # Minimal parameters
        k_pop = pm.Normal("k_pop", mu=0, sigma=1)
        vd_pop = pm.Normal("vd_pop", mu=0, sigma=1)
        k_intercept_mu = pm.Normal("k_intercept_mu", mu=0, sigma=1)
        vd_intercept_mu = pm.Normal("vd_intercept_mu", mu=0, sigma=1)
        k_intercept_sigma = pm.HalfNormal("k_intercept_sigma", sigma=1)
        vd_intercept_sigma = pm.HalfNormal("vd_intercept_sigma", sigma=1)
        z_k = pm.Normal("z_k", mu=0, sigma=1, dims="subject")
        z_vd = pm.Normal("z_vd", mu=0, sigma=1, dims="subject")

        k_intercept = pm.Deterministic("k_intercept", k_intercept_mu + z_k * k_intercept_sigma, dims="subject")
        vd_intercept = pm.Deterministic("vd_intercept", vd_intercept_mu + z_vd * vd_intercept_sigma, dims="subject")

        k_i = pm.Deterministic("k_i", pm.math.exp(k_pop + k_intercept), dims="subject")
        vd_i = pm.Deterministic("vd_i", pm.math.exp(vd_pop + vd_intercept), dims="subject")

        # Placeholder likelihood
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1)

        # Define a function to evaluate k_i and vd_i for each subject
        def evaluate_params_for_subject(subject_idx, k_i, vd_i):
            return k_i[subject_idx].eval(), vd_i[subject_idx].eval()

        def logp_func(value, subj_indices, time_mask_data, sigma_obs, timepoints, k_i, vd_i):
            """Calculates the log-likelihood of the observed data given the parameters."""
            logp = np.array(0.0)  # Use a NumPy array for accumulating logp
            for i in range(n_subjects):
                y0 = np.array([pm_subj_df["DV"].values[i]])

                # Evaluate k_i and vd_i for this subject
                k_val, vd_val = evaluate_params_for_subject(i, k_i, vd_i)
                theta = np.array([k_val, vd_val])

                # Get the time mask for this subject
                subject_mask = subj_indices == i
                time_mask = time_mask_data[i, :]

                # Solve the ODE for this subject
                filtered_sol = solve_ode_scipy(y0, theta, time_mask, timepoints)

                # Extract observed data for this subject
                subject_data = value[subject_mask]

                # Calculate log-likelihood contribution for this subject
                logp += pm.logp(pm.Normal.dist(mu=filtered_sol, sigma=sigma_obs), subject_data).sum()

            return logp

        # Create a unique array of subject indices corresponding to each observation in pm_df
        subj_indices = pm_df["SUBJID"].values

        # Convert observed data and time_mask_data to theano shared variables
        pm.Deterministic("logp", logp_func(pm_df["DV"].values, subj_indices, time_mask_data, sigma_obs, timepoints, k_i, vd_i))

    return model

minimal_model = make_pymc_model_potential(pm_subj_df, pm_df, model_params, model_param_dep_vars, time_mask_data, timepoints)

with minimal_model:
    trace = pm.sample(cores=1, draws=10)