import diffrax
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import numpy as np
import pymc as pm
import pandas as pd
import pytensor
import jax
import jax.numpy as jnp

class DiffraxODE(pt.Op):
    itypes = [pt.dvector, pt.dvector, pt.dvector]  # y0, times, theta
    otypes = [pt.dmatrix]  # solutions

    def __init__(self, func, t0, solver=diffrax.Tsit5()):
        self.func = func  # ODE function is now an argument
        self.t0 = t0
        self.solver = solver
        self.term = diffrax.ODETerm(self.func)

    def perform(self, node, inputs, outputs):
        (y0, times, theta) = inputs
        y0 = np.array(y0).reshape(-1, 1)
        sol = diffrax.diffeqsolve(
            self.term,
            self.solver,
            t0=self.t0,
            t1=times.get_value()[-1],
            dt0=0.1,
            y0=y0,
            args=(theta,),  # Pass theta as a tuple
            saveat=diffrax.SaveAt(ts=times.get_value()),
        )
        outputs[0][0] = sol.ys.squeeze()

    def grad(self, inputs, output_gradients):
        (y0, times, theta) = inputs
        (output_gradient,) = output_gradients

        return [
            pytensor.gradient.grad_not_implemented(self, 0, y0),
            pytensor.gradient.grad_not_implemented(self, 1, times),
            pytensor.gradient.grad_not_implemented(self, 2, theta),
        ]

    def jvp(self, inputs, tangents):
        (y0, times, theta) = inputs
        (tangent_y0, tangent_times, tangent_theta) = tangents
        (output_gradient,) = ([pt.ones_like(self(y0, times, theta))])

        def jvp_diffrax(y0_arr, times_arr, theta_arr):
            sol = diffrax.diffeqsolve(
                self.term,
                self.solver,
                t0=self.t0,
                t1=times_arr[-1],
                dt0=0.1,
                y0=y0_arr,
                args=(theta_arr,),  # Pass theta as a tuple
                saveat=diffrax.SaveAt(ts=times_arr),
            )
            return sol.ys.squeeze()

        y0_arr = jnp.asarray(y0.get_value()).reshape(-1, 1)
        times_arr = jnp.asarray(times.get_value())
        theta_arr = jnp.asarray(theta)

        fun_of_y0 = lambda y0_arr: jvp_diffrax(y0_arr, times_arr, theta_arr)
        JVP_y0 = jax.jvp(fun_of_y0, (y0_arr,), (jnp.asarray(tangent_y0.get_value()).reshape(-1, 1),))[1]

        fun_of_theta = lambda theta_arr: jvp_diffrax(y0_arr, times_arr, theta_arr)
        JVP_theta = jax.jvp(fun_of_theta, (theta_arr,), (jnp.asarray(tangent_theta),))[1]

        output = output_gradient[0, :] @ (JVP_y0 + JVP_theta)
        return [output]

def make_pymc_model(model, data, ode_func, model_error=None):
    # Generic model building function
    coords = {
        "subject": np.unique(data["ID"].values),
        "obs_time": data["TIME"].values[data["EVID"].values == 0],
        "param_mixed": range(model.num_mixed_effects),  # Dynamic mixed effects
        "ode_param" : range(model.num_params)
    }
    n_subjects = len(coords["subject"])

    with pm.Model(coords=coords) as pymc_model:
        data_obs = pm.Data("data_obs", data["DV"].values[data["EVID"].values == 0], dims="obs_time")
        time_mask_data = pm.Data("time_mask_data", data["EVID"].values == 0, dims = "time")
        subject_id_data = pm.Data("subject_id_data", data["ID"].values, dims="time")
        tp_data_vector = pm.Data("tp_data_vector", data["TIME"].values, dims = "time")
        subject_init_conc = pm.Data("subject_init_conc", data.groupby("ID")["AMT"].first().values, dims="subject")

        # --- Parameter Handling (Generic) ---
        population_params = []
        for i in range(model.num_params):
          population_params.append(pm.Normal(f"pop_param_{i}", mu=0, sigma=1))

        omega_params = []
        for i in range(model.num_params):
          omega_params.append(pm.HalfNormal(f"omega_{i}", sigma=1))
        
        eta = pm.Normal("eta", mu=0, sigma=1, dims=("subject", "param_mixed"))

        theta_matrix = []
        for i in range(model.num_params):
          theta_matrix.append(population_params[i] + eta[:, i] * omega_params[i])
        theta_matrix = pt.stack(theta_matrix, axis=1)
        # --- ODE Solving Loop ---
        sol = []
        for sub_idx in range(n_subjects):
            subject_y0 = pt.as_tensor_variable([subject_init_conc[sub_idx]])
            subject_model_params = theta_matrix[sub_idx, :]  # All parameters for the subject
            subject_timepoints = tp_data_vector

            diffrax_op = DiffraxODE(ode_func, t0=0.0)  # Pass the ODE function
            ode_sol = diffrax_op(subject_y0, subject_timepoints, subject_model_params)

            if ode_sol.shape[0] == 0:
                print(f"Warning: Empty solution for subject {sub_idx}. Skipping.")
                continue
            sol.append(ode_sol)

        if sol:
            sol = pt.concatenate(sol, axis=0)
        else:
             raise ValueError("All solutions are empty. Check your data and model.")

        filtered_ode_sol = sol[time_mask_data]

        model_error = 1 if model_error is None else model_error
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=model_error)

        pm.LogNormal("obs", mu=pt.log(filtered_ode_sol), sigma=sigma_obs, observed=data_obs)
        return pymc_model

# --- Example Usage (One-Compartment) ---

def one_compartment_diffrax(t, y, args):
    (theta,) = args  # Unpack theta within the ODE function
    cl, Vd = theta[0], theta[1] #Correct unpacking
    C = y
    dCdt = -(cl / Vd) * C
    return dCdt
#Dummy class to represent model metadata
class Model:
  def __init__(self,num_params, num_mixed_effects):
    self.num_params = num_params
    self.num_mixed_effects = num_mixed_effects

if __name__ == '__main__':
    num_subjects = 5
    times = np.linspace(0, 24, 50)
    data_list = []
    model = Model(num_params=2, num_mixed_effects=2) # Example model info

    for i in range(num_subjects):
        cl_true = np.random.lognormal(np.log(0.2), 0.1)
        vd_true = np.random.lognormal(np.log(5), 0.1)
        dose = 100
        y0 = np.array([dose])
        theta_true = np.array([cl_true, vd_true])
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(one_compartment_diffrax),
            diffrax.Tsit5(),
            t0=0,
            t1=times[-1],
            dt0=0.1,
            y0=y0.reshape(-1,1),
            args=(theta_true,),  # Pass as a tuple
            saveat=diffrax.SaveAt(ts=times)
        )
        concentrations = sol.ys.squeeze()
        noise = np.random.normal(0, 0.2, size=len(times))
        observed_concentrations = np.exp(np.log(concentrations) + noise)

        subject_data = {
            'ID': np.full(len(times), i + 1),
            'TIME': times,
            'DV': observed_concentrations,
            'AMT': np.array([dose] + [0] * (len(times) -1)),
            "EVID": np.array([1] + [0] * (len(times) -1)),
            'param_mixed': np.array([0,1]) #dims

        }
        data_list.append(subject_data)
    empty_subject_data = {
        'ID': [num_subjects + 1] * len(times),
        'TIME': times,
        'DV': [np.nan] * len(times),
        'AMT': [0] * len(times),
        'EVID': [0] * len(times),
    }
    data = pd.concat([data_list, pd.DataFrame(empty_subject_data)], ignore_index=True)

    model = make_pymc_model(model, data, one_compartment_diffrax) # Pass ode function
    with model:
        trace = pm.sample(200, tune=200, cores=1, target_accept=0.95)
        print(pm.summary(trace))

# --- Example Usage (Two-Compartment) ---

def two_compartment_diffrax(t, y, args):
    (theta,) = args
    k10, k12, k21, Vc = theta
    C1, C2 = y
    dC1dt = -(k10 + k12) * C1 + k21 * C2 * (Vc)
    dC2dt = k12 * C1 - k21 * C2
    return pt.stack([dC1dt, dC2dt])

#With slightly different data generation
if __name__ == '__main__':
    num_subjects = 5
    times = np.linspace(0, 24, 50)
    data_list = []
    model = Model(num_params=4, num_mixed_effects=4) # Example model info

    for i in range(num_subjects):
        k10_true = np.random.lognormal(np.log(0.2), 0.1)
        k12_true = np.random.lognormal(np.log(0.3), 0.1)
        k21_true = np.random.lognormal(np.log(0.1), 0.1)
        vd_true = np.random.lognormal(np.log(5), 0.1)
        dose = 100
        y0 = np.array([dose, 0.0]) # Initial condition for both compartments
        theta_true = np.array([k10_true,k12_true, k21_true, vd_true])
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(two_compartment_diffrax),
            diffrax.Tsit5(),
            t0=0,
            t1=times[-1],
            dt0=0.1,
            y0=y0.reshape(-1,1),
            args=(theta_true,),  # Pass as a tuple
            saveat=diffrax.SaveAt(ts=times)
        )
        concentrations = sol.ys.squeeze()
        noise = np.random.normal(0, 0.2, size=concentrations.shape)
        observed_concentrations = np.exp(np.log(concentrations) + noise)

        subject_data = {
            'ID': np.full(len(times), i + 1),
            'TIME': times,
            'DV': observed_concentrations[:,0], # Only observe the first compartment
            'AMT': np.array([dose] + [0] * (len(times) -1)),
            "EVID": np.array([1] + [0] * (len(times) -1)),
            'param_mixed': np.array([0,1,2,3]) #dims

        }
        data_list.append(subject_data)
    empty_subject_data = {
        'ID': [num_subjects + 1] * len(times),
        'TIME': times,
        'DV': [np.nan] * len(times),
        'AMT': [0] * len(times),
        'EVID': [0] * len(times),
    }
    data = pd.concat([data_list, pd.DataFrame(empty_subject_data)], ignore_index=True)
    model = make_pymc_model(model, data, two_compartment_diffrax)
    with model:
        trace = pm.sample(200, tune=200, cores=1, target_accept=0.95)
        print(pm.summary(trace))