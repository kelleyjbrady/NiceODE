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

# Dummy data for demonstration
n_subjects = 5
n_timepoints = 10
timepoints = np.linspace(0, 10, n_timepoints)
time_mask_data = np.random.choice([True, False], size=(n_subjects, n_timepoints))
pm_subj_df = pd.DataFrame({"DV": np.random.rand(n_subjects)}) #need to update to correct shape
pm_df = pd.DataFrame({"DV": np.random.rand(n_subjects * n_timepoints)[time_mask_data.flatten()]}) #need to update to correct shape

# Model parameters
t0 = 0.0
t1 = timepoints[-1]
dt0 = 0.1


def one_compartment_diffrax(t, y, args):
    k, Vd = args
    C = y
    dCdt = -(k / Vd) * C
    return dCdt

class DiffraxJaxOp(Op):
    __props__ = ("term", "solver", "t0", "t1", "dt0", "saveat", "stepsize_controller")

    itypes = [pt.dvector, pt.dvector, pt.bmatrix]
    otypes = [pt.dmatrix, pt.dvector]

    def __init__(self, func, t0, t1, dt0, timepoints, time_mask):
        if not callable(func):
            raise ValueError("`func` must be a callable object.")
        if not isinstance(timepoints, (list, tuple, np.ndarray)) or len(timepoints) < 2:
            raise ValueError("`timepoints` must be a list, tuple, or array with at least two elements.")

        self.func = func
        self.t0 = t0
        self.t1 = t1
        self.dt0 = dt0
        self.term = diffrax.ODETerm(self.func)
        self.solver = diffrax.Dopri5()
        self.saveat = diffrax.SaveAt(ts=timepoints)
        self.stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
        self.time_mask = time_mask

        self.jax_op = self.setup_jax_op()
        self.vjp_op = None

    def setup_jax_op(self):
        def jax_sol_op(y0, args, time_mask):
            sol = diffrax.diffeqsolve(
                self.term,
                self.solver,
                t0=self.t0,
                t1=self.t1,
                dt0=self.dt0,
                y0=y0,
                args=args,
                saveat=self.saveat,
                stepsize_controller=self.stepsize_controller
            )
            filtered_sol = sol.ys.T[time_mask]
            return filtered_sol, args
        return jax.jit(jax_sol_op)

    def make_node(self, y0, args, time_mask):
        inputs = [
            pt.as_tensor_variable(y0),
            pt.as_tensor_variable(args),
            pt.as_tensor_variable(time_mask)
        ]
        outputs = [
            pt.matrix(dtype="float64"),
            pt.vector(dtype="float64")
        ]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        y0, args, time_mask = inputs
        # Convert NumPy arrays to JAX DeviceArrays
        y0 = jnp.asarray(y0)
        args = jnp.asarray(args)
        time_mask = jnp.asarray(time_mask)
        result, args_out = self.jax_op(y0, args, time_mask)
        print("Type of result:", type(result))
        print("Shape of result:", result.shape)
        print("Result:", result)
        outputs[0][0] = np.asarray(result, dtype="float64")
        outputs[1][0] = np.asarray(args_out, dtype="float64")

    def grad(self, inputs, output_gradients):
        (y0, args, time_mask) = inputs
        (gz,_) = output_gradients
        return [
            self.vjp_op(y0, args, time_mask, gz)[0],
            pm.gradient_not_implemented(
            op=self,
            x_pos=1,
            x=args,
            msg="Gradient w.r.t. args not implemented",
        ),
            pm.gradient_not_implemented(
            op=self,
            x_pos=2,
            x=time_mask,
            msg="Gradient w.r.t. time_mask not implemented",
        )
        ]

class DiffraxVJPOp(Op):
    __props__ = ()

    itypes = [pt.dvector, pt.dvector, pt.bmatrix, pt.dmatrix]
    otypes = [pt.dvector]

    def __init__(self, jax_op):
        self.jax_op = jax_op
        self.grad_op = None

    def make_node(self, y0, args, time_mask, gz):
        inputs = [
            pt.as_tensor_variable(y0),
            pt.as_tensor_variable(args),
            pt.as_tensor_variable(time_mask),
            pt.as_tensor_variable(gz)
        ]
        outputs = [
            pt.vector(dtype="float64")
        ]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        y0, args, time_mask, gz = inputs
        # Convert NumPy arrays to JAX DeviceArrays
        y0 = jnp.asarray(y0)
        args = jnp.asarray(args)
        time_mask = jnp.asarray(time_mask)
        gz = jnp.asarray(gz)
        result = self.jax_op(y0, args, time_mask, gz)
        outputs[0][0] = np.asarray(result, dtype="float64")

    def setup_vjp_op(self):
        def vjp_sol_op(y0, args, time_mask, g):
            jax_op_lambda = lambda y0, args: self.jax_op(y0, args, time_mask)[0]
            _, vjp_fn = jax.vjp(jax_op_lambda, y0, args)
            (result,) = vjp_fn(g)
            return result
        return jax.jit(vjp_sol_op)

def make_pymc_model(pm_subj_df, pm_df, time_mask_data, timepoints): 
    t1 = timepoints[-1]
    t0 = 0 
    dt0 = 0.1
    coords = {'subject':list(pm_subj_df.index.values), 
          'obs_id': list(pm_df.index.values)
          }

    with pm.Model(coords=coords) as model:
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

        theta_matrix = pt.concatenate([k_i.reshape((-1, 1)), vd_i.reshape((-1, 1))], axis=1)
        # Make k_i depend on a random variable
        k_i_noise = pm.Normal("k_i_noise", mu=0, sigma=1, dims="subject")
        k_i_rv = pm.Deterministic("k_i_rv", k_i + k_i_noise, dims="subject")

        #k_i_rv = pm.Normal("k_i_rv", mu=k_i, sigma=0.1, dims="subject")
        
        diffrax_op = DiffraxJaxOp(one_compartment_diffrax, t0, t1, dt0, timepoints, time_mask_data)
        vjp_op = DiffraxVJPOp(diffrax_op.jax_op)
        diffrax_op.vjp_op = vjp_op.setup_vjp_op()
        ode_sol = diffrax_op(pm_subj_df["DV"].values.astype("float64"),
                             pt.concatenate([k_i_rv.reshape((-1, 1)), vd_i.reshape((-1, 1))], axis=1).astype("float64"),
                             time_mask_data)[0]

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1)
        pm.LogNormal("obs", mu=ode_sol.flatten(), sigma=sigma_obs, observed=pm_df["DV"].values)

    return model

model = make_pymc_model(pm_subj_df, pm_df, time_mask_data, timepoints)
print("Model's initial point:", model.initial_point())


with model:
    # Get the logp function
    logp_fn = model.logp()

    # Compile the function
    compiled_logp = logp_fn.compile_fn(
        allow_input_downcast=True,
        on_unused_input='ignore'
    )

    # Print the computational graph
    print("Computational Graph:")
    pytensor.printing.debugprint(compiled_logp.fn)

#with model:
#    trace = pm.sample(cores=1, draws = 10)