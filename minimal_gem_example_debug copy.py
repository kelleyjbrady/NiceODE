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


# Placeholder for your actual model_params and model_param_dep_vars
model_params = []
model_param_dep_vars = []

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
        print(f"In perform: y0.shape = {y0.shape}, y0.dtype = {y0.dtype}")
        print(f"In perform: args.shape = {args.shape}, args.dtype = {args.dtype}")
        print(f"In perform: time_mask.shape = {time_mask.shape}, time_mask.dtype = {time_mask.dtype}")
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

def make_minimal_pymc_model(pm_subj_df, time_mask_data, timepoints):
    t1 = timepoints[-1]
    t0 = 0
    dt0 = 0.1
    coords = {"subject": pm_subj_df.index.values}

    with pm.Model(coords=coords) as model:
        # Minimal parameters
        k_i = pm.Normal("k_i", mu=0, sigma=1, dims="subject")
        vd_i = pm.Normal("vd_i", mu=0, sigma=1, dims="subject")

        # Inputs to DiffraxJaxOp
        y0 = pm.Data("y0", pm_subj_df["DV"].values.astype("float64"), dims="subject")
        theta_matrix = pt.as_tensor_variable(np.column_stack((np.exp(k_i.eval()), np.exp(vd_i.eval()))))
        time_mask = pm.Data("time_mask", time_mask_data.astype('int32'), dims=("subject", "time"))

        # DiffraxJaxOp
        diffrax_op = DiffraxJaxOp(one_compartment_diffrax, t0, t1, dt0, timepoints, time_mask_data)
        vjp_op = DiffraxVJPOp(diffrax_op.jax_op)
        diffrax_op.vjp_op = vjp_op.setup_vjp_op()
        ode_sol, _ = diffrax_op(y0, theta_matrix, time_mask)

        # Placeholder likelihood
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1)
        #pm.Normal("obs", mu=ode_sol.flatten(), sigma=sigma_obs, observed=np.random.rand(n_subjects * n_timepoints))
        pm.Normal("obs", mu=ode_sol.flatten(), sigma=sigma_obs, observed=pm_df["DV"].values)

    return model

minimal_model = make_minimal_pymc_model(pm_subj_df, time_mask_data, timepoints)

# with minimal_model:
#     trace = pm.sample(cores=1, draws=10)

# Call model.logp() to trigger graph construction
try:
    minimal_model.logp()
except Exception as e:
    import traceback
    print(traceback.format_exc())