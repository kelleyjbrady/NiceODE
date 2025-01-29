import jax.numpy as jnp
import pytensor.tensor as pt
from pytensor.graph.op import Op
import jax
from jax.experimental.ode import odeint
import numpy as np
import pymc as pm
import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
from pytensor.gradient import grad_not_implemented
from pytensor.graph.op import Apply, Op
from pytensor.compile.ops import as_op
from pytensor.tensor.type import TensorType
import pytensor.tensor as pt
from scipy.integrate import solve_ivp

# One-compartment model using diffrax
def one_compartment_diffrax(t, y, args):
    k, Vd = args
    C = y
    dCdt = -(k / Vd) * C
    return dCdt

class DiffraxJaxOp(Op):
    __props__ = ("term", "solver", "t0", "t1", "dt0", "saveat", "stepsize_controller")

    itypes = [pt.dvector, pt.dvector]
    otypes = [pt.dmatrix]

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
        self.vjp_op = self.setup_vjp_op()

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
    
    def setup_vjp_op(self):
        def vjp_sol_op(y0, args, time_mask, g):
            jax_op_lambda = lambda y0, args: self.jax_op(y0, args, time_mask)[0]
            _, vjp_fn = jax.vjp(jax_op_lambda, y0, args)
            (result,) = vjp_fn(g)
            return result
        return jax.jit(vjp_sol_op)

    def make_node(self, y0, args, time_mask):
        inputs = [
            pt.as_tensor_variable(y0),
            pt.as_tensor_variable(args),
            pt.as_tensor_variable(time_mask)
        ]
        outputs = [
            pt.matrix(dtype="float64"),
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
        result, args_out = self.jax_op(y0, args, time_mask)
        outputs[0][0] = np.asarray(result, dtype="float64")
        outputs[1][0] = np.asarray(args_out, dtype="float64")
    
    def grad(self, inputs, output_gradients):
        (y0, args, time_mask) = inputs
        (gz,_) = output_gradients
        return [
            self.vjp_op(y0, args, gz)[0],
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

    itypes = [pt.dvector, pt.dvector, pt.dmatrix]
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

def one_compartment_model(t, y, *theta ):
    """
    Defines the differential equation for a one-compartment pharmacokinetic model.

    This function calculates the rate of change of drug concentration in the central 
    compartment over time.

    Args:
      t (float): Time point (not used in this specific model, but required by solve_ivp).
      y (list): Current drug concentration in the central compartment.
      k (float): Elimination rate constant.
      Vd (float): Volume of distribution.


    Returns:
      float: The rate of change of drug concentration (dC/dt).
    """
    k, Vd = theta
    C = y#[0]  # Extract concentration from the state vector
    dCdt = -(k/Vd) * C  # Calculate the rate of change
    #print("Shape of dCdt (inside one_compartment_model):", jnp.shape(dCdt))
    #print("Shape of jnp.array(dCdt) (inside one_compartment_model):", jnp.shape(jnp.array(dCdt)))
    assert jnp.shape(dCdt) == (), "dCdt should be a scalar"
    return dCdt



class JaxOdeintOp(Op):
    itypes = [pt.dvector, pt.dvector, pt.dvector, pt.dmatrix, pt.dmatrix]
    # y0 (n_compartments,), t0 (scalar), tf (scalar), theta (n_subjects, n_params), all_t (n_subjects, max_time_points)
    otypes = [pt.dmatrix]

    def __init__(self, func):
        self.func = func
        self.grad_op = None  # Define grad_op if you need gradients

    def perform(self, node, inputs, outputs):
        y0, t0, tf, theta, all_t = inputs

        def solve_for_subject(y0_i, theta_i, t_i):

            # Solve ODE using odeint
            #y0_i = jnp.atleast_1d(y0_i)
            #print("Shape of y0_i:", y0_i.shape)
            #print("Shape of theta_i:", theta_i.shape)
            #print("Shape of t_i:", t_i.shape)
            solution = odeint(self.func, y0_i, t_i, *theta_i, )
            #print("Shape of solution (inside solve_for_subject):", solution.shape) 
            return solution

        # Vectorize across subjects using jax.vmap
        subject_solutions = jax.vmap(solve_for_subject)(y0, theta, all_t)
        #print("Shape of subject_solutions (inside JaxOdeintOp):", subject_solutions.shape)
        #print("Shape of subject_solutions (inside JaxOdeintOp):", subject_solutions.shape)

        # Store the result, ensuring the correct data type
        outputs[0][0] = np.asarray(subject_solutions, dtype=all_t.dtype)


    def infer_shape(self, fgraph, node, input_shapes):
        # The output shape will be (n_subjects, n_time_points, n_compartments)
        n_subjects = input_shapes[3][0]  # theta
        n_time_points = input_shapes[4][1]  # all_t
        n_compartments = input_shapes[0][0]  # y0
        #return [(n_subjects, n_time_points, n_compartments)]
        return [(n_subjects, n_time_points)]

def solve_ode_scipy(y0, t0, t1, theta, timepoints):
    """Solves the ODE for a single subject using SciPy."""
    sol = solve_ivp(
        one_compartment_diffrax,
        [t0, t1],
        y0,
        args=(theta,),
        t_eval=timepoints,
        method="RK45"
    )
    return sol.y.T

def evaluate_params_for_subject(subject_idx, k_i, vd_i):
            return k_i[subject_idx].eval(), vd_i[subject_idx].eval()
        
def solve_subject_odes(subject_indices, k_i, vd_i, y0_i, time_mask, timepoints):
    full_sol = []
    for sub_idx in subject_indices:
        y0 = y0_i[sub_idx]
        k_val, vd_val = evaluate_params_for_subject(sub_idx, k_i, vd_i)
        theta = np.array([k_val, vd_val])
        time_mask_i = time_mask[sub_idx, :]
        filtered_sol = solve_ode_scipy(y0, theta, time_mask_i, timepoints)
        full_sol.extend(filtered_sol)
    return np.array(full_sol)

def generate_subject_ivp_op(subject_y0, subject_t0, subject_t1, subject_timepoints ):
    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model_matrix(theta):
        return solve_ode_scipy(y0 = subject_y0,
                               t0 = subject_t0,
                               t1 = subject_t1,
                               theta = theta,
                               timepoints = subject_timepoints )
    return pytensor_forward_model_matrix


def make_pymc_model(pm_subj_df, pm_df, model_params, model_param_dep_vars, ode_method:str = 'scipy'): 
    
    pm_df['tmp'] = 1
    time_mask_df = pm_df.pivot( index = 'SUBJID', columns = 'TIME', values = 'tmp').fillna(0)
    time_mask = time_mask_df.to_numpy().astype(bool)
    all_sub_tp_alt = pm_df.pivot( index = 'SUBJID', columns = 'TIME', values = 'TIME')    
    all_sub_tp = np.tile(all_sub_tp_alt.columns, (len(time_mask_df),1))
    timepoints = np.array(all_sub_tp_alt.columns)
    #t1 = timepoints[-1]
    #t0 = 0 
    dt0 = 0.1
    coords = {'subject':list(pm_subj_df['SUBJID'].values), 
          'obs_id': list(pm_df.index.values)
          }
    
    jax_odeint_op = JaxOdeintOp(one_compartment_model)
    old_subj_loop = True
    pt_printing = True
    with pm.Model(coords=coords) as model:
        
        data_obs = pm.Data('dv', pm_df['DV'].values, dims = 'obs_id')
        time_mask_data = pm.Data('time_mask', time_mask, dims = ('subject', 'time'))
        tp_data = pm.Data('timepoints', all_sub_tp, dims = ('subject', 'time'))
        subject_init_conc = pm.Data('c0', pm_subj_df['DV'].values, dims = 'subject')
        global_t0 = tp_data[0,0]
        global_t1 = tp_data[0,-1]
        #subject_tps = pm.MutableData('subject_tp', pm_subj_df['subj_tp'].values, dims = 'subject')
        #sub_tps = {}
        #for sub in coords['subject']:
        #    one_subject_tps = np.array(pm_subj_df.loc[pm_subj_df['SUBJID'] == 1.0, 'subj_tp'].values[0])
        #    sub_tps[sub] = pm.Data(f"subject{sub}_timepoints", one_subject_tps)
        subject_max_tp_data = pm.Data('subject_tp_max', pm_subj_df['subj_tp_max'].values, dims = 'subject')
        subject_min_tp_data = pm.Data('subject_tp_min', pm_subj_df['subj_tp_min'].values, dims = 'subject')
        
        subject_init_conc_eval = subject_init_conc.eval()
        tp_data_eval = tp_data.eval()
        time_mask_data_eval = time_mask_data.eval()

        subject_data = {}
        betas = {}
        seen_coeff = []
        for idx, row in model_param_dep_vars.iterrows():
            coeff_name = row['model_coeff']
            
            beta_name = row['model_coeff_dep_var']
            if coeff_name not in seen_coeff:
                betas[coeff_name] = {}
                subject_data[coeff_name] = {}
            betas[coeff_name].update({beta_name:pm.Normal(f"beta_{coeff_name}_{beta_name}", mu = 0, sigma = 10)})
            subject_data[coeff_name].update(
                {beta_name:pm.Data(f"data_{coeff_name}_{beta_name}", pm_subj_df[beta_name].values,
                                        dims = 'subject'
                                        )
                }
                )
            seen_coeff.append(coeff_name)
            
        population_coeff = {}
        coeff_intercept_mu = {}
        coeff_intercept_sigma = {}
        coeff_intercept_i = {}
        z_coeff = {}
        pm_model_params = []
        for idx, row in model_params.iterrows():
            coeff_name = row['model_coeff']
            #pop_coeff_init = 0
            #while pop_coeff_init == 0:
            #    pop_coeff_init = np.random.rand()
                
            #ensure that neither init value is zero, I think that can make the graphviz fail
            population_coeff[coeff_name]=pm.Normal(f"{coeff_name}_pop", mu = 0, sigma = 3)
            
            coeff_intercept_mu[coeff_name] = pm.Normal(f"{coeff_name}_intercept_mu", mu = 0, sigma = 3)
            coeff_intercept_sigma[coeff_name] = pm.HalfNormal(f"{coeff_name}_intercept_sigma", sigma = 10)
                    # Non-centered subject-level deviations (standard normal prior)
            z_coeff[coeff_name] = pm.Normal(
                f"z_{coeff_name}", mu=0, sigma=1, dims="subject"
            )

            # Subject-level intercept (non-centered)
            coeff_intercept_i[coeff_name] = pm.Deterministic(
                f"{coeff_name}_intercept",
                coeff_intercept_mu[coeff_name]
                + z_coeff[coeff_name] * coeff_intercept_sigma[coeff_name],
                dims="subject",
            )

            print(f"Shape of coeff_intercept_i[{coeff_name}]: {coeff_intercept_i[coeff_name].shape.eval()}")
            model_coeff = (population_coeff[coeff_name] + coeff_intercept_i[coeff_name])
            if coeff_name not in betas:
                betas[coeff_name] = {}
                subject_data[coeff_name] = {} 
            for beta_name in betas[coeff_name]:
                print(f"Shape of model_coeff: {model_coeff.shape.eval()}")
                print(f"Shape of betas[{coeff_name}][{beta_name}]: {betas[coeff_name][beta_name].shape.eval()}")
                print(f"Shape of pm_subj_df[{beta_name}]: {subject_data[coeff_name][beta_name].shape.eval()}")
                #print(f"Shape of pm_subj_df[{beta_name}][{sub_idx}]: {pm_subj_df[beta_name][sub_idx].shape}")
                model_coeff = (model_coeff + (betas[coeff_name][beta_name] * subject_data[coeff_name][beta_name]))
            pm_model_params.append(
                pm.Deterministic(f"{coeff_name}_i", pm.math.exp(model_coeff), dims = 'subject')
            )
        
        
        print(f"Shape of intial conc: {subject_init_conc_eval.shape}")
        print(f"Shape of subject min tp: {subject_min_tp_data.shape.eval()}")
        print(f"Shape of subject max tp: {subject_max_tp_data.shape.eval()}")
        theta_matrix = pt.concatenate([param.reshape((1, -1)) for param in pm_model_params], axis=0).T
        theta_matrix_eval = theta_matrix.eval()
        print("Shape of theta_matrix:", theta_matrix_eval.shape)
        print("Shape of tp_data:",  tp_data_eval.shape)
        print("Shape of tp_data[0,:]:",  tp_data_eval[0,:].shape)
        #use_diffrax = True
        if ode_method == 'diffrax':
            diffrax_op = DiffraxJaxOp(one_compartment_diffrax, t0, t1, dt0, timepoints, time_mask_data.eval())
            vjp_op = DiffraxVJPOp(diffrax_op.jax_op)
            diffrax_op.vjp_op = vjp_op
            ode_sol = diffrax_op(pm_subj_df["DV"].values.astype("float64"), theta_matrix.astype("float64"), time_mask_data)[0]
            sol = pm.Deterministic("sol", ode_sol.flatten())
        elif ode_method == 'jax_odeint':
            ode_sol = jax_odeint_op(subject_init_conc.astype('float64'),
                                    subject_min_tp_data.astype('float64'),
                                    subject_max_tp_data.astype('float64'),
                                    theta_matrix.astype('float64'),
                                    tp_data.astype('float64'),
                                    )
            print("Shape of evaluated time_mask_data:", time_mask_data.shape.eval())
            print("Shape of evaluated ode_sol:", ode_sol.shape.eval())
            filtered_ode_sol = ode_sol[time_mask_data].flatten()
            sol = pm.Deterministic("sol", filtered_ode_sol)
        elif ode_method == 'scipy':
            bad_solution = False
            #sol = []
            if bad_solution:
                sol = np.zeros_like(tp_data_eval)
                for sub_idx, subject in enumerate(coords['subject']):
                    subject_y0 = np.array([subject_init_conc_eval[sub_idx]])
                    subject_model_params = theta_matrix_eval[sub_idx, :]
                    subject_timepoints = tp_data_eval[sub_idx, :]
                    subject_t0 = tp_data_eval[sub_idx, 0]
                    subject_t1 = tp_data_eval[sub_idx, -1]
                    ode_sol = solve_ode_scipy(subject_y0, subject_t0, subject_t1, subject_model_params, subject_timepoints )
                    #sol.append(ode_sol)
                    print(ode_sol.shape)
                    sol[sub_idx, :] = ode_sol.flatten()
                #sol = np.array(sol)
                print(sol.shape)
                print(time_mask_data_eval.shape)
                #sol = np.copy(sol[time_mask_data_eval].flatten())
                sol = sol.flatten()
                time_mask_data_eval = time_mask_data_eval.flatten()
                print(sol.shape)
                print(time_mask_data_eval.shape)
                #sol = np.copy(sol[time_mask_data_eval])#.flatten()
                sol = pt.as_tensor(sol)
                print(sol.shape)
                sol = pm.Deterministic("sol", sol)
            else:
                sol = []
                for sub_idx, subject in enumerate(coords['subject']):
                    subject_y0 = [subject_init_conc[sub_idx]]
                    subject_model_params = theta_matrix[sub_idx, :]
                    subject_timepoints = tp_data[sub_idx, :]
                    subject_t0 = tp_data[sub_idx, 0]
                    subject_t1 = tp_data[sub_idx, -1]
                    fwd_func = generate_subject_ivp_op(
                        subject_init_conc, subject_t0=subject_t0,
                        subject_t1=subject_t1,
                        subject_timepoints = subject_timepoints
                    )
                    ode_sol = fwd_func(subject_model_params)
                    print(ode_sol.shape.eval())
                    #sol[sub_idx, :].set(ode_sol.flatten())
                    sol.extend(ode_sol.flatten())
                sol = np.array(sol)
                #sol = sol.flatten()
                time_mask_data_f = time_mask_data.flatten()
                
                sol = sol[time_mask_data_f]
                #sol = pt.as_tensor(sol)
                #print(sol.shape)
                sol = pm.Deterministic("sol", sol)
                    
        #time_mask_data_reshaped = time_mask_data.reshape(n_subjects, max_time_points, 1)
        #tmp_ode_sol = pm.Deterministic("tmp_sol", ode_sol)

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1)
        #print("Shape of ode_sol (in PyMC model):", ode_sol.shape)
        # or
        #print("Shape of ode_sol (in PyMC model):", pt.shape(ode_sol).eval())
        pm.LogNormal("obs", mu=sol, sigma=sigma_obs, observed=data_obs)
        

    return model