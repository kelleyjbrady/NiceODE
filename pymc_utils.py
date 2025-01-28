import jax.numpy as jnp
import pytensor.tensor as pt
from pytensor.graph.op import Op
import jax
from jax.experimental.ode import odeint
import numpy as np
import pymc as pm

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


def make_pymc_model(pm_subj_df, pm_df, model_params, model_param_dep_vars): 
    
    pm_df['tmp'] = 1
    time_mask_df = pm_df.pivot( index = 'SUBJID', columns = 'TIME', values = 'tmp').fillna(0)
    time_mask = time_mask_df.to_numpy().astype(bool)
    all_sub_tp_alt = pm_df.pivot( index = 'SUBJID', columns = 'TIME', values = 'TIME')    
    all_sub_tp = np.tile(all_sub_tp_alt.columns, (len(time_mask_df),1))
    
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
        #subject_tps = pm.MutableData('subject_tp', pm_subj_df['subj_tp'].values, dims = 'subject')
        #sub_tps = {}
        #for sub in coords['subject']:
        #    one_subject_tps = np.array(pm_subj_df.loc[pm_subj_df['SUBJID'] == 1.0, 'subj_tp'].values[0])
        #    sub_tps[sub] = pm.Data(f"subject{sub}_timepoints", one_subject_tps)
        subject_max_tp_data = pm.Data('subject_tp_max', pm_subj_df['subj_tp_max'].values, dims = 'subject')
        subject_min_tp_data = pm.Data('subject_tp_min', pm_subj_df['subj_tp_min'].values, dims = 'subject')

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
        pop_coeff_intercept_mu = {}
        pop_coeff_intercept_sigma = {}
        pop_coeff_intercept_i = {}
        pm_model_params = []
        for idx, row in model_params.iterrows():
            coeff_name = row['model_coeff']
            #pop_coeff_init = 0
            #while pop_coeff_init == 0:
            #    pop_coeff_init = np.random.rand()
                
            #ensure that neither init value is zero, I think that can make the graphviz fail
            population_coeff[coeff_name]=pm.Normal(f"{coeff_name}_pop", mu = 0, sigma = 3)
            
            pop_coeff_intercept_mu[coeff_name] = pm.Normal(f"{coeff_name}_intercept_mu", mu = 0, sigma = 3)
            pop_coeff_intercept_sigma[coeff_name] = pm.HalfNormal(f"{coeff_name}_intercept_sigma", sigma = 10)
            pop_coeff_intercept_i[coeff_name] = pm.Normal(f"{coeff_name}_intercept_sub",
                                                        mu = pop_coeff_intercept_mu[coeff_name], 
                                                        sigma = pop_coeff_intercept_sigma[coeff_name],
                                                        dims = 'subject'
                                                        )
            print(f"Shape of pop_coeff_intercept_i[{coeff_name}]: {pop_coeff_intercept_i[coeff_name].shape.eval()}")
            model_coeff = (population_coeff[coeff_name] + pop_coeff_intercept_i[coeff_name])
            for beta_name in betas[coeff_name]:
                print(f"Shape of model_coeff: {model_coeff.shape.eval()}")
                print(f"Shape of betas[{coeff_name}][{beta_name}]: {betas[coeff_name][beta_name].shape.eval()}")
                print(f"Shape of pm_subj_df[{beta_name}]: {subject_data[coeff_name][beta_name].shape.eval()}")
                #print(f"Shape of pm_subj_df[{beta_name}][{sub_idx}]: {pm_subj_df[beta_name][sub_idx].shape}")
                model_coeff = (model_coeff + (betas[coeff_name][beta_name] * subject_data[coeff_name][beta_name]))
            pm_model_params.append(
                pm.Deterministic(f"{coeff_name}_i", pm.math.exp(model_coeff), dims = 'subject')
            )
        
        
        print(f"Shape of intial conc: {subject_init_conc.shape.eval()}")
        print(f"Shape of subject min tp: {subject_min_tp_data.shape.eval()}")
        print(f"Shape of subject max tp: {subject_max_tp_data.shape.eval()}")
        theta_matrix = pt.concatenate([param.reshape((1, -1)) for param in pm_model_params], axis=0).T
        print("Shape of theta_matrix:", theta_matrix.shape.eval())
        ode_sol = jax_odeint_op(subject_init_conc.astype('float64'),
                                subject_min_tp_data.astype('float64'),
                                subject_max_tp_data.astype('float64'),
                                theta_matrix.astype('float64'),
                                tp_data.astype('float64'),
                                )
        #time_mask_data_reshaped = time_mask_data.reshape(n_subjects, max_time_points, 1)
        #tmp_ode_sol = pm.Deterministic("tmp_sol", ode_sol)
        filtered_ode_sol = ode_sol[time_mask_data].flatten()
        sol = pm.Deterministic("sol", filtered_ode_sol)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1)
        #print("Shape of ode_sol (in PyMC model):", ode_sol.shape)
        # or
        #print("Shape of ode_sol (in PyMC model):", pt.shape(ode_sol).eval())
        pm.LogNormal("obs", mu=sol, sigma=sigma_obs, observed=data_obs)
        

    return model