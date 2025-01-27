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
    C = y[0]  # Extract concentration from the state vector
    dCdt = -(k/Vd) * C  # Calculate the rate of change
    return jnp.array(dCdt)



class JaxOdeintOp(Op):
    itypes = [pt.dvector, pt.dscalar, pt.dscalar, pt.dmatrix, pt.dmatrix, pt.dmatrix]
    # y0 (n_compartments,), t0 (scalar), tf (scalar), theta (n_subjects, n_params), all_t (n_subjects, max_time_points), mask (n_subjects, max_time_points)
    otypes = [pt.dmatrix]

    def __init__(self, func):
        self.func = func
        self.grad_op = None  # Define grad_op if you need gradients

    def perform(self, node, inputs, outputs):
        y0, t0, tf, theta, all_t, time_mask = inputs

        def solve_for_subject(y0_i, theta_i, t_i, mask_i):
            # Select valid times using the mask, replace invalid with the first time point
            times = jnp.where(mask_i, t_i, t_i[0])

            # Solve ODE using odeint
            solution = odeint(self.func, y0_i, jnp.array([t0, tf]), *theta_i, t_eval=times)
            return solution

        # Vectorize across subjects using jax.vmap
        subject_solutions = jax.vmap(solve_for_subject)(y0, theta, all_t, time_mask)

        # Store the result, ensuring the correct data type
        outputs[0][0] = np.asarray(subject_solutions, dtype=all_t.dtype)


    def infer_shape(self, fgraph, node, input_shapes):
        # The output shape will be (n_subjects, n_time_points, n_compartments)
        n_subjects = input_shapes[3][0]  # theta
        n_time_points = input_shapes[4][1]  # all_t
        n_compartments = input_shapes[0][0]  # y0
        return [(n_subjects, n_time_points, n_compartments)]

def make_pymc_model(pm_subj_df, pm_df, model_params, model_param_dep_vars): 
    
    pm_df['tmp'] = 1
    time_mask_df = pm_df.pivot( index = 'SUBJID', columns = 'TIME', values = 'tmp').fillna(0)
    time_mask = time_mask_df.to_numpy().astype(bool)
    all_sub_tp_alt = pm_df.pivot( index = 'SUBJID', columns = 'TIME', values = 'TIME').fillna(-1).values
    
    coords = {'subject':list(pm_subj_df['SUBJID'].values), 
          'obs_id': list(pm_df.index.values)
          }
    
    jax_odeint_op = JaxOdeintOp(one_compartment_model)
    old_subj_loop = True
    pt_printing = True
    with pm.Model(coords=coords) as model:
        
        data_obs = pm.Data('dv', pm_df['DV'].values, dims = 'obs_id')
        time_mask_data = pm.Data('time_mask', time_mask, dims = ('subject', 'time'))
        tp_data = pm.Data('timepoints', all_sub_tp_alt, dims = ('subject', 'time'))
        subject_init_conc = pm.Data('c0', pm_subj_df['DV'].values, dims = 'subject')
        #subject_tps = pm.MutableData('subject_tp', pm_subj_df['subj_tp'].values, dims = 'subject')
        #sub_tps = {}
        #for sub in coords['subject']:
        #    one_subject_tps = np.array(pm_subj_df.loc[pm_subj_df['SUBJID'] == 1.0, 'subj_tp'].values[0])
        #    sub_tps[sub] = pm.Data(f"subject{sub}_timepoints", one_subject_tps)
        #subject_max_tp = pm.Data('subject_tp_max', pm_subj_df['subj_tp_max'].values, dims = 'subject')
        #subject_min_tp = pm.Data('subject_tp_min', pm_subj_df['subj_tp_min'].values, dims = 'subject')

        subject_data = {}
        betas = {}
        seen_coeff = []
        for idx, row in model_param_dep_vars.iterrows():
            coeff_name = row['model_coeff']
            
            beta_name = row['model_coeff_dep_var']
            if coeff_name not in seen_coeff:
                betas[coeff_name] = {}
                subject_data[coeff_name] = {}
            betas[coeff_name].update({beta_name:pm.Normal(f"beta_{coeff_name}_{beta_name}", mu = 0, sigma = 1)})
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
            pop_coeff_init = 0
            while pop_coeff_init == 0:
                pop_coeff_init = np.random.rand()
                
            #ensure that neither init value is zero, I think that can make the graphviz fail
            population_coeff[coeff_name]=pm.Normal(f"{coeff_name}_pop", mu = 0, sigma = 1, initval=pop_coeff_init)
            
            pop_coeff_intercept_mu[coeff_name] = pm.Normal(f"{coeff_name}_intercept_mu", mu = 0, sigma = 1)
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
                model_coeff = pm.math.exp((model_coeff + betas[coeff_name][beta_name] * subject_data[coeff_name][beta_name]))
            pm_model_params.append(
                pm.Deterministic(f"{coeff_name}_i", model_coeff, dims = 'subject')
            )
        all_conc = [] 
        print(coords['subject'])
        for sub_idx, subject_id in enumerate(coords['subject']):
            print(subject_id)
            subject_data = pm_df.loc[pm_df['SUBJID'] == subject_id, :]
            print(subject_data.shape)
            #initial_conc = subject_init_conc[sub_idx]
            initial_conc = subject_data['DV'].values[0]#.item()
            t_eval = subject_data['TIME'].values
            #t_eval = sub_tps[subject]
            #t_span = [subject_min_tp[sub_idx], subject_max_tp[sub_idx]]
            t_span = [np.min(subject_data['TIME'].values), np.max(subject_data['TIME'].values)]
            theta = [i[sub_idx] for i in pm_model_params]
            if old_subj_loop:
                
                def create_forward_model(subject_id_val, t_span, t_eval, initial_conc):
                    @as_op(itypes=[pt.dscalar for i in pm_model_params], otypes=[pt.dvector])
                    #@functools.wraps(pytensor_forward_model_matrix)
                    def pytensor_forward_model_matrix(*args):
                        #print(subject_id_val)
                        theta = [i for i in args]
                        sol = solve_ivp(one_compartment_model, t_span, [initial_conc], t_eval=t_eval, args=(theta,) )
                        #print("sol.status:", sol.status)  # Print the status code
                        #print("sol.message:", sol.message) # Print the status message
                        ode_sol = sol.y[0]
                        #print("\nShape of ode_sol within function:", ode_sol.shape)
                        #print("\nValues of ode_sol within function:", ode_sol)
                        return ode_sol
                    return pytensor_forward_model_matrix
            pytensor_forward_model_matrix = create_forward_model(subject_id, t_span, t_eval, initial_conc, )
            
            #theta = pytensor.printing.Print("\nShape of theta before stack")(pt.shape(theta))
            #theta = pm.math.stack(theta)
            #theta = pytensor.printing.Print("\nShape of theta after stack")(pt.shape(theta))
            #print
            if old_subj_loop:
                ode_sol = pytensor_forward_model_matrix(*theta) #issue could be that this is not the same length for each subject
            else:
                sol = solve_ivp(one_compartment_model, t_span, [initial_conc], t_eval=t_eval, args=(*theta,) )
                #print(sol)
                ode_sol = sol#.y[0] 
            #if pt_printing:
                #_ = pytensor.printing.Print("Shape of ode_sol")(pt.shape(ode_sol))
                #ode_sol = pytensor.printing.Print("ode_sol Values:")(ode_sol)
            all_conc.append(ode_sol)
        all_conc = pt.concatenate(all_conc, axis=0)
        all_conc_pm = pm.Deterministic('all_conc', all_conc)
        #if pt_printing:
        #    all_conc = pytensor.printing.Print("Shape of all_conc")(
        #    pt.shape(all_conc)
        #)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1)
        pm.LogNormal("obs", mu=all_conc_pm, sigma=sigma_obs, observed=data_obs)
    return model