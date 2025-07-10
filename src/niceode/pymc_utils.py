import jax.numpy as jnp
import pytensor.tensor as pt
from pytensor.graph.op import Op
import jax
from jax.experimental.ode import odeint
import numpy as np
import pymc as pm
import diffrax
from pytensor.graph.op import Apply
from pytensor.compile.ops import as_op
import pytensor as ptb
from scipy.integrate import solve_ivp
import pandas as pd
from pytensor import scan
import icomo
from copy import deepcopy
from typing import Literal



def debug_print(print_obj, *args ):
    print_str = False
    if print_str:
        for arg in args:
            print_obj = print_obj + str(arg)
        if isinstance(print_obj, str):
            print(print_obj)

#TO DO: Add allometric scaling prep
#TO DO: Verify no changes are required/it is not beneficial to implement log scale opt of bayes equivlanets of 
# sigma and omega
def make_pymc_model(model_obj,
                    pm_subj_df = None,
                    pm_df = None,
                    model_params = None,
                    model_param_dep_vars = None,
                    model_error = None,
                    link_function:Literal['exp', 'softplus'] = 'exp',
                    error_model:Literal['additive', 'proportional', 'combined'] = 'combined',
                    pop_effect_prior_cv:float = 0.3,
                    use_existing_fit = True,
                    ): 
    req_kwargs_are_none = [i is None for i in 
                           (pm_subj_df, pm_df, model_params, model_param_dep_vars) ]
    if all(req_kwargs_are_none):
        use_internal_prep = True
    else:
        use_internal_prep = False
    if use_internal_prep:
        #prepare data for pymc from previous fit
        init_summary = model_obj.init_vals_pd.copy()
        model_params = init_summary.loc[init_summary['population_coeff'], :].copy()
        model_param_dep_vars = init_summary.loc[(init_summary['population_coeff'] == False)
                                                    & (init_summary['model_error'] == False), :].copy()
        model_error = init_summary.loc[init_summary['model_error'], :].copy()
        if use_existing_fit:
            best_fit_df = model_obj.fit_result_summary_.reset_index().copy()
            
            pop_coeff_f1 = best_fit_df['population_coeff']

            best_model_params = best_fit_df.loc[pop_coeff_f1, 
                                                ['model_coeff', 'best_fit_param_val']]
            model_params = model_params.merge(best_model_params, how = 'left', on = 'model_coeff')
            init_val_src = 'best_fit_param_val'
        else:
            init_val_src = 'init_val'
        model_params['init_val'] = model_params[init_val_src].copy()
        model_params['init_val_log_scale'] = model_params['init_val'].copy()
        model_params['init_val_true_scale'] = np.exp(model_params['init_val'])
        model_params['init_val_softplus'] = np.log(np.exp(model_params['init_val_true_scale']) - 1)
         
        prior_cv = pop_effect_prior_cv
        softplus = False
        if softplus:
            f = model_params['init_val_true_scale'] < 0.5
            model_params.loc[f, 'sigma'] = np.sqrt(np.log( 1 +  prior_cv**2))
            f = model_params['init_val_true_scale'] > 10
            model_params.loc[f, 'sigma'] = model_params.loc[f, 'init_val_true_scale'] * prior_cv
            f1 = model_params['init_val_true_scale'] <= 10
            f2 = model_params['init_val_true_scale'] >= 0.5
            model_params.loc[f1 & f2, 'sigma'] = model_params.loc[f1 & f2, 'init_val_softplus'] * prior_cv
            model_params['init_val'] = model_params['init_val_softplus'].copy()
        else:
            model_params['sigma'] = np.sqrt(np.log( 1 +  prior_cv**2))

        if use_existing_fit:
            best_me_params = best_fit_df.loc[best_fit_df['subject_level_intercept']]
            best_me_params = best_me_params.rename(columns = {'best_fit_param_val':'best_fit_param_val_me'})
            best_me_params = best_me_params[['model_coeff',
                                            'subject_level_intercept_name',
                                            'best_fit_param_val_me']]
            model_params = model_params.merge(best_me_params,
                                            how = 'left',
                                            on = ['model_coeff',
                                                    'subject_level_intercept_name'])
            me_init_src = 'best_fit_param_val_me'
        else:
            me_init_src = 'subject_level_intercept_sd_init_val'
        if softplus:
            f = model_params['init_val_true_scale'] < 0.5
            model_params.loc[f, 'subject_level_intercept_sd_init_val'] = model_params.loc[f, me_init_src].copy()
            f = model_params['init_val_true_scale'] > 10
            model_params.loc[f, 'subject_level_intercept_sd_init_val'] = (model_params.loc[f, me_init_src] 
                                            * model_params.loc[f, 'init_val_true_scale'])
            f1 = model_params['init_val_true_scale'] <= 10
            f2 = model_params['init_val_true_scale'] >= 0.5
            model_params.loc[f1 & f2, 'subject_level_intercept_sd_init_val'] = (model_params.loc[f1 & f2, 'init_val_softplus'] 
                                                * model_params.loc[f1 & f2, me_init_src])
        else:
            model_params['subject_level_intercept_sd_init_val'] = model_params[me_init_src].copy()
        #Assume 20% CV for a nice wide but informative prior
        #tmp_c = 'subject_level_intercept_sd_init_val'
        #model_params['sigma_subject_level_intercept_sd_init_val'] = np.log(np.exp(model_params[tmp_c]) * .2
        #                                                                   
        #                                                                   )
        

        if use_existing_fit:
            model_error = best_fit_df.loc[best_fit_df['model_error']
                                        , 'best_fit_param_val'].to_numpy()[0]

            #find a nice reference for this approxmiationFalse
            if error_model == 'proportional':
                pm_error = model_error / np.mean(model_obj.data[model_obj.conc_at_time_col])
            if error_model == 'additive':
                pm_error = model_error
        else:
            pm_error = None

        pm_subj_df = model_obj.subject_data.copy()
        pm_df = model_obj.data.copy()
        model_params = model_params.copy()
        model_param_dep_vars = model_param_dep_vars.copy()
    
    
    pm_df['tmp'] = 1
    #time_mask_df = pm_df.pivot( index = 'SUBJID', columns = 'TIME', values = 'tmp').fillna(0)
    #time_mask = time_mask_df.to_numpy().astype(bool)
    time_mask = np.copy(model_obj.time_mask)
    #all_sub_tp_alt = pm_df.pivot( index = 'SUBJID', columns = 'TIME', values = 'TIME')    
    all_sub_tp = np.tile(model_obj.global_tp_eval, (len(time_mask),1))
    timepoints = model_obj.global_tp_eval
    dose_t0 = model_obj.dose_t0.to_numpy()
    nondim_ode_t0_vals = model_obj.ode_t0_vals_nondim.to_numpy()
    n_ode_params = len(model_params)
    model_obj = deepcopy(model_obj)
    model_obj.stiff_ode = True
    t1 = model_obj.global_tf_eval
    t0 = model_obj.global_t0_eval 
    dt0 = 0.1
    coords = {'subject':list(pm_subj_df[model_obj.groupby_col].values), 
          'obs_id': list(pm_df.index.values), 
          'global_time':timepoints, 
          'ode_output':list(model_obj.ode_t0_vals.columns)
          }
    
    #jax_odeint_op = JaxOdeintOp(one_compartment_model)
    #ode_func = ytp_ode_from_typ(one_compartment_diffrax2)
    
    #pymc_ode_model = DifferentialEquation(
    #func=ode_func, times=timepoints, n_states=1, n_theta=n_ode_params, t0=t0
#)
    
    
    
    
    old_subj_loop = True
    pt_printing = True
    with pm.Model(coords=coords) as model:
        
        data_obs = pm.Data('dv', pm_df[model_obj.conc_at_time_col].values, dims = 'obs_id')
        #debug_print(data_obs.shape.eval())
        time_mask_data = pm.Data('time_mask', time_mask, dims = ('subject', 'global_time'))
        tp_data = pm.Data('timepoints', all_sub_tp, dims = ('subject', 'global_time'))
        tp_data_vector = pm.Data('timepoints_vector', timepoints.flatten(), dims = 'global_time')
        #subject_init_conc = pm.Data('c0', pm_subj_df[model_obj.conc_at_time_col].values, dims = 'subject')
        subject_init_y0 = pm.Data('y0', model_obj.ode_t0_vals.to_numpy(), dims = ('subject', 'ode_output'))
        subject_init_y0_nondim = pm.Data('y0_nondim',
                                         nondim_ode_t0_vals,
                                         dims = ('subject', 'ode_output'))
        dose_t0 = pm.Data('dose_t0', dose_t0, dims = 'subject')

        subject_data = {}
        thetas = {}
        seen_coeff = []
        #prepare priors for thetas if they exisit, create pm.Data to hold the 
        #relevant data for use with the thetas
        for idx, row in model_param_dep_vars.iterrows():
            coeff_name = row['model_coeff']
            
            theta_name = row['model_coeff_dep_var']
            if coeff_name not in seen_coeff:
                thetas[coeff_name] = {}
                subject_data[coeff_name] = {}
            thetas[coeff_name].update({theta_name:pm.Normal(f"theta_{coeff_name}_{theta_name}", mu = 0, sigma = 10)})
            subject_data[coeff_name].update(
                {theta_name:pm.Data(f"data_{coeff_name}_{theta_name}", pm_subj_df[theta_name].values,
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
        nondim_params_list = []
        hybrid_params_list = []
        #prepare the PK model parameters and subject level effects if they exisit
        for idx, row in model_params.iterrows():
            coeff_name = row['model_coeff']
            coeff_has_subject_intercept = row['subject_level_intercept']
                         
            #prior for population coeff
            population_coeff[coeff_name]=pm.Normal(f"{coeff_name}_pop", mu = row['init_val'], sigma = row['sigma'], initval=row['init_val'])
            #if the ODE parameter (row['model_coeff']) has subject level effects, include them in the ODE parameter
            if coeff_has_subject_intercept:
                #one average subject level effect 
                coeff_intercept_mu[coeff_name] = pm.Normal(f"{coeff_name}_intercept_mu", mu = 0, sigma = 2)
                #one sd of subject level effect
                coeff_intercept_sigma[coeff_name] = pm.HalfNormal(f"{coeff_name}_intercept_sigma",
                                                                  sigma = row['subject_level_intercept_sd_init_val'], initval= row['subject_level_intercept_sd_init_val'])
                #per subject deviation from coeff_intercept_mu
                # Non-centered subject-level deviations (standard normal prior)
                z_coeff[coeff_name] = pm.Normal(
                    f"z_{coeff_name}", mu=0, sigma=1, dims="subject"
                )

                # Subject-level effect (non-centered)
                coeff_intercept_i[coeff_name] = pm.Deterministic(
                    f"{coeff_name}_intercept",
                    coeff_intercept_mu[coeff_name]
                    + z_coeff[coeff_name] * coeff_intercept_sigma[coeff_name],
                    dims="subject",
                )

                #debug_print(f"Shape of coeff_intercept_i[{coeff_name}]: {coeff_intercept_i[coeff_name].shape.eval()}")
                model_coeff = (population_coeff[coeff_name] + coeff_intercept_i[coeff_name])
            #if the ODE parameter (row['model_coeff']) DOES NOT have subject level effects
            #and thus The ODE parameter (model_coeff) remains the pop_coeff
            else:
                model_coeff = population_coeff[coeff_name]
            #if the ODE parameter includes fixed effects, include those
            if coeff_name not in thetas:
                thetas[coeff_name] = {}
                subject_data[coeff_name] = {} 
            for theta_name in thetas[coeff_name]:
                #debug_print(f"Shape of model_coeff: {model_coeff.shape.eval()}")
                #debug_print(f"Shape of thetas[{coeff_name}][{theta_name}]: {thetas[coeff_name][theta_name].shape.eval()}")
                #debug_print(f"Shape of pm_subj_df[{theta_name}]: {subject_data[coeff_name][theta_name].shape.eval()}")
                ##debug_print(f"Shape of pm_subj_df[{theta_name}][{sub_idx}]: {pm_subj_df[theta_name][sub_idx].shape}")
                model_coeff = (model_coeff + (thetas[coeff_name][theta_name] * subject_data[coeff_name][theta_name]))
            #If there are subject effects, the params will have dims = 'subject'
            if link_function == 'softplus':
                link_f = pm.math.log1pexp
                #coeffs = (model_coeff)
            if link_function == 'exp':
                link_f = pm.math.exp
                #coeffs = pm.math.exp(model_coeff)
            #coeffs = pm.math.exp(model_coeff)
            if coeff_has_subject_intercept:
                
                pm_model_params.append(
                    pm.Deterministic(f"{coeff_name}_i",
                                    link_f(model_coeff),
                                    dims = 'subject' )
                )
            #if not, we need to repeat the params n_subject's time
            else:
                pm_model_params.append(
                    pm.Deterministic(f"{coeff_name}_i",
                                    pt.repeat(link_f(model_coeff), len(coords['subject']) ),
                                    dims = 'subject'
                                     )
                )
        
        #debug_print(f"Shape of intial conc: {subject_init_conc_eval.shape}")
        #this should be called something other than theta, this is the inputs to the PK model ODE
        nondimensional = True
        hybrid_dim = False
        if hybrid_dim:
            hybrid_params = model_obj.pk_model_class.get_hybrid_nondim_defs(pm_model_params)
            for name in hybrid_params:
                hybrid_params_list.append(
                    pm.Deterministic(name, hybrid_params[name], dims="subject")
                )
            theta_matrix_hybrid = pt.concatenate([param.reshape((1, -1)) for param in hybrid_params_list], axis=0).T
        if nondimensional:
            nondim_time = model_obj.pk_model_class.get_nondim_time(pm_model_params, tp_data) #tau, (n_subject, n_timepoints_global)
            nondim_time = pm.Deterministic('tau', nondim_time, dims = ('subject', "global_time" ))
            nondim_ivp_dt0 = (pt.min(pt.diff(nondim_time, axis = 1), axis = 1) * .1).flatten()
            #nondim_ivp_dt0 = pm.Deterministic('ivp_dt0', nondim_ivp_dt0, dims = "subject")
            nondim_ivp_t1 = pt.repeat(pt.max(nondim_time), len(coords['subject'])).flatten()
            #nondim_ivp_t1 = pm.Deterministic('ivp_t1', nondim_ivp_t1, dims = "subject")
            nondim_ivp_t0 = pt.repeat(0.0, len(coords['subject'])).flatten()
            #nondim_ivp_t0 = pt.min(nondim_time, axis = 1).flatten()
            #nondim_ivp_t0 = pm.Deterministic('ivp_t0', nondim_ivp_t0, dims = "subject")
            nondim_params = model_obj.pk_model_class.get_nondim_defs(pm_model_params)
            for name in nondim_params:
                nondim_params_list.append(
                    pm.Deterministic(name, nondim_params[name], dims="subject")
                )
            theta_matrix_nondim = pt.concatenate([param.reshape((1, -1)) for param in nondim_params_list], axis=0).T  
        
        theta_matrix = pt.concatenate([param.reshape((1, -1)) for param in pm_model_params], axis=0).T
        
        #debug_print("Shape of theta_matrix:", theta_matrix_eval.shape)
        #debug_print("Shape of tp_data:",  tp_data_eval.shape)
        #debug_print("Shape of tp_data[0,:]:",  tp_data_eval[0,:].shape)

        #the loop could be useful for debugging, also demonstrates how the 
        #jax compiler works, the for loop is MUCH slower and does not 
        #sample chains in parallel on CPU
        icomo_for_loop = False
        if icomo_for_loop:
            sol_alt = []
            for sub_idx, subject in enumerate(coords['subject']):
                subject_y0 = subject_init_y0[sub_idx]
                #debug_print(subject_y0[0].shape.eval())
                subject_model_params = theta_matrix[sub_idx, :]
                #debug_print(subject_model_params.shape.eval())
            
                subject_timepoints = tp_data_vector
                #debug_print(subject_timepoints.shape)
                subject_t0 = subject_timepoints[ 0]
                #debug_print(subject_t0.shape.eval())
                subject_t1 = subject_timepoints[-1]
                #debug_print(subject_t1.shape.eval())
                args = [i[sub_idx] for i in pm_model_params]
                ode_sol = icomo.jax2pytensor(icomo.diffeqsolve)(
                        ts_out=subject_timepoints,
                        y0=subject_y0,
                        args=args,
                        ODE=model_obj.pk_model_class.diffrax_ode,
                    ).ys
                central_mass_trajectory = ode_sol[:, 0]
                concentrations = icomo.jax2pytensor(model_obj.pk_model_class.diffrax_mass_to_depvar)(
                    central_mass_trajectory, 
                    args # Pass the same parameter tuple
                )
                ode_sol = concentrations.flatten()
                sol_alt.append(ode_sol)
            
                #debug_print(ode_sol.shape.eval())
            
            sol = pt.concatenate(sol_alt)
            #debug_print(sol.shape.eval())
            time_mask_data_f = time_mask_data.flatten()
            
            sol = sol[time_mask_data_f]
            #debug_print(sol.shape.eval())

            sol = pm.Deterministic("sol", sol)
        else:
            model_coeffs = theta_matrix
            if nondimensional:
                nondim_model_coeffs = theta_matrix_nondim
                
                
                masses, concs = icomo.jax2pytensor(model_obj.jax_ivp_pymcstiff_nondim_jittable_)(
                    subject_init_y0_nondim,
                    nondim_model_coeffs,
                    model_coeffs, 
                    dose_t0, 
                    nondim_time, 
                    nondim_ivp_dt0, 
                    nondim_ivp_t0,
                    nondim_ivp_t1
                    )
            
            if hybrid_dim:
                hybrid_model_coeffs = theta_matrix_hybrid
                
                masses, concs = icomo.jax2pytensor(model_obj.jax_ivp_pymcnonstiff_hybrid_jittable_)(
                    subject_init_y0_nondim,
                    hybrid_model_coeffs,
                    model_coeffs, 
                    dose_t0, 

                    )
            else:
                masses, concs = icomo.jax2pytensor(model_obj.jax_ivp_pymcnonstiff_jittable_)(
                    subject_init_y0,
                    model_coeffs
                    )
            sol_full = masses
            
            sol_full = sol_full[:,:,0].set(concs)
            sol_dep_var = concs
            sol_dep_var = sol_dep_var.flatten()
            
            time_mask_data_f = time_mask_data.flatten()
            
            sol = sol_dep_var[time_mask_data_f]
            
            #will deal with sol_full later
            #sol_full = pt.vstack(sol_full)
            #sol_full = sol_full[time_mask_data_f]
            sol = pm.Deterministic("sol", sol)                   

        #error_model = 'combined'
        if error_model == 'additive':
            model_error = 1 if pm_error is None else pm_error
            sigma_obs = pm.HalfNormal("sigma_additive", sigma=model_error)
            pm.Normal("obs", mu=sol, sigma=sigma_obs, observed=data_obs)
        elif error_model == 'proportional':
            model_error = 1 if pm_error is None else pm_error
            sigma_obs = pm.HalfNormal("sigma_proportional", sigma=model_error)
            #this requires censoring the intial per subject vals if the t0 y is zero
            pm.LogNormal("obs", mu=pt.log(sol), sigma=sigma_obs, observed=data_obs)
        elif error_model == 'combined':
            sigma_add = pm.HalfNormal("sigma_additive", sigma=0.5)
            sigma_prop = pm.HalfNormal("sigma_proportional", sigma=0.5)
            F_for_error_model = pt.maximum(sol, 0.0)
            
            var_prop = (sigma_prop * F_for_error_model)**2
            var_add = sigma_add**2
            
            epsilon_variance = 1e-8 # A small constant for variance
            combined_sigma_obs = pt.sqrt(var_prop + var_add + epsilon_variance)
            
            pm.Normal("obs",
              mu=sol, # The model prediction F
              sigma=combined_sigma_obs,
              observed=data_obs
             )
        #additive error model 
        

    return model