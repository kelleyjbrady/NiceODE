from scipy.integrate import solve_ivp
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from joblib import dump, load
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from typing import List, Dict
from joblib import Parallel, delayed
from numba import njit
from scipy.special import huber
import re
import os
import warnings
import pymc as pm
import arviz as az
import pytensor
import pytensor.tensor as pt

def softplus(x):
    return np.log(1 + np.exp(x))


def plot_subject_levels(df: pd.DataFrame, x='TIME', y='DV', subject='SUBJID', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    for c in df.columns:
        df.loc[df[c] == '.', c] = pd.NA
    df[y] = df[y].astype(pd.Float32Dtype())
    df[x] = df[x].astype(pd.Float32Dtype())
    sns.lineplot(data=df, x=x, y=y, hue=subject, ax=ax)

@njit
def numba_one_compartment_model(t, y, k, Vd, dose):
    return one_compartment_model(t, y, k, Vd, dose)


def one_compartment_model(t, y, k, Vd, dose):
    """
    Defines the differential equation for a one-compartment pharmacokinetic model.

    This function calculates the rate of change of drug concentration in the central 
    compartment over time.

    Args:
      t (float): Time point (not used in this specific model, but required by solve_ivp).
      y (list): Current drug concentration in the central compartment.
      k (float): Elimination rate constant.
      Vd (float): Volume of distribution.
      dose (float): Administered drug dose (not used in this model, as it assumes 
                     intravenous bolus administration where the initial concentration 
                     is directly given).

    Returns:
      float: The rate of change of drug concentration (dC/dt).
    """
    C = y[0]  # Extract concentration from the state vector
    dCdt = -(k/Vd) * C  # Calculate the rate of change
    return dCdt


@dataclass
class ObjectiveFunctionColumn:
    column_name: str
    allometric_norm_value: float = None
    model_method: str = 'linear'
    optimization_init_val: np.float64 = 0.0
    optimization_history: list[np.float64] = field(default_factory=list)
    optimization_lower_bound: np.float64 = None
    optimization_upper_bound: np.float64 = None

    def __post_init__(self):
        # Define the allowed methods. Using a dictionary for better lookup performance
        # if the list becomes very large. You can also use a set.
        allowed_methods = ['linear', 'allometric']

        if self.model_method not in allowed_methods:
            raise ValueError(f"""Method '{self.model_method.__name__}' is not an allowed method. Allowed methods are: {
                             [method.__name__ for method in allowed_methods]}""")
        if self.model_method == 'allometric' and self.allometric_norm_value is None:
            raise ValueError(f"""Method '{
                             self.model_method.__name__}' is not allowed without providing the `allometric_norm_value`""")


@dataclass
class PopulationCoeffcient:
    coeff_name: str
    optimization_init_val: np.float64
    log_transform_init_val: bool = True
    optimization_history: list[np.float64] = field(default_factory=list)
    optimization_lower_bound: np.float64 = None
    optimization_upper_bound: np.float64 = None
    subject_level_intercept:bool = False


    def __post_init__(self):
        self.optimization_init_val = (np.log(self.optimization_init_val) if self.log_transform_init_val
                                      else self.optimization_init_val)


@dataclass
class ObjectiveFunctionBeta:
    column_name: str
    value: np.float64
    allometric_norm_value: float = None
    model_method: str = 'linear'
    optimization_lower_bound: np.float64 = None
    optimization_upper_bound: np.float64 = None

    def __post_init__(self):
        # Define the allowed methods. Using a dictionary for better lookup performance
        # if the list becomes very large. You can also use a set.
        allowed_methods = ['linear', 'allometric']

        if self.model_method not in allowed_methods:
            raise ValueError(f"""Method '{self.model_method.__name__}' is not an allowed method. Allowed methods are: {
                             [method.__name__ for method in allowed_methods]}""")

def sum_of_squares_loss(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)

def mean_squared_error_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def huber_loss(y_true, y_pred, delta=1.0):
    resid = y_pred - y_true
    loss = huber(delta, resid)
    return np.mean(loss)
class OneCompartmentModel(RegressorMixin, BaseEstimator):

    def __init__(
        self,
        groupby_col: str = 'SUBJID',
        conc_at_time_col: str = 'DV',
        time_col='TIME',
        pk_model_function=one_compartment_model,
        population_coeff: List[PopulationCoeffcient] = [
            PopulationCoeffcient('k', 0.6), PopulationCoeffcient('vd', 2.0)],
        dep_vars: Dict[str, ObjectiveFunctionColumn] = {'k': [ObjectiveFunctionColumn('mgkg'), ObjectiveFunctionColumn('age')],
                                                        'vd': [ObjectiveFunctionColumn('mgkg'), ObjectiveFunctionColumn('age')]},
        loss_function = mean_squared_error,
        loss_params = {},
        optimizer_tol = None,
        verbose=False

    ):
        self.groupby_col = groupby_col
        self.conc_at_time_col = conc_at_time_col
        self.time_col = time_col
        self.verbose = verbose
        self.pk_model_function = pk_model_function
        self.loss_function = loss_function
        # not sure if the section below needs to be in two places
        for coef_obj in population_coeff:
            c = coef_obj.coeff_name
            if c not in (i for i in dep_vars):
                dep_vars[c] = []
        assert [i.coeff_name for i in population_coeff] == [
            i for i in dep_vars]
        # END section
        self.population_coeff = population_coeff
        self.dep_vars = dep_vars
        self.init_vals = self._unpack_init_vals()
        self.bounds = self._unpack_upper_lower_bounds()
        self.loss_params = loss_params
        self.optimzer_tol = optimizer_tol

    def _unpack_init_vals(self,):
        #unpack the population coeffs
        init_vals = [
            obj.optimization_init_val for obj in self.population_coeff]
        init_vals_pd = []
        for pop_coeff in self.population_coeff:
            init_vals_pd.append({
                'model_coeff': pop_coeff.coeff_name,
                'model_coeff_dep_var':None,
                'population_coeff':True, 
                'init_val':pop_coeff.optimization_init_val,
                'allometric':False,
                'allometric_norm_value':None,
                'subject_level_intercept':pop_coeff.subject_level_intercept
            })
        #unpack the dep vars for the population coeffs
        for model_coeff in self.dep_vars:
            coeff_dep_vars = self.dep_vars[model_coeff]
            init_vals.extend([coeff_obj.optimization_init_val for coeff_obj in coeff_dep_vars
                              if isinstance(coeff_obj, ObjectiveFunctionColumn)])
            for coeff_dep_var in coeff_dep_vars:
                init_vals_pd.append({
                    'model_coeff': model_coeff,
                    'model_coeff_dep_var': coeff_dep_var.column_name,
                    'population_coeff':False, 
                    'init_val':coeff_dep_var.optimization_init_val,
                    'allometric': True if coeff_dep_var.model_method == 'allometric' else False,
                    'allometric_norm_value':coeff_dep_var.allometric_norm_value, 
                    'subject_level_intercept':False
                })
        self.init_vals_pd = pd.DataFrame(init_vals_pd)
        self.n_optimized_coeff = len(init_vals)
        return np.array(init_vals, dtype = np.float64)
    
    def _unpack_upper_lower_bounds(self,):
        bounds = [(obj.optimization_lower_bound, obj.optimization_upper_bound)
                  for obj in self.population_coeff]
        
        for model_coeff in self.dep_vars:
            coeff_dep_vars = self.dep_vars[model_coeff]
            bounds.extend([(obj.optimization_lower_bound, obj.optimization_upper_bound) for obj in coeff_dep_vars
                           if isinstance(obj, ObjectiveFunctionColumn)])
        return deepcopy(bounds)

    def _unpack_validate_params(self, params):
        population_coeff = deepcopy(self.population_coeff)
        dep_vars = deepcopy(self.dep_vars)
        for coeff_obj in population_coeff:
            if coeff_obj.coeff_name not in (i for i in dep_vars):
                dep_vars[coeff_obj.coeff_name] = []
        # Ensure the dep vars correspond to the pop coeff in the correct order
        assert [i.coeff_name for i in population_coeff] == [
            i for i in dep_vars]
        # change how this in unpacked to be based on the length of `population_coeff`
        # population_coeff_alt = {}  # overwrite the list with a dict in the same order
        for idx, coeff_obj in enumerate(population_coeff):
            coeff_obj.optimization_history.append(params[idx])
        self.population_coeff = deepcopy(population_coeff)
        self.dep_vars = deepcopy(dep_vars)

    
    def _populate_model_betas(self, other_params):
        self.n_model_vars = len([i for i in self.dep_vars])
        self.n_dep_vars_per_model_var = {i:len(self.dep_vars[i]) for i in self.dep_vars}
        dep_vars = self.dep_vars
        betas = {}
        betas_np = {}
        params = []
        other_params_idx = 0
        for model_param in dep_vars:
            # beta_names.extend([f'{model_param}_i' for i in dep_vars[model_param]])
            betas[model_param] = {}
            betas_np[model_param] = {}
            for param_col_obj in dep_vars[model_param]:
                param_col = param_col_obj.column_name
                betas[model_param][param_col] = ObjectiveFunctionBeta(column_name=param_col,
                                                                      model_method=param_col_obj.model_method,
                                                                      value=other_params[other_params_idx],
                                                                      allometric_norm_value=param_col_obj.allometric_norm_value)
                param_col_obj.optimization_history.append(
                    other_params[other_params_idx])
                params.append(param_col)
                other_params_idx = other_params_idx + 1
        self.betas = deepcopy(betas)
        self.dep_vars = deepcopy(self.dep_vars)
        return deepcopy(betas)

    def _pymc_model(self, data):
        params = self.init_vals
        self._unpack_validate_params(params)

        
        
        with pm.Model() as model:
            # Priors for population parameters
            pm_pop_coeff = {}
            for pop_coeff in self.population_coeff:
                #k_pop = pm.Normal("k_pop", mu=0, sigma=1)  # Adjust priors as needed
                #vd_pop = pm.Normal("vd_pop", mu=0, sigma=1)
                pm_pop_coeff[pop_coeff.coeff_name] = pm.Normal(f"{pop_coeff.coeff_name}_pop", mu=0, sigma=1)


            # Model betas (coefficients for dependent variables)
            pm_betas = {}
            for model_param in self.dep_vars:
                pm_betas[model_param] = {}
                for param_col_obj in self.dep_vars[model_param]:
                    param_col = param_col_obj.column_name
                    pm_betas[model_param][param_col] = pm.Normal(f"beta_{model_param}_{param_col}", mu=0, sigma=1)

            # Calculate subject-specific parameters
            subject_params = {}
            for subject in data[self.groupby_col].unique():
                subject_data = data[data[self.groupby_col] == subject].iloc[0]
                subject_params[subject] = {}
                for model_param in self.dep_vars:
                    param_value = subject_params[subject][model_param] = deepcopy(pm_pop_coeff[model_param])
                    for param_col_obj in self.dep_vars[model_param]:
                        param_col = param_col_obj.column_name
                        beta = pm_betas[model_param][param_col]
                        covariate_value = subject_data[param_col]
                        param_value += beta * covariate_value
                
                    subject_params[subject][model_param] = pm.math.exp(subject_params[subject][model_param]) + 1e-6
                    #subject_params[subject]['vd'] = pm.math.exp(subject_params[subject]['vd']) + 1e-6
            
            # Define the likelihood
            y_pred = []
            for subject in data[self.groupby_col].unique():
                subject_data = data[data[self.groupby_col] == subject]
                initial_conc = subject_data[self.conc_at_time_col].values[0]

                sol = solve_ivp(self.pk_model_function,
                                [subject_data[self.time_col].min(), subject_data[self.time_col].max()],
                                [initial_conc],
                                t_eval=subject_data[self.time_col],
                                args=(subject_params[subject]['k'], subject_params[subject]['vd'], 1))
                
                y_pred_subject = pm.Normal("y_pred_subject", mu=sol.y[0], sigma=0.1, observed=subject_data[self.conc_at_time_col])
                y_pred.append(y_pred_subject)

        return model



    def fit_pymc(self, data, **kwargs):
        model = self._pymc_model(data)
        with model:
            self.trace_ = pm.sample(**kwargs)  # Perform sampling
            self.idata_ = az.from_pymc3(self.trace_) # Convert to InferenceData object for ArviZ analysis


        # Extract posterior means for predictions (example)
        self.posterior_means_ = {
            param: self.trace_[param].mean(axis=0) for param in self.trace_.varnames
        }

        return self

    
    def _subject_iterator(self, data):
        #data_out = {}
        subject_id_c = self.groupby_col
        #data_out['subject_id_c'] = deepcopy(self.groupby_col)
        conc_at_time_c = self.conc_at_time_col
        #data_out['conc_at_time_c'] = deepcopy(self.conc_at_time_col)
        #data_out['pk_model_function'] = deepcopy(self.pk_model_function)
        pk_model_function = deepcopy(self.pk_model_function)
        verbose = self.verbose
        
        population_coeff = deepcopy(self.population_coeff)
        #data_out['betas'] = deepcopy(self.betas)
        #data_out['time_c'] = deepcopy(self.time_col)
        betas = deepcopy(self.betas)
        time_col = deepcopy(self.time_col)
        subs = data[subject_id_c].unique()
        for subject in subs:
            data_out = {}
            data_out['subject_id_c'] = subject_id_c 
            data_out['conc_at_time_c'] = conc_at_time_c
            data_out['pk_model_function'] = pk_model_function
            data_out['time_c'] = time_col
            data_out['betas'] = deepcopy(betas)
            subject_filt = data[subject_id_c] == subject
            subject_data = data.loc[subject_filt, :].copy()
            initial_conc = subject_data[conc_at_time_c].values[0]
            subject_coeff = deepcopy(population_coeff)
            #subject_coeff = deepcopy(population_coeff)
            subject_coeff = {
                obj.coeff_name: obj.optimization_history[-1] for obj in subject_coeff}
            #subject_coeff_history = [subject_coeff]
            data_out['subject_coeff'] = deepcopy(subject_coeff)
            data_out['subject_data'] = deepcopy(subject_data)
            data_out['initial_conc'] = deepcopy(initial_conc)
            yield deepcopy(data_out)
    
    def _assemble_pred_matrices(self, data):
        subject_id_c = self.groupby_col
        #data_out['subject_id_c'] = deepcopy(self.groupby_col)
        conc_at_time_c = self.conc_at_time_col
        #data_out['conc_at_time_c'] = deepcopy(self.conc_at_time_col)
        #data_out['pk_model_function'] = deepcopy(self.pk_model_function)
        pk_model_function = deepcopy(self.pk_model_function)
        verbose = self.verbose
        
        population_coeff = deepcopy(self.population_coeff)
        #data_out['betas'] = deepcopy(self.betas)
        #data_out['time_c'] = deepcopy(self.time_col)
        betas = deepcopy(self.betas)
        time_col = deepcopy(self.time_col)
        subs = data[subject_id_c].unique()
        n_subs = len(subs)
        init_vals = self.init_vals_pd.copy()
        init_vals['init_val'] = init_vals['init_val'].fillna(0.0)
        model_params = init_vals.loc[init_vals['population_coeff'], :]
        model_param_dep_vars = init_vals.loc[init_vals['population_coeff'] == False, :]
        seen_coeff = []
        betas = pd.DataFrame()
        beta_data = pd.DataFrame()
        for idx, row in model_param_dep_vars.iterrows():
            coeff_name = row['model_coeff']
            beta_name = row['model_coeff_dep_var']
            betas[(coeff_name, beta_name)] = row['init_val']
            beta_data[(coeff_name, beta_name)] = data[beta_name].values
        pop_coeff = pd.DataFrame()
        for idx, row in model_params.iterrows():
            coeff_name = row['model_coeff']
            pop_coeff[coeff_name] = row['init_val']
            #betas[coeff_name] = row['init_val']
        return pop_coeff, betas, beta_data
    
   # def _predict_vectorized(self, pop_coeff, betas, beta_data):
   #     for c in pop_coeff.columns:
   #         theta = beta_data
            
    
    def _predict_inner(self, data_out):
        subject_data = data_out['subject_data']
        initial_conc = data_out['initial_conc']
        subject_coeff = data_out['subject_coeff']
        betas = data_out['betas']
        time_c = data_out['time_c']
        pk_model_function = data_out['pk_model_function']
        #conc_at_time_c = data_out['conc_at_time_c']
        allometric_effects = []
        for model_param in betas:
            for param_col in betas[model_param]:
                param_beta_obj = betas[model_param][param_col]
                param_beta = param_beta_obj.value
                param_beta_method = param_beta_obj.model_method
                param_value = subject_data[param_col].values[0]
                if param_beta_method == 'linear':
                    subject_coeff[model_param] = subject_coeff[model_param] + \
                        (param_beta*param_value)
                   # subject_coeff_history.append(subject_coeff)
                elif param_beta_method == 'allometric':
                    norm_val = param_beta_obj.allometric_norm_value
                    param_value = 1e-6 if param_value == 0 else param_value
                    norm_val = 1e-6 if norm_val == 0 else norm_val
                    allometric_effects.append(
                        # (param_value/70)**(param_beta)
                        np.sign(param_value/norm_val) * \
                        ((np.abs(param_value/norm_val))**param_beta)
                    )
            for allometic_effect in allometric_effects:
                subject_coeff[model_param] = subject_coeff[model_param] * \
                    allometic_effect
                #subject_coeff_history.append(subject_coeff)
        #subject_coeffs_history[subject] = subject_coeff_history
        subject_coeff = {model_param: np.exp(
            subject_coeff[model_param]) for model_param in subject_coeff}
        subject_coeff = np.array([subject_coeff[i] for i in subject_coeff])+1e-6
        #subject_coeff[subject_coeff < 1e-6] = 1e-6
        subject_coeff = subject_coeff.tolist()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                sol = solve_ivp(pk_model_function, np.array([subject_data[time_c].min(), subject_data[time_c].max()], dtype = np.float64),
                            np.array([initial_conc], dtype = np.float64),
                            t_eval=np.array(subject_data[time_c], dtype = np.float64), args=(*subject_coeff, 1 ))
        except (ZeroDivisionError, RuntimeWarning) as e:
            self.ivp_error_params_ = {
            'f':deepcopy(pk_model_function),
            'tspan': np.array([subject_data[time_c].min(), subject_data[time_c].max()]),
            'init_conc':np.array([initial_conc]),
            't_eval':np.array(subject_data[time_c]), 
            'args': "args=(*subject_coeff, 1 )",
            'args0':subject_coeff
            
            }
            raise ZeroDivisionError
        return sol.y[0].astype(np.float64)
    
    def _predict_parallel(self, data, parallel_n_jobs = -1):
        predictions = Parallel(n_jobs=parallel_n_jobs)(delayed(self._predict_inner)(data_out) for data_out in self._subject_iterator(data))
        return np.concatenate(predictions)
    
    
    
    def _predict_alt(self, data):
        predictions = []
        for subject_data in self._subject_iterator(data):
            subject_preds = self._predict_inner(subject_data)
            predictions.extend(subject_preds)
        return predictions
            
    
    def predict(self, data, parallel = True, parallel_n_jobs = -1, return_df=False):
        if parallel:
            predictions = self._predict_parallel(data, parallel_n_jobs=parallel_n_jobs)
        else:
            predictions = self._predict_alt(data)
        if return_df:
            data['pred_y'] = predictions
            predictions = data.copy()
        return predictions
    # def score(self, y_true, y_pred, sample_weight=None, multioutput='uniform_average', method = mean_squared_error, *kwargs):
    #    return method(y_true, y_pred, sample_weight, multioutput, *kwargs)

    def _objective_function(self, params, data, parallel = False, parallel_n_jobs = -1):
        params = np.array(params, dtype = np.float64)
        self._unpack_validate_params(params)
        n_pop_coeff = len(self.population_coeff)
        other_params = params[n_pop_coeff:]
        _betas = self._populate_model_betas(other_params)
        #preds = np.concatenate(self._predict_parallel(data = data)) if parallel else self._predict_alt(data=data)
        preds = self.predict(data, parallel = parallel, parallel_n_jobs = parallel_n_jobs)
        #residuals = data[self.conc_at_time_col] - preds
        #sse = np.sum(residuals**2)
        error = self.loss_function(data[self.conc_at_time_col], preds, **self.loss_params)
        return error

    def fit(self, data, parallel = False, parallel_n_jobs = -1 , warm_start = False, checkpoint_filename='check_test.jb'):
        # bounds = [(None, None) for i in range(len(self.dep_vars) + len(self.population_coeff))]
        objective_function = partial(self._objective_function, parallel = parallel, parallel_n_jobs = parallel_n_jobs)
        self.fit_result_ = optimize_with_checkpoint_joblib(objective_function,
                                                           self.init_vals,
                                                           n_checkpoint=5,
                                                           checkpoint_filename=checkpoint_filename,
                                                           args=(data,),
                                                           warm_start=warm_start,
                                                           tol = self.optimzer_tol,
                                                           bounds=self.bounds
                                                           )
        res_df = []
        written_indep_var_rows = 0
        for idx, model_coeff_obj in enumerate(self.population_coeff):
            fit_res_0 = self.fit_result_['x'][idx]
            fit_res_1 = model_coeff_obj.optimization_history[-1]
            # assert fit_res_0 == fit_res_1
            res_df.append(
                {
                    'population_coeff': True,
                    'model_coeff': model_coeff_obj.coeff_name,
                    'model_coeff_indep_var': None,
                    'log_coeff': fit_res_0,
                    'log_coeff_history_final': fit_res_1,
                    'coeff_estimates_equal': fit_res_0 == fit_res_1,
                    'coeff': np.exp(fit_res_0)
                }
            )
            written_indep_var_rows = written_indep_var_rows + 1

        for idx, model_coeff_name in enumerate(self.dep_vars):
            coeff_indep_vars = self.dep_vars[model_coeff_name]
            for indep_var_obj in coeff_indep_vars:
                fit_res_0 = self.fit_result_['x'][written_indep_var_rows]
                fit_res_1 = indep_var_obj.optimization_history[-1]
                # assert fit_res_0 == fit_res_1
                res_df.append({
                    'population_coeff': False,
                    'model_coeff': model_coeff_name,
                    'model_coeff_indep_var': indep_var_obj.column_name,
                    'log_coeff': fit_res_0,
                    'log_coeff_history_final': fit_res_1,
                    'coeff_estimates_equal': fit_res_0 == fit_res_1,
                    'coeff': np.exp(fit_res_0)
                }
                )
                written_indep_var_rows = written_indep_var_rows + 1
        self.fit_result_summary_ = pd.DataFrame(res_df)

        return deepcopy(self)


def arbitrary_objective_function(params, data, model_function=one_compartment_model, subject_id_c='SUBJID', conc_at_time_c='DV',
                                 time_c='TIME', population_coeff=['k', 'vd'],
                                 dep_vars={'k': [ObjectiveFunctionColumn('mgkg'), ObjectiveFunctionColumn('age')],
                                           'vd': [ObjectiveFunctionColumn('mgkg'), ObjectiveFunctionColumn('age')]},
                                 verbose=False,
                                 parallel=False
                                 ):
    for c in population_coeff:
        if c not in (i for i in dep_vars):
            dep_vars[c] = []
    assert population_coeff == [i for i in dep_vars]
    # change how this in unpacked to be based on the length of `population_coeff`
    k_pop, Vd_pop, *other = params
    population_coeff = {}  # overwrite the list with a dict in the same order
    for idx, coeff in enumerate(dep_vars):
        population_coeff[coeff] = params[idx]
    betas = {}
    params = []
    other_params_idx = 0
    for model_param in dep_vars:
        # beta_names.extend([f'{model_param}_i' for i in dep_vars[model_param]])
        betas[model_param] = {}
        for param_col_obj in dep_vars[model_param]:
            param_col = param_col_obj.column_name
            betas[model_param][param_col] = ObjectiveFunctionBeta(column_name=param_col, model_method=param_col_obj.model_method,
                                                                  value=other[other_params_idx],
                                                                  allometric_norm_value=param_col_obj.allometric_norm_value)
            params.append(param_col)
            other_params_idx = other_params_idx + 1
    subject_coeffs_history = {}
    predictions = []
    subject_iter_obj = tqdm(data[subject_id_c].unique(
    )) if verbose else data[subject_id_c].unique()
    for subject in subject_iter_obj:
        subject_filt = data[subject_id_c] == subject
        subject_data = data.loc[subject_filt, :].copy()
        initial_conc = subject_data[conc_at_time_c].values[0]
        subject_coeff = deepcopy(population_coeff)
        subject_coeff_history = [subject_coeff]
        allometric_effects = []
        for model_param in betas:  # for each of the coeff to be input to the model
            # for each of the columns in the data which contribute to `model_param`
            for param_col in betas[model_param]:
                param_beta_obj = betas[model_param][param_col]
                param_beta = param_beta_obj.value
                param_beta_method = param_beta_obj.model_method
                param_value = subject_data[param_col].values[0]
                if param_beta_method == 'linear':
                    subject_coeff[model_param] = subject_coeff[model_param] + \
                        (param_beta*param_value)
                    subject_coeff_history.append(subject_coeff)
                elif param_beta_method == 'allometric':
                    norm_val = param_beta_obj.allometric_norm_value
                    param_value = 1e-6 if param_value == 0 else param_value
                    norm_val = 1e-6 if norm_val == 0 else norm_val
                    allometric_effects.append(
                        # (param_value/70)**(param_beta)
                        np.sign(param_value/norm_val) * \
                        ((np.abs(param_value/norm_val))**param_beta)
                    )
            for allometic_effect in allometric_effects:
                subject_coeff[model_param] = subject_coeff[model_param] * \
                    allometic_effect
                subject_coeff_history.append(subject_coeff)
        subject_coeffs_history[subject] = subject_coeff_history
        subject_coeff = {model_param: np.exp(
            subject_coeff[model_param]) for model_param in subject_coeff}
        subject_coeff = [subject_coeff[i] for i in subject_coeff]
        sol = solve_ivp(model_function, [subject_data[time_c].min(), subject_data[time_c].max()],
                        [initial_conc],
                        t_eval=subject_data[time_c], args=(*subject_coeff, 1))
        predictions.extend(sol.y[0])
    # Calculate the difference between observed and predicted values
    residuals = data[conc_at_time_c] - predictions
    sse = np.sum(residuals**2)  # Calculate the sum of squared errors
    return sse

# j = arbitrary_objective_function(params = (1,2,3,4,5,6), data = pd.DataFrame())


def objective_function(params, data, subject_id_c='SUBJID', dose_c='DOSR', time_c='TIME', conc_at_time_c='DV'):
    """
    Calculates the sum of squared errors (SSE) between observed and predicted drug 
    concentrations.

    This function simulates drug concentrations for each subject in the dataset using 
    a one-compartment model and compares the predictions to the actual observations. 
    The SSE is used as a measure of the goodness of fit for the given model parameters.

    Args:
      params (tuple): Tuple containing the model parameters (k, Vd).
      data (DataFrame): Pandas DataFrame containing the pharmacokinetic data, with columns
                        for 'SUBJID', 'DOSR', 'DV' (observed concentration), and 'TIME'.

    Returns:
      float: The sum of squared errors (SSE).
    """
    k, Vd = params  # Unpack parameters
    # Vd = Vd + 1e-6 if Vd == 0 else Vd  # Add a small value to Vd to avoid division by zero (commented out)
    predictions = []
    # Loop through each subject in the dataset
    for subject in tqdm(data[subject_id_c].unique()):
        # Extract dose information for the subject
        d = data.loc[data[subject_id_c] == subject, dose_c]
        d = d.drop_duplicates()  # Ensure only one dose value is used
        dose = d.values[0]  # Get the dose value
        # Get data for the current subject
        subject_data = data[data[subject_id_c] == subject]
        # Get the initial concentration
        initial_conc = subject_data[conc_at_time_c].values[0]

        # Solve the differential equation for the current subject
        sol = solve_ivp(one_compartment_model, [subject_data[time_c].min(), subject_data[time_c].max()], [initial_conc],
                        t_eval=subject_data[time_c], args=(k, Vd, dose))

        # Add the predictions for this subject to the list
        predictions.extend(sol.y[0])

    # Calculate the difference between observed and predicted values
    residuals = data[conc_at_time_c] - predictions
    sse = np.sum(residuals**2)  # Calculate the sum of squared errors
    return sse


def objective_function__mgkg_age(params, data, subject_id_c='SUBJID', dose_c='DOSR', time_c='TIME', conc_at_time_c='DV', mgkg_c='MGKG', age_c='AGE'):
    """
    Calculates the sum of squared errors (SSE) between observed and predicted drug 
    concentrations.

    This function simulates drug concentrations for each subject in the dataset using 
    a one-compartment model and compares the predictions to the actual observations. 
    The SSE is used as a measure of the goodness of fit for the given model parameters.

    Args:
    params (tuple): Tuple containing the model parameters (k, Vd).
    data (DataFrame): Pandas DataFrame containing the pharmacokinetic data, with columns
                        for 'SUBJID', 'DOSR', 'DV' (observed concentration), and 'TIME'.

    Returns:
    float: The sum of squared errors (SSE).
    """
    k_pop, Vd_pop, k_beta_age, k_beta_mgkg, Vd_beta_age, Vd_beta_mgkg = params  # Unpack parameters
    # Vd = Vd + 1e-6 if Vd == 0 else Vd  # Add a small value to Vd to avoid division by zero (commented out)
    predictions = []
    # Loop through each subject in the dataset
    for subject in data[subject_id_c].unique():
        subject_filt = data[subject_id_c] == subject
        subject_data = data.loc[subject_filt, :].copy()

        # Extract dose information for the subject
        mgkg = subject_data[mgkg_c].values[0]
        age = subject_data[age_c].values[0]
        # Get data for the current subject
        # Get the initial concentration
        initial_conc = subject_data[conc_at_time_c].values[0]
        with np.errstate(over='ignore'):
            k_i = np.exp(k_pop + (k_beta_age * age) + (k_beta_mgkg * mgkg))
            Vd_i = np.exp(Vd_pop + (Vd_beta_age * age) + (Vd_beta_mgkg * mgkg))
        # Solve the differential equation for the current subject
        sol = solve_ivp(one_compartment_model, [subject_data[time_c].min(), subject_data[time_c].max()], [initial_conc],
                        t_eval=subject_data[time_c], args=(k_i, Vd_i, mgkg))

        # Add the predictions for this subject to the list
        predictions.extend(sol.y[0])

    # Calculate the difference between observed and predicted values
    residuals = data[conc_at_time_c] - predictions
    sse = np.sum(residuals**2)  # Calculate the sum of squared errors
    return sse


def optimize_with_checkpoint_joblib(func, x0, n_checkpoint, checkpoint_filename, *args, warm_start = False, tol = None, **kwargs):
    """
    Optimizes a function using scipy.optimize.minimize() with checkpointing every n iterations,
    using joblib for saving and loading checkpoints.

    Args:
        func: The objective function to be minimized.
        x0: The initial guess.
        n_checkpoint: The number of iterations between checkpoints.
        checkpoint_filename: The filename to save checkpoints to.
        *args: Additional positional arguments to be passed to minimize().
        **kwargs: Additional keyword arguments to be passed to minimize().

    Returns:
        The optimization result from scipy.optimize.minimize().
    """

    iteration = 0

    # Try to load a previous checkpoint if it exists
    check_name = checkpoint_filename.replace('.jb', '')
    if not os.path.exists('logs'):
        os.mkdir('logs')    
    checkpoints = [i.replace('.jb', '') for i in os.listdir('logs') if check_name in i]
    max_check_idx = [i.split('__')[-1] for i in checkpoints if len(i.split('__')) > 1]
    try:
        max_check_idx = [int(i) for i in max_check_idx]
    except ValueError:
        max_check_idx = []
    max_check_idx = f"__{np.max(max_check_idx)}" if len(max_check_idx) > 1 else ''
    checkpoint_filename = os.path.join('logs',check_name + f'{max_check_idx}.jb')
    if warm_start:
        try:
            checkpoint = load(checkpoint_filename)
            x0 = checkpoint['x']
            iteration = checkpoint['iteration']
            print(f"Resuming optimization from iteration {iteration}")
        except FileNotFoundError:
            print("No checkpoint found, starting from initial guess.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}, starting from initial guess.")

    def callback_with_checkpoint(xk, checkpoint_filename):
        nonlocal iteration
        iteration += 1
        #print(iteration)
        if iteration % n_checkpoint == 0:
            checkpoint = {
                'x': xk,
                'iteration': iteration
            }
            checkpoint_filename = checkpoint_filename.replace(
                '.jb', f'__{iteration}.jb')
            dump(checkpoint, checkpoint_filename)
            print(f"Iteration {iteration}: Checkpoint saved to {
                  checkpoint_filename}")
        #print('no log')

    # Ensure callback is not already in kwargs
    if 'callback' not in kwargs:
        kwargs['callback'] = partial(
            callback_with_checkpoint, checkpoint_filename=checkpoint_filename)
    else:
        # If callback exists, combine it with the existing one
        user_callback = kwargs['callback']

        def combined_callback(xk):
            callback_with_checkpoint(xk, checkpoint_filename)
            user_callback(xk)

        kwargs['callback'] = combined_callback

    result = minimize(func, x0, *args, tol = tol, **kwargs)

    # Remove checkpoint file at end.
    # try:
    #    os.remove(checkpoint_filename)
    # except:
    #    pass

    return result


def stack_ivp_predictions(ivp_predictions, time_c='TIME', pred_DV_c='Pred_DV', subject_id_c='SUBJID'):
    dfs = []
    for subject in ivp_predictions:
        loop_df = pd.DataFrame()
        time_vector = ivp_predictions[subject].t
        preds_vector = ivp_predictions[subject].y[0]
        loop_df[time_c] = time_vector
        loop_df[pred_DV_c] = preds_vector
        loop_df[subject_id_c] = subject
        dfs.append(loop_df)
    return pd.concat(dfs)


def merge_ivp_predictions(df, ivp_predictions, time_c='TIME', pred_DV_c='Pred_DV', subject_id_c='SUBJID'):
    df = df.copy()
    result_df = stack_ivp_predictions(
        ivp_predictions, time_c, pred_DV_c, subject_id_c)
    merge_df = df.merge(result_df, how='left', on=[subject_id_c, time_c])
    return merge_df


def generate_ivp_predictions(optimized_result, df, subject_id_c='SUBJID', dose_c='DOSR', time_c='TIME', conc_at_time_c='DV'):
    predictions = {}
    est_k, est_vd = optimized_result.x
    data = df.copy()
    for subject in data[subject_id_c].unique():
        d = data.loc[data[subject_id_c] == subject, dose_c]
        d = d.drop_duplicates()
        dose = d.values[0]
        subject_data = data[data[subject_id_c] == subject]

        initial_conc = subject_data[conc_at_time_c].values[0]
        # the initial value is initial_conc in this setup. If absorbtion was being modeled it would be [dose/est_vd]
        sol = solve_ivp(one_compartment_model, [subject_data[time_c].min(), subject_data[time_c].max()], [initial_conc],
                        t_eval=subject_data[time_c], args=(est_k, est_vd, dose))
        predictions[subject] = sol
    return predictions
