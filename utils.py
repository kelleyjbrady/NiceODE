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
from dataclasses import dataclass
from sklearn.base import BaseEstimator, RegressorMixin


def plot_subject_levels(df: pd.DataFrame, x='TIME', y='DV', subject='SUBJID', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    for c in df.columns:
        df.loc[df[c] == '.', c] = pd.NA
    df[y] = df[y].astype(pd.Float32Dtype())
    df[x] = df[x].astype(pd.Float32Dtype())
    sns.lineplot(data=df, x=x, y=y, hue=subject, ax=ax)


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
    allometric_norm_value:float = None
    model_method: str = 'linear'
    def __post_init__(self):
        # Define the allowed methods. Using a dictionary for better lookup performance
        # if the list becomes very large. You can also use a set.
        allowed_methods = ['linear', 'allometric']

        if self.model_method not in allowed_methods:
            raise ValueError(f"""Method '{self.model_method.__name__}' is not an allowed method. Allowed methods are: {
                             [method.__name__ for method in allowed_methods]}""")
        if self.model_method == 'allometric' and self.allometric_norm_value is None:
            raise ValueError(f"""Method '{self.model_method.__name__}' is not allowed without providing the `allometric_norm_value`""")

@dataclass 
class ObjectiveFunctionBeta:
    column_name:str
    value:np.float64
    allometric_norm_value:float = None
    model_method: str = 'linear'
    def __post_init__(self):
        # Define the allowed methods. Using a dictionary for better lookup performance
        # if the list becomes very large. You can also use a set.
        allowed_methods = ['linear', 'allometric']

        if self.model_method not in allowed_methods:
            raise ValueError(f"""Method '{self.model_method.__name__}' is not an allowed method. Allowed methods are: {
                             [method.__name__ for method in allowed_methods]}""")

    
class OneCompartmentModel(RegressorMixin, BaseEstimator):
    
    def __init__(
        self, 
        groupby_col:str = 'SUBJID', 
        conc_at_time_col:str = 'DV', 
        time_c = 'TIME', 
        population_coeff:list = ['k', 'vd'], 
        dep_vars:dict = {'k': [ObjectiveFunctionColumn('mgkg'), ObjectiveFunctionColumn('age')],
                                           'vd': [ObjectiveFunctionColumn('mgkg'), ObjectiveFunctionColumn('age')]}, 
        verbose = False
        
    ):
        self.groupby_col = groupby_col
        self.conc_at_time_col = conc_at_time_col
        self.time_c = time_c
        self.verbose = verbose
        
        for c in population_coeff:
            if c not in (i for i in dep_vars):
                dep_vars[c] = []
        assert population_coeff == [i for i in dep_vars]
        self.population_coeff = population_coeff
        self.dep_vars = dep_vars
        
    def _unpack_validate_params(self,params ):
        population_coeff = deepcopy(self.population_coeff)
        dep_vars = deepcopy(self.dep_vars)
        for c in population_coeff:
            if c not in (i for i in dep_vars):
                dep_vars[c] = []
        assert population_coeff == [i for i in dep_vars]
        # change how this in unpacked to be based on the length of `population_coeff`
        population_coeff = {}  # overwrite the list with a dict in the same order
        for idx, coeff in enumerate(dep_vars):
            population_coeff[coeff] = params[idx]
        self.population_coeff = deepcopy(population_coeff)
        self.dep_vars = deepcopy(dep_vars)
    
    def _populate_model_betas(self, other_params):
        dep_vars = self.dep_vars
        betas = {}
        params = []
        other_params_idx = 0
        for model_param in dep_vars:
            # beta_names.extend([f'{model_param}_i' for i in dep_vars[model_param]])
            betas[model_param] = {}
            for param_col_obj in dep_vars[model_param]:
                param_col = param_col_obj.column_name
                betas[model_param][param_col] =  ObjectiveFunctionBeta(column_name=param_col,
                                                                       model_method=param_col_obj.model_method,
                                                                        value = other_params[other_params_idx],
                                                                        allometric_norm_value=param_col_obj.allometric_norm_value )
                params.append(param_col)
                other_params_idx = other_params_idx + 1
        return deepcopy(betas)
    
    def predict(self, data):
        subject_coeffs_history = {}
        predictions = []
        subject_iter_obj = tqdm(data[subject_id_c].unique()) if verbose else data[subject_id_c].unique()
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
                            #(param_value/70)**(param_beta)                       
                            np.sign(param_value/norm_val) * ((np.abs(param_value/norm_val))**param_beta)
                        )
                for allometic_effect in allometric_effects:
                    subject_coeff[model_param] = subject_coeff[model_param] * allometic_effect
                    subject_coeff_history.append(subject_coeff)
            subject_coeffs_history[subject] = subject_coeff_history
            subject_coeff = {model_param: np.exp(
                subject_coeff[model_param]) for model_param in subject_coeff}
            subject_coeff = [subject_coeff[i] for i in subject_coeff]
            sol = solve_ivp(one_compartment_model, [subject_data[time_c].min(), subject_data[time_c].max()],
                            [initial_conc],
                            t_eval=subject_data[time_c], args=(*subject_coeff, 1))
            predictions.extend(sol.y[0])
        
        
        
       

def arbitrary_objective_function(params, data, subject_id_c='SUBJID', conc_at_time_c='DV',
                                 time_c='TIME', population_coeff=['k', 'vd'],
                                 dep_vars={'k': [ObjectiveFunctionColumn('mgkg'), ObjectiveFunctionColumn('age')],
                                           'vd': [ObjectiveFunctionColumn('mgkg'), ObjectiveFunctionColumn('age')]}, 
                                 verbose = False, 
                                 parallel = False
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
            betas[model_param][param_col] =  ObjectiveFunctionBeta(column_name=param_col, model_method=param_col_obj.model_method,
                                                                   value = other[other_params_idx],
                                                                   allometric_norm_value=param_col_obj.allometric_norm_value )
            params.append(param_col)
            other_params_idx = other_params_idx + 1
    subject_coeffs_history = {}
    predictions = []
    subject_iter_obj = tqdm(data[subject_id_c].unique()) if verbose else data[subject_id_c].unique()
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
                        #(param_value/70)**(param_beta)                       
                        np.sign(param_value/norm_val) * ((np.abs(param_value/norm_val))**param_beta)
                    )
            for allometic_effect in allometric_effects:
                subject_coeff[model_param] = subject_coeff[model_param] * allometic_effect
                subject_coeff_history.append(subject_coeff)
        subject_coeffs_history[subject] = subject_coeff_history
        subject_coeff = {model_param: np.exp(
            subject_coeff[model_param]) for model_param in subject_coeff}
        subject_coeff = [subject_coeff[i] for i in subject_coeff]
        sol = solve_ivp(one_compartment_model, [subject_data[time_c].min(), subject_data[time_c].max()],
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


def optimize_with_checkpoint_joblib(func, x0, n_checkpoint, checkpoint_filename, *args, **kwargs):
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
        print(iteration)
        if iteration % n_checkpoint == 0:
            checkpoint = {
                'x': xk,
                'iteration': iteration
            }
            checkpoint_filename = checkpoint_filename.replace(
                '.jb', f'_{iteration}.jb')
            dump(checkpoint, checkpoint_filename)
            print(f"Iteration {iteration}: Checkpoint saved to {
                  checkpoint_filename}")
        print('no log')

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

    result = minimize(func, x0, *args, **kwargs)

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
