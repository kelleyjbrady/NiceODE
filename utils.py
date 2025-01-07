from scipy.integrate import solve_ivp
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from joblib import dump, load
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_subject_levels(df:pd.DataFrame,x = 'TIME' ,y = 'DV', subject = 'SUBJID', ax = None):
    if ax is None:
        fig, ax = plt.subplots(1)
    for c in df.columns:
        df.loc[df[c] == '.', c] = pd.NA
    df[y] = df[y].astype(pd.Float32Dtype())
    df[x] = df[x].astype(pd.Float32Dtype())
    sns.lineplot(data = df, x = x, y = y, hue = subject, ax = ax)

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

def objective_function(params, data, subject_id_c = 'SUBJID', dose_c = 'DOSR', time_c = 'TIME', conc_at_time_c = 'DV'):
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
  #Vd = Vd + 1e-6 if Vd == 0 else Vd  # Add a small value to Vd to avoid division by zero (commented out)
  predictions = []
  for subject in tqdm(data[subject_id_c].unique()):  # Loop through each subject in the dataset
      d = data.loc[data[subject_id_c] == subject, dose_c]  # Extract dose information for the subject
      d = d.drop_duplicates()  # Ensure only one dose value is used
      dose = d.values[0]  # Get the dose value
      subject_data = data[data[subject_id_c] == subject]  # Get data for the current subject
      initial_conc = subject_data[conc_at_time_c].values[0]  # Get the initial concentration

      # Solve the differential equation for the current subject
      sol = solve_ivp(one_compartment_model, [subject_data[time_c].min(), subject_data[time_c].max()], [initial_conc], 
                      t_eval=subject_data[time_c], args=(k, Vd, dose))
      
      predictions.extend(sol.y[0])  # Add the predictions for this subject to the list

  residuals = data[conc_at_time_c] - predictions  # Calculate the difference between observed and predicted values
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
            checkpoint_filename = checkpoint_filename.replace('.jb', f'_{iteration}.jb')
            dump(checkpoint, checkpoint_filename)
            print(f"Iteration {iteration}: Checkpoint saved to {checkpoint_filename}")
        print('no log')

    # Ensure callback is not already in kwargs
    if 'callback' not in kwargs:
        kwargs['callback'] = partial(callback_with_checkpoint, checkpoint_filename = checkpoint_filename)
    else:
        # If callback exists, combine it with the existing one
        user_callback = kwargs['callback']
        def combined_callback(xk):
            callback_with_checkpoint(xk, checkpoint_filename)
            user_callback(xk)

        kwargs['callback'] = combined_callback
    
    result = minimize(func, x0, *args, **kwargs)
    
    # Remove checkpoint file at end.
    #try:
    #    os.remove(checkpoint_filename)
    #except:
    #    pass

    return result

def stack_ivp_predictions(ivp_predictions, time_c = 'TIME', pred_DV_c = 'Pred_DV', subject_id_c = 'SUBJID'):
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

def merge_ivp_predictions(df, ivp_predictions, time_c = 'TIME', pred_DV_c = 'Pred_DV', subject_id_c = 'SUBJID'):
    df = df.copy()
    result_df = stack_ivp_predictions(ivp_predictions, time_c, pred_DV_c, subject_id_c)
    merge_df = df.merge(result_df, how = 'left', on = [subject_id_c, time_c])
    return merge_df

def generate_ivp_predictions(optimized_result, df, subject_id_c = 'SUBJID', dose_c = 'DOSR', time_c = 'TIME', conc_at_time_c = 'DV'):
    predictions = {}
    est_k, est_vd = optimized_result.x
    data = df.copy()
    for subject in data[subject_id_c].unique():
        d = data.loc[data[subject_id_c] == subject, dose_c]
        d =  d.drop_duplicates()
        dose = d.values[0]
        subject_data = data[data[subject_id_c] == subject]

        initial_conc = subject_data[conc_at_time_c].values[0]
        #the initial value is initial_conc in this setup. If absorbtion was being modeled it would be [dose/est_vd]
        sol = solve_ivp(one_compartment_model, [subject_data[time_c].min(), subject_data[time_c].max()], [initial_conc],
                        t_eval=subject_data[time_c], args=(est_k, est_vd, dose))
        predictions[subject] = sol
    return predictions