from scipy.integrate import solve_ivp
import numpy as np
from tqdm import tqdm

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

def objective_function(params, data, subject_id = 'SUBJID', dose = 'DOSR', time = 'TIME', conc_at_time = 'DV'):
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
  for subject in tqdm(data[subject_id].unique()):  # Loop through each subject in the dataset
      d = data.loc[data[subject_id] == subject, dose]  # Extract dose information for the subject
      d = d.drop_duplicates()  # Ensure only one dose value is used
      dose = d.values[0]  # Get the dose value
      subject_data = data[data[subject_id] == subject]  # Get data for the current subject
      initial_conc = subject_data[conc_at_time].values[0]  # Get the initial concentration

      # Solve the differential equation for the current subject
      sol = solve_ivp(one_compartment_model, [subject_data[time].min(), subject_data[time].max()], [initial_conc], 
                      t_eval=subject_data[time], args=(k, Vd, dose))
      
      predictions.extend(sol.y[0])  # Add the predictions for this subject to the list

  residuals = data[conc_at_time] - predictions  # Calculate the difference between observed and predicted values
  sse = np.sum(residuals**2)  # Calculate the sum of squared errors
  return sse