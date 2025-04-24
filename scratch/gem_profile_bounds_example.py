import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import chi2
import pandas as pd
from scipy.integrate import solve_ivp

# --- Mock ODE Model and Data (for a runnable example) ---

def first_order_one_compartment_model(t, y, k):
    """Simple 1st-order decay ODE."""
    return -k * y

def generate_mock_data(k_true, sigma_true, timepoints, y0=100):
    """Generates mock data with noise."""
    sol = solve_ivp(first_order_one_compartment_model, [timepoints.min(), timepoints.max()], [y0], args=(k_true,), t_eval=timepoints)
    y_true = sol.y[0]
    y_obs = y_true + np.random.normal(0, sigma_true, size=len(timepoints))
    return y_obs, y_true

# --- Core Functions ---

def neg_log_likelihood_loss(y_true, y_pred, sigma):
    """Calculates -2 * log-likelihood, assuming a normal distribution."""
    residual = y_true - y_pred
    return np.sum((residual / sigma)**2 + 2*np.log(sigma) + np.log(2 * np.pi))

def _objective_function2(params, time, y_obs, ode_model): #Added explicit data arg
    """Objective function for optimization (no mixed effects)."""
    k = params[0]
    sigma = params[1]
    sol = solve_ivp(ode_model, [time.min(), time.max()], [y_obs[0]], args=(k,), t_eval=time)  # Use y_obs[0] as initial condition
    y_pred = sol.y[0]
    return neg_log_likelihood_loss(y_obs, y_pred, sigma)

def profile_likelihood(theta_i, theta_i_value, other_params_initial, time, y_obs, ode_model, ofv_function):
    """Calculates the profile likelihood for a single parameter (no mixed effects)."""

    def objective_to_minimize(other_params):
        all_params = list(other_params_initial)
        all_params.insert(theta_i, theta_i_value)
        return ofv_function(all_params, time, y_obs, ode_model)

    result = minimize(objective_to_minimize, other_params_initial, method='L-BFGS-B', bounds=[(1e-9, None), (1e-9, None)]) # Added bounds here.
    return result.fun

def find_profile_bound(theta_i, mle_params, time, y_obs, ode_model, ofv_function, confidence_level=0.95, lower=True):
    """Finds the profile likelihood confidence interval bound for a single parameter using brentq."""

    ofv_mle = ofv_function(mle_params, time, y_obs, ode_model)
    critical_value = chi2.ppf(confidence_level, df=1)
    threshold = ofv_mle + critical_value

    def root_function(theta_i_value):
        """Function whose root we want to find."""
        other_params_initial = np.delete(mle_params, theta_i)
        ofv_profile = profile_likelihood(theta_i, theta_i_value, other_params_initial, time, y_obs, ode_model, ofv_function)
        return ofv_profile - threshold

    # Determine search interval based on MLE and direction
    mle_value = mle_params[theta_i]
    if lower:
        a = mle_value * 0.1  # Start significantly lower
        b = mle_value       # MLE is the upper bound
        if root_function(a) > 0:
            print(f"Warning: Lower bound not found within initial range for parameter {theta_i}.")
            return None
    else:
        a = mle_value       # MLE is the lower bound
        b = mle_value * 10  # Start significantly higher
        if root_function(b) > 0 :
            print(f"Warning: Upper bound not found within initial range for parameter {theta_i}.")
            return None
    try:
        bound = brentq(root_function, a, b)
        return bound
    except ValueError as e:
        print(f"Warning: Brentq failed to find a bound for parameter {theta_i}: {e}")
        return None


# --- Example Usage ---

# 1. Generate mock data
np.random.seed(42)
timepoints = np.sort(np.random.rand(20) * 10)  # 20 time points between 0 and 10
k_true = 0.5
sigma_true = 2.0
y_obs, y_true = generate_mock_data(k_true, sigma_true, timepoints)
data = pd.DataFrame({'time': timepoints, 'y_obs': y_obs})

# 2. Initial parameter estimates (k, sigma)
initial_params = [0.4, 1.5]

# 3. Fit the model (MLE)
result = minimize(_objective_function2, initial_params, args=(data['time'], data['y_obs'], first_order_one_compartment_model),
                  method='L-BFGS-B', bounds=[(1e-9, None), (1e-9, None)])  # Add bounds for stability
mle_params = result.x
print(f"MLE Parameters: k={mle_params[0]:.4f}, sigma={mle_params[1]:.4f}")


# 4. Profile Likelihood CIs
for i in range(len(mle_params)):
    lower_bound = find_profile_bound(i, mle_params, data['time'], data['y_obs'], first_order_one_compartment_model, _objective_function2, lower=True)
    upper_bound = find_profile_bound(i, mle_params, data['time'], data['y_obs'], first_order_one_compartment_model, _objective_function2, lower=False)
    print(f"Parameter {i} (k if i=0, sigma if i=1): CI = ({lower_bound:.4f}, {upper_bound:.4f})")

# 5. (Optional) Finer Profile for Plotting -  This part remains essentially the same
if lower_bound is not None and upper_bound is not None:
    profile_parameter_values = np.linspace(lower_bound, upper_bound, 20)
    profile_nll_values = []
    for val in profile_parameter_values:
        other_params_initial = np.delete(mle_params, i) # 'i' is defined in the loop above.
        ofv = profile_likelihood(i, val, other_params_initial, data['time'], data['y_obs'], first_order_one_compartment_model, _objective_function2)
        profile_nll_values.append(ofv)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(profile_parameter_values, profile_nll_values, marker='o')
    plt.axhline(result.fun + chi2.ppf(0.95, df=1), color='r', linestyle='--', label='95% CI Threshold')
    plt.axvline(lower_bound, color='g', linestyle=':', label='Lower Bound')
    plt.axvline(upper_bound, color='g', linestyle=':', label='Upper Bound')
    plt.xlabel(f"Parameter {i} Value")
    plt.ylabel("-2 * Log-Likelihood")
    plt.title(f"Profile Likelihood for Parameter {i}")
    plt.legend()
    plt.grid(True)
    plt.show()