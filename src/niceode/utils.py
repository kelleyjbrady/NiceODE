import mlflow.data.pandas_dataset
from scipy.integrate import solve_ivp
import numpy as np
import scipy.optimize
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
from typing import List, Dict
from joblib import Parallel, delayed
from numba import njit
from scipy.special import huber
import os
import warnings
import inspect
import ast
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.optimize import approx_fprime
from line_profiler import profile
from typing import Literal, Tuple
import scipy
from scipy.optimize._minimize import MINIMIZE_METHODS_NEW_CB
import joblib as jb
from typing import Self
from .diffeqs import PKBaseODE
import uuid
import mlflow
import multiprocessing
import jax.numpy as jnp

from mlflow.data.pandas_dataset import from_pandas
from .mlflow_utils import (get_class_source_without_docstrings,
                                  generate_class_contents_hash, 
                                  get_function_source_without_docstrings_or_comments, 
                                  MLflowCallback)
from typing import Type
from .pd_templates import InitValsPdCols
from warnings import warn
import diffrax
import jax
from io import BytesIO
from .model_assesment import construct_profile_ci
import re


def debug_print(print_obj, debug=False):
    if debug:
        if isinstance(print_obj, str):
            print(print_obj)



def plot_subject_levels(df: pd.DataFrame, x="TIME", y="DV", subject="SUBJID", ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    for c in df.columns:
        df.loc[df[c] == ".", c] = pd.NA
    df[y] = df[y].astype(pd.Float32Dtype())
    df[x] = df[x].astype(pd.Float32Dtype())
    sns.lineplot(data=df, x=x, y=y, hue=subject, ax=ax)


@dataclass
class ObjectiveFunctionColumn:
    coeff_name:str
    column_name: str
    log_name:str = None
    allometric_norm_value: float = None
    model_method: Literal['linear', 'allometric'] = "linear"
    optimization_init_val: np.float64 = 0.0
    optimization_history: list[np.float64] = field(default_factory=list)
    optimization_lower_bound: np.float64 = None
    optimization_upper_bound: np.float64 = None

    def __post_init__(self):
        log_name = self.coeff_name + '__' + self.column_name
        self.log_name = log_name if self.log_name is None else log_name
        # Define the allowed methods.
        allowed_methods = ["linear", "allometric"]

        if self.model_method not in allowed_methods:
            raise ValueError(
                f"""Method '{
                    self.model_method.__name__
                }' is not an allowed method. Allowed methods are: {
                    [method.__name__ for method in allowed_methods]
                }"""
            )
        if self.model_method == "allometric" and self.allometric_norm_value is None:
            raise ValueError(
                f"""Method '{
                    self.model_method.__name__
                }' is not allowed without providing the `allometric_norm_value`"""
            )
        self.allometric = (True
                        if self.model_method == 'allometric'
                        else False)
    
    def to_pandas(self) -> pd.DataFrame:
        cols = InitValsPdCols()
        row_data = {
                        cols.model_coeff: self.coeff_name,
                        cols.log_name: self.log_name,
                        cols.model_coeff_dep_var: self.column_name,
                        cols.population_coeff: False,
                        cols.model_error: False,
                        cols.init_val: self.optimization_init_val,
                        cols.model_coeff_lower_bound: self.optimization_lower_bound,
                        cols.model_coeff_upper_bound: self.optimization_upper_bound,
                        cols.allometric: self.allometric,
                        cols.allometric_norm_value: self.allometric_norm_value,
                        cols.subject_level_intercept: False,
                        cols.subject_level_intercept_name: None,
                        cols.subject_level_intercept_sd_init_val: False,
                        cols.subject_level_intercept_init_vals_column_name: None,
                        cols.subject_level_intercect_sd_lower_bound: None,
                        cols.subject_level_intercect_sd_upper_bound: None,
                    }
        row_data = cols.validate_df_row(row_data)
        pd_row = (pd.DataFrame([row_data], ))
        return pd_row.copy()


@dataclass
class PopulationCoeffcient:
    coeff_name: str
    log_name:str = None 
    optimization_init_val: np.float64 = None
    log_transform_init_val: bool = True
    optimization_history: list[np.float64] = field(default_factory=list)
    optimization_lower_bound: np.float64 = None
    optimization_upper_bound: np.float64 = None
    subject_level_intercept: bool = False
    subject_level_intercept_sd_name: str = None
    subject_level_intercept_sd_init_val: np.float64 = (
        None  # this is on the log scale, but the opt inti val is not, confusing
    )
    subject_level_intercept_sd_lower_bound: np.float64 = None
    subject_level_intercept_sd_upper_bound: np.float64 = None
    subject_level_intercept_opt_step_size: np.float64 = None
    subject_level_intercept_init_vals_column_name: str = None

    def __post_init__(self):
        self.log_name = (self.coeff_name + "_pop" 
                         if self.log_name is None 
                         else self.log_name)
        if self.optimization_init_val is None:
            self.optimization_init_val = np.random.rand() + 1e-6
        self.optimization_init_val = (
            np.log(self.optimization_init_val)
            if self.log_transform_init_val
            else self.optimization_init_val
        )

        c1 = self.subject_level_intercept_sd_init_val is None
        c2 = self.subject_level_intercept
        c3 = self.subject_level_intercept_sd_name is None
        if c2 and c3:
            self.subject_level_intercept_sd_name = f"omega2_{self.coeff_name}"
        if c1 and c2:
            self.subject_level_intercept_sd_init_val = np.random.rand() + 1e-6

    def to_pandas(self):
        cols = InitValsPdCols()
        row_data = {   
                #the name of the ODE parameter
                cols.model_coeff: self.coeff_name,
                #the name of the population component of the parameter's variance
                cols.log_name:self.log_name,
                cols.model_coeff_dep_var: None,
                cols.population_coeff: True,
                cols.model_error: False,
                cols.init_val: self.optimization_init_val,
                cols.model_coeff_lower_bound: self.optimization_lower_bound,
                cols.model_coeff_upper_bound: self.optimization_upper_bound,
                cols.allometric: None,
                cols.allometric_norm_value: None,
                cols.subject_level_intercept: self.subject_level_intercept,
                cols.subject_level_intercept_name: self.subject_level_intercept_sd_name,
                cols.subject_level_intercept_sd_init_val: self.subject_level_intercept_sd_init_val,
                cols.subject_level_intercept_init_vals_column_name: self.subject_level_intercept_init_vals_column_name,
                cols.subject_level_intercect_sd_lower_bound: self.subject_level_intercept_sd_lower_bound,
                cols.subject_level_intercect_sd_upper_bound: self.subject_level_intercept_sd_upper_bound,
                }
        row_data = cols.validate_df_row(row_data)
        pd_row = (pd.DataFrame([row_data], ))
        return pd_row.copy()


@dataclass
class ModelError:
    coeff_name: str
    model_method: Literal['constant', 'proportional', 'combined'] = 'constant'
    optimization_init_val: np.float64 = None
    log_transform_init_val: bool = True
    optimization_history: list[np.float64] = field(default_factory=list)
    optimization_lower_bound: np.float64 = None
    optimization_upper_bound: np.float64 = None
    subject_level_intercept: bool = False
    subject_level_intercept_sd_name: str = None
    subject_level_intercept_sd_init_val: np.float64 = (
        None  # this is on the log scale, but the opt inti val is not, confusing
    )
    subject_level_intercept_sd_lower_bound: np.float64 = None
    subject_level_intercept_sd_upper_bound: np.float64 = None
    subject_level_intercept_opt_step_size: np.float64 = None
    subject_level_intercept_init_vals_column_name: str = None

    def __post_init__(self):
        self.log_name = self.coeff_name + "_const"
        if self.model_method != 'constant':
            warn(f"Error model {self.model_method} is not currently implemented, defaulting to constant")
            self.model_method = 'constant'
        if self.optimization_init_val is None:
            self.optimization_init_val = np.random.rand() + 1e-6
        self.optimization_init_val = (
            np.log(self.optimization_init_val)
            if self.log_transform_init_val
            else self.optimization_init_val
        )

        c1 = self.subject_level_intercept_sd_init_val is None
        c2 = self.subject_level_intercept
        c3 = self.subject_level_intercept_sd_name is None
        if c2 and c3:
            self.subject_level_intercept_sd_name = f"omega2_{self.coeff_name}"
        if c1 and c2:
            self.subject_level_intercept_sd_init_val = np.random.rand() + 1e-6

    def to_pandas(self):
        cols = InitValsPdCols()
        row_data = {
                    cols.model_coeff: self.coeff_name,
                    cols.log_name:self.log_name ,
                    cols.model_coeff_dep_var: None,
                    cols.population_coeff: False,
                    cols.model_error: True,
                    cols.init_val: self.optimization_init_val,
                    cols.model_coeff_lower_bound: self.optimization_lower_bound,
                    cols.model_coeff_upper_bound: self.optimization_upper_bound,
                    cols.allometric: None,
                    cols.allometric_norm_value: None,
                    #Consider settings these to None as they do not currenly do anything
                        #That is, there is currently no subject level model error and it seems like
                        #it would be very difficult to estimate if it was included in the model as 
                        #such alongside the 
                    cols.subject_level_intercept: self.subject_level_intercept,
                    cols.subject_level_intercept_name: self.subject_level_intercept_sd_name,
                    cols.subject_level_intercept_sd_init_val: self.subject_level_intercept_sd_init_val,
                    cols.subject_level_intercept_init_vals_column_name: self.subject_level_intercept_init_vals_column_name,
                    cols.subject_level_intercect_sd_lower_bound: self.subject_level_intercept_sd_lower_bound,
                    cols.subject_level_intercect_sd_upper_bound: self.subject_level_intercept_sd_upper_bound,
                }
        row_data = cols.validate_df_row(row_data)
        pd_row = (pd.DataFrame([row_data], ))
        return pd_row.copy()

@dataclass
class ObjectiveFunctionBeta:
    column_name: str
    value: np.float64
    allometric_norm_value: float = None
    model_method: Literal["linear", "allometric"] = "linear"
    optimization_lower_bound: np.float64 = None
    optimization_upper_bound: np.float64 = None

    def __post_init__(self):
        # Define the allowed methods. Using a dictionary for better lookup performance
        # if the list becomes very large. You can also use a set.
        allowed_methods = ["linear", "allometric"]

        if self.model_method not in allowed_methods:
            raise ValueError(
                f"""Method '{
                    self.model_method.__name__
                }' is not an allowed method. Allowed methods are: {
                    [method.__name__ for method in allowed_methods]
                }"""
            )


@dataclass
class ODEInitVals:
    column_name: str


#    init_val_is_measured_t0_val: bool = True


def sum_of_squares_loss(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)


def mean_squared_error_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def huber_loss(y_true, y_pred, delta=1.0):
    resid = y_pred - y_true
    loss = huber(delta, resid)
    return np.mean(loss)

#consider changing this to allow sigma to be None, would be much more simple
def neg2_log_likelihood_loss(y_true, y_pred, sigma):
    sigma = sigma.to_numpy(dtype = np.float64)
    residuals = y_true - y_pred
    ss = np.sum(residuals**2)
    n = len(y_true)
    if len(sigma) == 0:
        sigma = np.array([np.sqrt(ss/n)])
    neg2_log_likelihood = n * np.log(2 * np.pi * sigma**2) + ss / sigma**2
    return neg2_log_likelihood[0]


def get_function_args(func):
    signature = inspect.signature(func)
    params = signature.parameters
    arg_names = [param.name.lower() for param in params.values()]
    return arg_names


def determine_ode_output_size(ode_func):
    """Reads the signature of a function to determine the length of the output. 
    This length is used in CompartmentalModel.__init__ to verify that the number of 
    columns in the data indicated to contain the per subject initial conditions (y0)
    matches the length of the defined ODE

    Args:
        ode_func (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        source = inspect.getsource(ode_func)  # ode_func is a class method
        source = source.lstrip()  # thus we need to strip out the indentation in the method to make it look like a normal function
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                if isinstance(node.value, ast.List):
                    return len(node.value.elts)  # Return the length of the tuple
                else:
                    return 0  # Not a tuple, implies a scalar (or single value)

        return None  # No return statement found

    except (TypeError, OSError):
        return None


def safe_signed_log(x):
    sign = np.sign(x)
    return sign * np.log1p(np.abs(x))


def estimate_fprime_jac(
    pop_coeffs, thetas, theta_data, apprx_fprime_j_cols_filt, y, model_obj
):
    def wrapped_solve_ivp(pop_coeffs_array):
        # pop_coeffs_array will have shape n_pop,
        # pop_coeffs_series = pd.Series(pop_coeffs_array, index = model_obj.pop_cols)
        pop_coeffs_array = (
            pop_coeffs_array.reshape(1, -1)
            if len(pop_coeffs_array.shape) == 1
            else pop_coeffs_array
        )
        pop_coeffs_inner = pd.DataFrame(
            pop_coeffs_array, dtype=pd.Float64Dtype(), columns=pop_coeffs.columns
        )
        model_coeffs_local = model_obj._generate_pk_model_coeff_vectorized(
            pop_coeffs_inner, thetas, theta_data
        )
        dep_var_preds, full_preds = model_obj._solve_ivp(model_coeffs_local, parallel=False)
        return dep_var_preds

    # the flatten is causing issues when pop_coeffs is not one row
    if pop_coeffs.shape[0] == 1:
        pop_coeffs_in = np.copy(pop_coeffs.values.flatten())
    else:
        pop_coeffs_in = np.copy(pop_coeffs.values)
    # need to call for each row in pop_coeffs_in when it is not one row long
    # this is way slower than the loop in `estimate_cdiff_jac`
    # Thus, this only works for the FO approximation, not FOCE
    J_afp = approx_fprime(pop_coeffs_in, wrapped_solve_ivp, epsilon=1e-6)
    # J_afp = J_afp.reshape(len(y), len(pop_coeffs.columns))
    # The reshape below will fail when there are less random effects than pop_coeffs
    # J_afp = J_afp.reshape(len(y), n_random_effects)
    J_afp = J_afp[:, apprx_fprime_j_cols_filt]
    return J_afp


def estimate_cdiff_jac(
    pop_coeffs, thetas, theta_data, n_random_effects, omegas_names, y, model_obj
):
    plus_pop_coeffs = pop_coeffs.copy()
    minus_pop_coeffs = pop_coeffs.copy()
    epsilon = 1e-6
    J_cd = np.zeros((len(y), n_random_effects), dtype=np.float64)
    for omega_idx, omega_i in enumerate(omegas_names):
        c = omega_i[0]
        plus_pop_coeffs[c] = plus_pop_coeffs[c] + epsilon
        plus_model_coeffs = model_obj._generate_pk_model_coeff_vectorized(
            plus_pop_coeffs, thetas, theta_data
        )
        plus_preds, full_plus_preds = model_obj._solve_ivp(
            plus_model_coeffs,
            parallel=False,
        )

        minus_pop_coeffs[c] = minus_pop_coeffs[c] - epsilon
        minus_model_coeffs = model_obj._generate_pk_model_coeff_vectorized(
            minus_pop_coeffs, thetas, theta_data
        )
        minus_preds, full_minus_preds = model_obj._solve_ivp(
            minus_model_coeffs,
            parallel=False,
        )

        J_cd[:, omega_idx] = (plus_preds - minus_preds) / (
            2 * epsilon
        )  # the central difference
    return J_cd

#@profile
def estimate_cdiff_jac_adaptive(
    pop_coeffs, thetas, theta_data, n_random_effects, omegas_names, y, model_obj
):
    """
    Estimates the Jacobian using central differences with an adaptive step size.

    The step size h for perturbing parameter x_j is calculated as:
    h_j = rel_step * |x_j| + abs_step
    where rel_step is sqrt(machine_epsilon) and abs_step is a small floor value.

    Args:
        pop_coeffs (dict): Dictionary of population coefficients.
        thetas (...): Thetas parameters.
        theta_data (...): Theta data.
        n_random_effects (int): Number of random effects (columns in Jacobian).
        omegas_names (list): List of tuples, where the first element [0] is the key
                             in pop_coeffs corresponding to the random effect.
        y (np.ndarray): Output vector from the model for the base coefficients.
                        Used here only to determine the size of the Jacobian.
        model_obj (object): An object with methods _generate_pk_model_coeff_vectorized
                            and _solve_ivp.

    Returns:
        np.ndarray: The estimated Jacobian matrix (len(y) x n_random_effects).
    """
    # Determine machine epsilon for the float type used in pop_coeffs
    # Use a default float64 if pop_coeffs is empty or types are mixed/non-float
    try:
        # Attempt to get dtype from a value, assuming dictionary values are numeric
        first_val = next(iter(pop_coeffs.values()))
        coeff_dtype = np.array(first_val).dtype
        if not np.issubdtype(coeff_dtype, np.floating):
            coeff_dtype = np.float64  # Default if not float
    except (StopIteration, TypeError):
        coeff_dtype = np.float64  # Default if empty or error

    machine_epsilon = np.finfo(coeff_dtype).eps
    rel_step = machine_epsilon ** (1 / 3)  # Relative step size factor
    # abs_step can be rel_step or a smaller fixed value like 1e-10 or machine_epsilon
    abs_step = rel_step  # Absolute step size floor (handles zero coeffs)
    # Or: abs_step = machine_epsilon

    J_cd = np.zeros((len(y), n_random_effects), dtype=coeff_dtype)  # Match dtype
    perturbed_pop_coeffs = pop_coeffs.copy()  # Work on a copy

    for omega_idx, omega_i in enumerate(omegas_names):
        c = omega_i[0]  # Key for the coefficient to perturb

        # Ensure the coefficient exists before perturbing
        if c not in pop_coeffs:
            # Option 1: Raise an error
            # raise KeyError(f"Coefficient key '{c}' from omegas_names not found in pop_coeffs")
            # Option 2: Skip this parameter (Jacobian column will be zero)
            print(f"Warning: Coefficient key '{c}' not found in pop_coeffs. Skipping.")
            continue  # Jacobian column remains zero
            # Option 3: Assume value is 0 (if appropriate for your model)
            # original_coeff_val = 0.0

        original_coeff_val = pop_coeffs[c]

        # --- Calculate adaptive step size h_j for this coefficient ---
        h_j = rel_step * np.abs(original_coeff_val) + abs_step
        h_j = h_j.values[0]
        # ---

        # --- Forward Perturbation ---
        perturbed_pop_coeffs[c] = original_coeff_val + h_j
        plus_model_coeffs = model_obj._generate_pk_model_coeff_vectorized(
            perturbed_pop_coeffs, thetas, theta_data
        )
        plus_preds, full_plus_preds = model_obj._solve_ivp(plus_model_coeffs, parallel=False)

        # --- Backward Perturbation ---
        perturbed_pop_coeffs[c] = original_coeff_val - h_j
        minus_model_coeffs = model_obj._generate_pk_model_coeff_vectorized(
            perturbed_pop_coeffs, thetas, theta_data
        )
        minus_preds, full_minus_preds = model_obj._solve_ivp(minus_model_coeffs, parallel=False)

        # --- Reset coefficient for the next iteration ---
        perturbed_pop_coeffs[c] = original_coeff_val

        # --- Central Difference Calculation ---
        # Ensure h_j is not zero if original_coeff_val was extremely small non-zero
        # and abs_step was also zero (unlikely with recommended abs_step).
        # Adding machine_epsilon avoids potential division by zero in extreme cases.
        denominator = 2.0 * h_j
        if denominator == 0:
            # Handle cases where h_j calculated to zero - shouldn't happen with abs_step > 0
            # If it does, the gradient is likely zero or numerically unstable anyway.
            # You might log a warning or set the column to zero explicitly.
            print(
                f"Warning: Calculated step size h_j is zero for coefficient {c}. Jacobian column set to zero."
            )
            J_cd[:, omega_idx] = 0.0
        else:
            J_cd[:, omega_idx] = (plus_preds - minus_preds) / denominator

    return J_cd


# This function needs to be updated to allow the dependent variable to be on the log scale


def estimate_jacobian(
    pop_coeffs: pd.DataFrame,
    thetas: pd.DataFrame,
    theta_data: List[pd.DataFrame],
    omega_names: List,
    y: np.array,
    model_obj,
    use_fprime: bool = True,
    use_cdiff: bool = True,
    use_adaptive=True,
    use_numdifftools=False,
):
    n_random_effects = len(omega_names)
    apprx_fprime_j_cols_filt = [
        pc in [i[0] for i in omega_names] for pc in pop_coeffs.columns
    ]
    J_afp = None
    J_cd = None
    J_cd_adap = None
    use_nonadaptive = False
    if use_fprime:  # this method currently only works when there is only one pop_coeffs
        J_afp = estimate_fprime_jac(
            pop_coeffs, thetas, theta_data, apprx_fprime_j_cols_filt, y, model_obj
        )
    if use_cdiff:
        if use_adaptive:
            J_cd_adap = estimate_cdiff_jac_adaptive(
                pop_coeffs, thetas, theta_data, n_random_effects, omega_names, y, model_obj
            )
        if use_nonadaptive: #for debugging
            J_cd = estimate_cdiff_jac(
                pop_coeffs, thetas, theta_data, n_random_effects, omega_names, y, model_obj
            )
        if use_adaptive and use_nonadaptive:
            if np.any(np.abs((J_cd_adap - J_cd)) > 1e-6):
                debug_this = True
                
    J = [i for i in [J_cd_adap,J_afp, J_cd] if i is not None][0]
    return J


def estimate_cov_chol(J, y_groups_idx, y, residuals, sigma2, omegas2, model_obj):
    # V_all = []
    log_det_V = 0
    L_all = []
    for sub in model_obj.unique_groups:
        filt = y_groups_idx == sub
        J_sub = J[filt]
        n_timepoints = len(J_sub)
        R_i = sigma2 * np.eye(n_timepoints)  # Constant error
        # Omega = np.diag(omegas**2) # Construct D matrix from omegas
        V_i = R_i + J_sub @ omegas2 @ J_sub.T

        L_i, lower = cho_factor(V_i)  # Cholesky of each V_i
        L_all.append(L_i)
        log_det_V += 2 * np.sum(np.log(np.diag(L_i)))  # log|V_i|

    L_block = block_diag(*L_all)  # key change from before
    V_inv_residuals = cho_solve((L_block, True), residuals)
    neg2_ll_chol = (
        log_det_V + residuals.T @ V_inv_residuals + len(y) * np.log(2 * np.pi)
    )

    return neg2_ll_chol


def estimate_cov_naive(
    J,
    y_groups_idx,
    y,
    residuals,
    sigma2,
    omega2,
    n_individuals,
    n_random_effects,
):
    J_vec = create_vectorizable_J(J, y_groups_idx, n_random_effects)
    Omega_expanded = np.kron(np.eye(n_individuals), omega2)
    covariance_matrix = (
        J_vec @ Omega_expanded @ J_vec.T
        + np.diag(np.full(len(y), sigma2))
        + 1e-6 * np.eye(len(y))
    )
    det_cov_matrix = np.linalg.det(covariance_matrix)
    if (
        (det_cov_matrix == 0)
        or (det_cov_matrix == np.inf)
        or (det_cov_matrix == -np.inf)
    ):
        need_debug = True
    inv_cov_matrix = np.linalg.inv(covariance_matrix)
    # this rarely is usuable due to inf or zero determinant when vectorized
    direct_neg2_ll = (
        len(y) * np.log(2 * np.pi)
        + np.log(det_cov_matrix)
        + residuals.T @ inv_cov_matrix @ residuals
    )
    return direct_neg2_ll


def estimate_cov_naive_subj(J, y_groups_idx, residuals, sigma2, omega2, model_obj):
    per_sub_direct_neg2_ll = 0.0
    debug_extreme_J = {}
    debug_near_zero_resid = {}
    debug_cov_diag_near_zero = {}
    cov_matrix_i = []
    for sub_idx, sub in enumerate(model_obj.unique_groups):
        filt = y_groups_idx == sub
        J_sub = J[filt]
        residuals_sub = residuals[filt]
        if np.any(np.abs(J_sub) < 1e-5) or np.any(np.abs(J_sub) > 1e5):
            debug_extreme_J[sub_idx] = J_sub
        n_timepoints = len(J_sub)
        covariance_matrix_sub = (
            J_sub @ omega2 @ J_sub.T
            + np.diag(np.full(n_timepoints, sigma2))
            + 1e-6 * np.eye(n_timepoints)
        )
        cov_matrix_i.append(covariance_matrix_sub)
        if np.any(np.abs(np.diag(covariance_matrix_sub)) < 1e-5):
            debug_cov_diag_near_zero[sub_idx] = np.diag(covariance_matrix_sub)

        if np.all(np.abs(residuals_sub) < 1e-5) or np.all(np.abs(residuals_sub) > 1e5):
            debug_near_zero_resid[sub_idx] = residuals_sub
        det_cov_matrix_i = np.linalg.det(covariance_matrix_sub)
        inv_cov_matrix_i = np.linalg.inv(covariance_matrix_sub)
        neg2_log_likelihood_i = (
            n_timepoints * np.log(2 * np.pi)
            + np.log(det_cov_matrix_i)
            + residuals_sub.T @ inv_cov_matrix_i @ residuals_sub
        )
        per_sub_direct_neg2_ll = per_sub_direct_neg2_ll + neg2_log_likelihood_i
    return per_sub_direct_neg2_ll


#@profile
def estimate_neg_log_likelihood(
    J,
    y_groups_idx,
    y,
    residuals,
    sigma2,
    omegas2,
    n_individuals,
    n_random_effects,
    model_obj,
    cholsky_cov=True,
    naive_cov_vec=False,
    naive_cov_subj=False,
):
    cholsky_neg_ll = None
    naive_vec_neg_ll = None
    naive_subj_neg_ll = None
    if cholsky_cov:
        cholsky_neg_ll = estimate_cov_chol(
            J, y_groups_idx, y, residuals, sigma2, omegas2, model_obj
        )
    if naive_cov_vec:
        naive_vec_neg_ll = estimate_cov_naive(
            J,
            y_groups_idx,
            y,
            residuals,
            sigma2,
            omegas2,
            n_individuals,
            n_random_effects,
        )
    if naive_cov_subj:
        naive_subj_neg_ll = estimate_cov_naive_subj(
            J, y_groups_idx, residuals, sigma2, omegas2, model_obj
        )
    neg_ll = [
        i
        for i in [cholsky_neg_ll, naive_vec_neg_ll, naive_subj_neg_ll]
        if i is not None
    ][0]

    return neg_ll


#@profile
def _estimate_b_i(
    model_obj,
    pop_coeffs,
    thetas,
    beta_data,
    sigma2,
    Omega2,
    omega_names,
    b_i_init,
    ode_t0_val,
    time_mask_i,
    y_i,
    sub,
    debug_print=debug_print,
):
    """Estimates b_i for a *single* individual using Newton-Raphson (or similar)."""

    def conditional_log_likelihood(
        b_i,
        y_i,
        pop_coeffs,
        thetas,
        beta_data,
        sigma2,
        Omega2,
        model_obj,
        debug_print=debug_print,
    ):
        # Combine the population coefficients and b_i for this individual
        debug_print("INNER OPTIMIZATION START ===================")
        # debug_print(f"inner_opt_calls: {inner_opt_calls}\n")
        debug_print(f"sub: {sub}\n")
        debug_print(f"b_i: {b_i}\n")
        debug_print(f"pop_coeffs: {pop_coeffs}\n")
        debug_print(f"thetas: {thetas}\n")
        debug_print(f"beta_data: {beta_data}\n")
        debug_print(f"sigma2: {sigma2}\n")
        debug_print(f"Omega: {Omega2}\n")

        check_pos_inf = np.any(np.isinf(b_i))
        check_neg_inf = np.any(np.isneginf(b_i))
        if check_pos_inf or check_neg_inf:
            warnings.warn(
                f"Estimated b_i for subject {sub} was infinite, setting b_i to zero."
            )
            b_i[check_pos_inf] = 0
            b_i[check_neg_inf] = 0
        combined_coeffs = pop_coeffs.copy()
        # Ensure b_i is a Series with correct index for merging

        b_i = (
            b_i.reshape(1, -1) if len(b_i) == 1 else b_i
        )  # not sure if this is necessary
        b_i_df = pd.DataFrame(
            b_i.reshape(1, -1), dtype=pd.Float64Dtype(), columns=omega_names
        )
        b_i_df.columns = pd.MultiIndex.from_tuples(b_i_df.columns)

        for col in combined_coeffs.columns:
            if col in b_i_df.columns:
                # not sure why these need to be flattened now, but that seems like the case.
                # these are flattened elsewhere, orginally didn't need to be b/v b_i_df was b_i_series
                # and this is the only place where pd.Series is being used, although it is probably
                # better to use pd.Series
                combined_coeffs[col] = (
                    pop_coeffs[col].values + b_i_df[col].values.flatten()
                )
        # Generate the model coefficients for this individual
        model_coeffs_i = model_obj._generate_pk_model_coeff_vectorized(
            combined_coeffs, thetas, beta_data, expected_len_out=1
        )
        # Solve the ODEs for this individual
        if np.any(np.isinf(model_coeffs_i.to_numpy().flatten())):
            log_likelihood_data = 1e12
        else:
            preds_i, full_preds_i = model_obj._solve_ivp(
                model_coeffs_i, ode_t0_val, time_mask_i, parallel=False
            )
            # compute residuals
            residuals_i = y_i - preds_i
            # Calculate the negative conditional log-likelihood
            error_model = 'additive'
            if error_model == 'additive':
                n_t = len(y_i)
                sum_sq_residuals = np.sum(residuals_i**2)
                log_likelihood_data = -0.5 * (n_t * np.log(2 * np.pi) 
                                    + n_t * np.log(sigma2) 
                                    + sum_sq_residuals / sigma2)
                assert len(log_likelihood_data) == 1
                log_likelihood_data = log_likelihood_data[0]
            
        diag_omega2 = np.diag(Omega2)
        # This check is critical for the shrinkage problem!
        if np.any(diag_omega2 <= 0):
            return np.inf # Penalize non-positive definite Omega matrices

        sum_log_diag_omega2 = np.sum(np.log(diag_omega2))
        b_i_flat = b_i.flatten() # Ensure b_i is 1D
        prior_penalty = np.sum(b_i_flat**2 / diag_omega2)

        log_likelihood_prior = -0.5 * (len(b_i_flat) * np.log(2 * np.pi) 
                            + sum_log_diag_omega2 
                            + prior_penalty)
        
        
        debug_print(f"log_likelihood_data: {-log_likelihood_data}\n")
        debug_print(f"log_likelihood_prior: {-log_likelihood_prior}\n")
        debug_print(
            f"total log_likelihood: {-log_likelihood_data - log_likelihood_prior}\n"
        )
        debug_print("INNER OPTIMIZATION END ===================")
        loss_out = -(log_likelihood_data + log_likelihood_prior)
        return loss_out

    # y_i = model_obj.y[model_obj.y_groups == b_i_init.name] #b_i_init.name will contain subject id
    bound_bi = False
    if bound_bi:
        b_i_bounds = []
        if np.any(b_i_init == 0):
            for omega in np.diag(Omega2) ** 0.5:  # Iterate through standard deviations
                lower_bound = -6 * omega
                upper_bound = 6 * omega
                b_i_bounds.append((lower_bound, upper_bound))
        else:
            for b_i in b_i_init.to_numpy().flatten():
                lower_bound = b_i - np.abs(
                    2 * b_i
                )  # 3 is a hyperparameter that should be visible to the user.
                upper_bound = b_i + np.abs(2 * b_i)
                b_i_bounds.append((lower_bound, upper_bound))
        debug_print(f"INNER b_i_bounds: {b_i_bounds}")
    else:
        b_i_bounds = None
    # Use scipy.optimize.minimize for the inner optimization
    result_b_i = minimize(
        conditional_log_likelihood,
        b_i_init.values.flatten(),  # Initial guess for b_i (flattened)
        args=(y_i, pop_coeffs, thetas, beta_data, sigma2, Omega2, model_obj),
        method="L-BFGS-B",
        bounds=b_i_bounds,
        # tol = 1e-6
    )

    b_i_estimated = pd.DataFrame(
        result_b_i.x.reshape(1, -1),
        dtype=pd.Float64Dtype(),
        columns=b_i_init.columns,
    )
    
    hess_inv = result_b_i.hess_inv

    return b_i_estimated, hess_inv


#@profile
def FOCE_approx_ll_loss(
    pop_coeffs,
    sigma,
    omegas,
    thetas,
    theta_data,
    model_obj,
    FO_b_i_apprx=None,
    tqdm_bi=False,
    debug=None,
    debug_print=debug_print,
    focei = False,
    **kwargs,
):
    debug_print("Objective Call Start")
    y = np.copy(model_obj.y)
    y_groups_idx = np.copy(model_obj.y_groups)
    omegas_names = list(omegas.columns)
    omegas = omegas.to_numpy(
        dtype=np.float64
    ).flatten()  # omegas as SD, we want Variance, thus **2 below
    omegas2 = np.diag(
        omegas**2
    )  # FO assumes that there is no cov btwn the random effects, thus off diags are zero
    sigma = sigma.to_numpy(dtype=np.float64)[0]
    sigma2 = sigma**2
    n_individuals = len(model_obj.unique_groups)
    n_random_effects = len(omegas_names)
    debug_print(
        "Inner Loop Optimizing b_i ================= INNER LOOP =================="
    )
    b_i_estimates = []
    b_i_hess_invs = []
    if tqdm_bi:
        iter_obj = tqdm(model_obj.unique_groups)
    else:
        iter_obj = model_obj.unique_groups
    for sub_idx, sub in enumerate(iter_obj):
        FO_b_i_approx_exists = False
        if FO_b_i_apprx is not None:
            if FO_b_i_apprx.size > 0:
                FO_b_i_approx_exists = True
        b_i_init = (
            FO_b_i_apprx.iloc[sub_idx, :].to_numpy().flatten()
            if FO_b_i_approx_exists
            else np.zeros_like(omegas)
        )
        b_i_init_s = pd.Series(b_i_init, index=omegas_names, name=sub)
        b_i_init = pd.DataFrame(
            b_i_init.reshape(1, -1), columns=omegas_names, dtype=pd.Float64Dtype()
        )
        b_i_init.columns = pd.MultiIndex.from_tuples(b_i_init.columns)
        thetas_i = thetas.iloc[sub_idx, :] if len(thetas) > 1 else thetas
        theta_data_i = (
            theta_data.iloc[sub_idx, :] if len(theta_data) > 1 else theta_data
        )
        ode_t0_i = model_obj.ode_t0_vals.iloc[[sub_idx], :]
        time_mask_i = model_obj.time_mask[
            sub_idx, :
        ]  # very confusing to have these sometimes be arrays
        y_i = model_obj.y[model_obj.y_groups == sub]
        debug_print("INNER LOOP VALUES IN START =================")
        debug_print(f"sub_idx: {sub_idx}\n")
        debug_print(f"sub: {sub}\n")
        debug_print(f"b_i: {b_i_init}\n")
        debug_print(f"pop_coeffs: {pop_coeffs}\n")
        debug_print(f"thetas_i: {thetas_i}\n")
        debug_print(f"theta_data_i: {theta_data_i}\n")
        debug_print(f"ode_t0_i: {ode_t0_i}\n")
        debug_print(f"time_mask_i: {time_mask_i}\n")
        debug_print(f"y_i: {y_i}\n")
        debug_print(f"omegas2: {omegas2}")
        debug_print("INNER LOOP VALUES IN END ===================")

        b_i_est, b_i_hess_inv = _estimate_b_i(
            model_obj,
            pop_coeffs,
            thetas_i,
            theta_data_i,
            sigma2,
            omegas2,
            omegas_names,
            b_i_init,
            ode_t0_i,
            time_mask_i,
            y_i,
            sub,
        )
        b_i_estimates.append(b_i_est)
        b_i_hess_invs.append(b_i_hess_inv)
    b_i_estimates = pd.concat(b_i_estimates)
    b_i_estimates.columns = pd.MultiIndex.from_tuples(b_i_estimates.columns)

    combined_coeffs = pd.DataFrame(
        np.tile(pop_coeffs.values, (b_i_estimates.shape[0], 1)),
        dtype=np.float64,
        columns=pop_coeffs.columns,
    )
    for col in combined_coeffs.columns:
        if col in b_i_estimates.columns:
            # so much fuckery with these df . . .
            combined_coeffs[col] = (
                combined_coeffs[col].values + b_i_estimates[col].values.flatten()
            )

    model_coeffs = model_obj._generate_pk_model_coeff_vectorized(
        combined_coeffs, thetas, theta_data
    )
    # would be good to create wrapper methods inside of the model obj with these args (eg. `parallel = model_obj.parallel`)
    # prepopulated with partial
    preds, full_preds = model_obj._solve_ivp(
        model_coeffs,
        parallel=False,
    )
    residuals = y - preds

    apprx_fprime_jac = False
    central_diff_jac = True
    J = estimate_jacobian(
        combined_coeffs,
        thetas,
        theta_data,
        omegas_names,
        y,
        model_obj,
        use_fprime=apprx_fprime_jac,
        use_cdiff=central_diff_jac,
    )

    if model_obj.ode_t0_vals_are_subject_y0:
        drop_idx = model_obj.subject_y0_idx
        J = np.delete(J, drop_idx, axis=0)
        residuals = np.delete(residuals, drop_idx)
        y = np.delete(y, drop_idx)
        y_groups_idx = np.delete(y_groups_idx, drop_idx)

    direct_det_cov = False
    per_sub_direct_neg_ll = False
    cholsky_cov = True
    neg2_ll = estimate_neg_log_likelihood(
        J,
        y_groups_idx,
        y,
        residuals,
        sigma2,
        omegas2,
        n_individuals,
        n_random_effects,
        model_obj,
        cholsky_cov=cholsky_cov,
        naive_cov_vec=direct_det_cov,
        naive_cov_subj=per_sub_direct_neg_ll,
    )
    
    if focei:
        interaction_term = 0
        calculation_successful = True 

        # Loop through the inverse Hessians from your optimizer results
        for hess_inv_i in b_i_hess_invs:
            try:
                # 1. Apply Cholesky directly to the INVERSE Hessian.
                # This serves as our stability and convergence check.
                L_inv_i = np.linalg.cholesky(hess_inv_i.todense())
                
                # 2. Calculate the log-determinant of the INVERSE Hessian.
                logdet_H_inv_i = 2 * np.sum(np.log(np.diag(L_inv_i)))
                
                # 3. The log-determinant of H is the NEGATIVE of the above.
                logdet_H_i = -logdet_H_inv_i
                
                interaction_term += logdet_H_i

            except np.linalg.LinAlgError:
                # This block executes if hess_inv_i is not positive-definite,
                # indicating a failure in the inner optimization step.
                calculation_successful = False
                break
        
    
        neg2_ll = neg2_ll + interaction_term
    debug_print(
        "Objective Call Complete ============= OBEJECTIVE CALL COMPLETE ======================="
    )
    debug_print(f"Loss: {neg2_ll}\n")
    debug_print(f"b_i_estimates: {b_i_estimates}\n")
    return neg2_ll, b_i_estimates, (preds, full_preds)

def FOCEi_approx_ll_loss(
    pop_coeffs,
    sigma,
    omegas,
    thetas,
    theta_data,
    model_obj,
    FO_b_i_apprx=None,
    tqdm_bi=False,
    debug=None,
    debug_print=debug_print,
    **kwargs
):

    res_collection = FOCE_approx_ll_loss(
        pop_coeffs,
        sigma,
        omegas,
        thetas,
        theta_data,
        model_obj,
        FO_b_i_apprx=FO_b_i_apprx,
        tqdm_bi=tqdm_bi,
        debug=debug,
        debug_print=debug_print,
        focei = True,
        **kwargs,)
    
    return res_collection

#@profile
def FO_approx_ll_loss(
    pop_coeffs,
    sigma,
    omegas,
    thetas,
    theta_data,
    model_obj,
    solve_for_omegas=False,
    
    **kwargs,
):
    # unpack some variables locally for clarity
    y = np.copy(model_obj.y)
    y_groups_idx = np.copy(model_obj.y_groups)
    omegas_names = list(omegas.columns)
    omegas = omegas.to_numpy(
        dtype=np.float64
    ).flatten()  # omegas as SD, we want Variance, thus **2 below
    omegas2 = np.diag(
        omegas**2
    )  # FO assumes that there is no cov btwn the random effects, thus off diags are zero
    #this is not actually FO's assumption, but a simplification, 
    sigma = sigma.to_numpy(dtype=np.float64)[0]
    sigma2 = sigma**2
    n_individuals = len(model_obj.unique_groups)
    n_random_effects = len(omegas_names)

    # estimate model coeffs when the omegas are zero -- the first term of the taylor exapansion apprx
    model_coeffs = model_obj._generate_pk_model_coeff_vectorized(
        pop_coeffs, thetas, theta_data
    )
    # would be good to create wrapper methods inside of the model obj with these args (eg. `parallel = model_obj.parallel`)
    # prepopulated with partial
    preds, full_preds = model_obj._solve_ivp(
        model_coeffs,
        parallel=False,
    )
    residuals = y - preds

    # estimate jacobian
    # there is something going on with the way this is filtered to use fprime and
    # when there are multiple omegas
    apprx_fprime_jac = False
    central_diff_jac = True
    use_adaptive = True
    J = estimate_jacobian(
        pop_coeffs,
        thetas,
        theta_data,
        omegas_names,
        y,
        model_obj,
        use_fprime=apprx_fprime_jac,
        use_cdiff=central_diff_jac,
        use_adaptive=use_adaptive,
    )

    if np.all(J == 0):
        raise ValueError("All elements of the Jacobian are zero")
    # drop initial values from relevant arrays if `model_obj.ode_t0_vals_are_subject_y0`
    # perfect predictions can cause issues during optimization and also add no information to the loss
    # If there are any subjects with only one data point this will fail by dropping the entire subject
    if model_obj.ode_t0_vals_are_subject_y0:
        drop_idx = model_obj.subject_y0_idx
        J = np.delete(J, drop_idx, axis=0)
        residuals = np.delete(residuals, drop_idx)
        y = np.delete(y, drop_idx)
        y_groups_idx = np.delete(y_groups_idx, drop_idx)

    # Estimate the covariance matrix, then estimate neg log likelihood
    direct_det_cov = False
    per_sub_direct_neg_ll = False
    cholsky_cov = True
    neg2_ll = estimate_neg_log_likelihood(
        J,
        y_groups_idx,
        y,
        residuals,
        sigma2,
        omegas2,
        n_individuals,
        n_random_effects,
        model_obj,
        cholsky_cov=cholsky_cov,
        naive_cov_vec=direct_det_cov,
        naive_cov_subj=per_sub_direct_neg_ll,
    )

    # if predicting or debugging, solve for the optimal b_i given the first order apprx
    # perhaps b_i approx is off in the 2cmpt case bc the model was learned on t1+ w/
    # t0 as the intial condition but now t0 is in the data w/out a conc in the DV
    # col, this should make the resdiduals very wrong
    b_i_approx = np.zeros((n_individuals, n_random_effects))
    if solve_for_omegas:
        for sub_idx, sub in enumerate(model_obj.unique_groups):
            filt = y_groups_idx == sub
            J_sub = J[filt]
            residuals_sub = residuals[filt]
            try:
                # Ensure omegas2 is invertible (handle near-zero omegas)
                # Add a small value to diagonal for stability if needed, or use pinv
                min_omega_var = 1e-9  # Example threshold
                stable_omegas2 = np.diag(np.maximum(np.diag(omegas2), min_omega_var))
                omega_inv = np.linalg.inv(stable_omegas2)

                # Corrected matrix A
                A = J_sub.T @ J_sub + sigma2 * omega_inv
                # Right-hand side
                rhs = J_sub.T @ residuals_sub
                # Solve
                b_i_approx[sub_idx, :] = np.linalg.solve(A, rhs)

            except np.linalg.LinAlgError:
                print(
                    f"Warning: Linear algebra error (likely singular matrix) for subject {sub}. Setting b_i to zero."
                )

            except Exception as e:
                print(f"Error calculating b_i for subject {sub}: {e}")

    return neg2_ll, b_i_approx, (preds, full_preds)


# @njit
def create_vectorizable_J(J, groups_idx, n_random_effects):
    unique_groups = np.unique(groups_idx)
    J_reshaped = np.zeros((len(J), len(unique_groups) * n_random_effects))
    start_row = 0
    for g_idx, group in enumerate(unique_groups):
        n_timepoints = np.sum(groups_idx == group)
        end_row = start_row + n_timepoints
        for j in range(n_random_effects):
            # start_row = g_idx * n_timepoints
            # end_row = (g_idx + 1) * n_timepoints
            start_col_original = j  # Use j directly
            start_col_reshaped = g_idx * n_random_effects + j
            end_col_reshaped = start_col_reshaped + 1
            J_tmp = J[start_row:end_row, start_col_original : start_col_original + 1]
            J_reshaped[start_row:end_row, start_col_reshaped:end_col_reshaped] = J_tmp
        start_row = end_row
    return J_reshaped


ALLOWED_MINIMIZE_METHODS: Tuple[str, ...] = tuple(MINIMIZE_METHODS_NEW_CB)


def optimize_with_checkpoint_joblib(
    func,
    x0,
    checkpoint_filename,
    *args,
    verbose_callback=False,
    n_checkpoint: int = 0,
    warm_start=False,
    tol=None,
    minimize_method: ALLOWED_MINIMIZE_METHODS = "l-bfgs-b",
    **kwargs,
):
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
    check_name = checkpoint_filename.replace(".jb", "")
    if not os.path.exists("logs"):
        os.mkdir("logs")
    checkpoints = [i.replace(".jb", "") for i in os.listdir("logs") if check_name in i]
    max_check_idx = [i.split("__")[-1] for i in checkpoints if len(i.split("__")) > 1]
    try:
        max_check_idx = [int(i) for i in max_check_idx]
    except ValueError:
        max_check_idx = []
    max_check_idx = f"__{np.max(max_check_idx)}" if len(max_check_idx) > 1 else ""
    checkpoint_filename = os.path.join("logs", check_name + f"{max_check_idx}.jb")
    if warm_start:
        try:
            checkpoint = load(checkpoint_filename)
            x0 = checkpoint["x"]
            iteration = checkpoint["iteration"]
            print(f"Resuming optimization from iteration {iteration}")
        except FileNotFoundError:
            print("No checkpoint found, starting from initial guess.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}, starting from initial guess.")

    def callback_with_checkpoint(xk, checkpoint_filename):
        nonlocal iteration
        iteration += 1
        # print(iteration)
        if iteration % n_checkpoint == 0:
            checkpoint = {"x": xk, "iteration": iteration}
            checkpoint_filename = checkpoint_filename.replace(
                ".jb", f"__{iteration}.jb"
            )
            dump(checkpoint, checkpoint_filename)
            print(f"Iteration {iteration}: Checkpoint saved to {checkpoint_filename}")
        # print('no log')

    # Ensure callback is not already in kwargs
    if n_checkpoint > 0:
        if "callback" not in kwargs:
            kwargs["callback"] = partial(
                callback_with_checkpoint, checkpoint_filename=checkpoint_filename
            )
        else:
            # If callback exists, combine it with the existing one
            user_callback = kwargs["callback"]

            def combined_callback(xk):
                callback_with_checkpoint(xk, checkpoint_filename)
                user_callback(xk)

            kwargs["callback"] = combined_callback

    result = minimize(
        func,
        x0,
        *args,
        method=minimize_method,
        tol=tol,
        options={"disp": False, 
                 },
        **kwargs,
    )

    # Remove checkpoint file at end.
    # try:
    #    os.remove(checkpoint_filename)
    # except:
    #    pass

    return result

class CompartmentalModel(RegressorMixin, BaseEstimator):
    #@profile
    def __init__(
        self,
        model_name: str = None,
        subject_id_col: str = "SUBJID",
        conc_at_time_col: str = "DV",
        time_col="TIME",
        solve_ode_at_time_col: str = None,
        pk_model_class: Type[PKBaseODE] = None,
        ode_t0_cols: List[ODEInitVals] = None,
        ode_t0_time_val: int | float = 0,
        population_coeff: List[PopulationCoeffcient] = [
            PopulationCoeffcient("k", 0.6),
            PopulationCoeffcient("vd", 2.0),
        ],
        dep_vars: Dict[str, List[ObjectiveFunctionColumn]] = None,
        dep_vars2: List[ObjectiveFunctionColumn] = None,
        no_me_loss_function=neg2_log_likelihood_loss,
        no_me_loss_params={},
        no_me_loss_needs_sigma=True,
        me_loss_function=FO_approx_ll_loss,
        loss_summary_name = "neg2_loglikelihood",
        model_error_sigma: PopulationCoeffcient = PopulationCoeffcient(
            "sigma",
            optimization_init_val=0.2,
            optimization_lower_bound=1e-6,
            optimization_upper_bound=20,
        ),
        model_error2:ModelError = None,
        optimizer_tol=1e-4,
        verbose=False,
        minimize_method: Literal[*MINIMIZE_METHODS_NEW_CB] = "l-bfgs-b",
        batch_id: uuid.UUID = None,
        model_id: uuid.UUID = None,
    ):
        #defaults related to ODE solving
        self.stiff_ode = None
        self.jax_ivp_nonstiff_solver_is_compiled = False 
        self.jax_ivp_nonstiff_compiled_solver_ = None
        self.jax_ivp_stiff_solver_is_compiled = False 
        self.jax_ivp_stiff_compiled_solver_ = None
        self.jax_ivp_pymcstiff_solver_is_compiled = False
        self.jax_ivp_pymcstiff_compiled_solver_ = None
        self.loss_summary_name = loss_summary_name
        self.init_vals_pd_cols = InitValsPdCols()
        self.b_i_approx = None
        self.batch_id = uuid.uuid4() if batch_id is None else batch_id
        self.model_id = uuid.uuid4() if model_id is None else model_id
        self.model_name = model_name
        #uninitialized pk_model_class to use for signature inspection
        self._pk_model_class = pk_model_class
        #initialized pk_model_class to use with an ivp solver
        self.pk_model_class = pk_model_class()
        #the ode within the pk_model_class
        self.pk_model_function = self.pk_model_class.ode
        
        dep_vars = {} if dep_vars is None else dep_vars
        population_coeff = [] if population_coeff is None else population_coeff
        self.model_error_sigma = model_error_sigma  # perhaps update this to do a check and set the attr to None if this param is not needed
        self.ode_output_size = determine_ode_output_size(self.pk_model_function)
        self.ode_t0_cols = self._validate_ode_t0_vals_size(ode_t0_cols)
        self.groupby_col = subject_id_col
        self.conc_at_time_col = conc_at_time_col
        self.time_col = time_col
        self.solve_ode_at_time_col = solve_ode_at_time_col
        self.ode_t0_time_val = ode_t0_time_val
        self.verbose = verbose
        self.minimize_method = minimize_method
        self.pk_args_diffeq = get_function_args(self.pk_model_function)[
            2:
        ]  # this relys on defining the dif eqs as I have done
        for arg_name in self.pk_args_diffeq:
            if len(population_coeff) == 0:
                population_coeff.append(PopulationCoeffcient(arg_name))
            else:
                assigned_args = [obj.coeff_name for obj in population_coeff]
                if arg_name not in assigned_args:
                    population_coeff.append(PopulationCoeffcient(arg_name))
        self.no_me_loss_function = no_me_loss_function
        self.no_me_loss_needs_sigma = no_me_loss_needs_sigma
        self.me_loss_function = me_loss_function
        # not sure if the section below needs to be in two places
        self.model_error2 = model_error2
        self.dep_vars2 = dep_vars2
        for coef_obj in population_coeff:
            c = coef_obj.coeff_name
            if c not in (i for i in dep_vars):
                dep_vars[c] = []
        assert sorted([i.coeff_name for i in population_coeff]) == sorted([i for i in dep_vars])
        # END section
        self.population_coeff = population_coeff
        self.dep_vars = dep_vars
        self.init_vals = self._unpack_init_vals()
        #self.init_vals_pd_alt = self._unpack_init_vals2()
        #self.result_template = self._initialize_result_template(self.init_vals_pd_alt)
        
        self.bounds = self._unpack_upper_lower_bounds(self.model_error_sigma)
        self.no_me_loss_params = no_me_loss_params
        self.optimizer_tol = optimizer_tol

        # helper attributes possibly defined later
        self.fit_result_ = None

    def _initialize_result_template(self, init_vals_pd):
        #pop_coeffs
        cols = self.init_vals_pd_cols
        pop_coeff_rows = init_vals_pd.loc[init_vals_pd[cols.population_coeff]]
        result_rows = []
        for idx, row in pop_coeff_rows.iterrows():
            result_rows.append(
                {
                "model_coeff": row[cols.model_coeff],
                "log_name":row[cols.log_name],
                "population_coeff": True,
                "model_error": False,
                "subject_level_intercept": False,
                "coeff_dep_var": False,
                "model_coeff_dep_var": None,
                "subject_level_intercept_name": None,
                "init_val":row[cols.init_val],
                "lower_bound":row[cols.model_coeff_lower_bound],
                "upper_bound":row[cols.model_coeff_upper_bound]
                }
            )
            
        c1 = self.no_me_loss_needs_sigma
        c2 = self._subject_intercept_detected
        model_error_rows = init_vals_pd.loc[init_vals_pd[cols.model_error]]
        if c1 or c2 or len(model_error_rows) > 0:
            for idx, row in model_error_rows.iterrows():
                result_rows.append(
                    {
                    "model_coeff": row[cols.model_coeff],
                    "log_name":row[cols.log_name],
                    "population_coeff": False,
                    "model_error": True,
                    "subject_level_intercept": False,
                    "coeff_dep_var": False,
                    "model_coeff_dep_var": None,
                    "subject_level_intercept_name": None,
                    "init_val":row[cols.init_val],
                    "lower_bound":row[cols.model_coeff_lower_bound],
                    "upper_bound":row[cols.model_coeff_upper_bound]
                    }
                )
        
        omega_rows = init_vals_pd.loc[init_vals_pd[cols.subject_level_intercept]]
        for idx, row in omega_rows.iterrows():
            result_rows.append(
                {
                        "model_coeff": row[cols.model_coeff],
                        "log_name": row[cols.subject_level_intercept_name],
                        "population_coeff": False,
                        "model_error": False,
                        "subject_level_intercept": True,
                        "coeff_dep_var": False,
                        "model_coeff_dep_var": None,
                        "subject_level_intercept_name": row[cols.subject_level_intercept_name],
                        "init_val":row[cols.subject_level_intercept_sd_init_val],
                        "lower_bound":row[cols.subject_level_intercect_sd_lower_bound],
                        "upper_bound":row[cols.subject_level_intercect_sd_upper_bound]
                    }
                
            )
        theta_rows = (init_vals_pd.loc[
            init_vals_pd[cols.model_coeff_dep_var].isnull() == False
        ])
        for idx, row in theta_rows.iterrows():
            result_rows.append(
                 {
                        "model_coeff": row[cols.model_coeff],
                        "log_name": row[cols.log_name],
                        "population_coeff": False,
                        "model_error": False,
                        "subject_level_intercept": False,
                        "coeff_dep_var": True,
                        "model_coeff_dep_var": row[cols.model_coeff_dep_var],
                        "subject_level_intercept_name": None,
                        "init_val":row[cols.init_val], 
                        "lower_bound":row[cols.model_coeff_lower_bound],
                        "upper_bound":row[cols.model_coeff_upper_bound]
                        
                    }
            )
        return pd.DataFrame(result_rows)
        
             

    def _initalize_mlflow_experiment(self, ):
        #this seems very brittle . . .
        #this func works, but is generally not correct
        try:
            experiment_id = mlflow.create_experiment(self.batch_id)
        except mlflow.exceptions.MlflowException as e:
            if "RESOURCE_ALREADY_EXISTS" in str(e):
                experiment_id = mlflow.get_experiment_by_name(self.batch_id).experiment_id
        else:
            #raises each time a new experiment is created
            raise
        mlflow.set_experiment(experiment_name=self.batch_id)
        self.mlf_experiment_id = experiment_id
    
    def _validate_ode_t0_vals_size(self, ode_t0_vals) -> List[ODEInitVals]:
        """Validates that the list of `ODEInitVals` stored in `self.ode_t0_vals`
        has the same length as the output of the ODE function to be integrated.

        Args:
            ode_t0_vals (list): A list of `ODEInitVals`

        Raises:
            ValueError: Indicates the size of the provided y0 does not match the shape of the ODE output. 

        Returns:
            list: The validated input `ode_t0_vals` list. 
        """
        size_tmp = len(ode_t0_vals)
        if size_tmp != self.ode_output_size:
            raise ValueError(
                f"n={size_tmp} ODE t0 args were provided (`ode_t0_vals`) but the provided ODE (`pk_model_function`) requires n = {self.ode_output_size}"
            )
        else:
            return ode_t0_vals

    def _unpack_init_vals2(self, ):
        init_vals_pd = []
        init_vals_pop = []
        init_vals_sigma = []
        init_vals_theta = []
        cols = self.init_vals_pd_cols
        subject_intercept_detected = False
        #pop_coeffs
        for pop_coeff in self.population_coeff:
            if pop_coeff.subject_level_intercept:
                subject_intercept_detected = True
            init_vals_pop.append(pop_coeff.to_pandas())
        #then sigma
        if subject_intercept_detected or self.no_me_loss_needs_sigma:
            init_vals_sigma.append(self.model_error2.to_pandas())
        #then thetas
        for theta in self.dep_vars2:
            init_vals_theta.append(theta.to_pandas())
        init_vals_pd = init_vals_pop + init_vals_sigma + init_vals_theta
        col_order = init_vals_pop[0].columns
        init_vals_pd = pd.concat(init_vals_pd)[col_order].copy()
        #self.n_optimized_coeff = len(init_vals)
        init_vals_pd = init_vals_pd.fillna(pd.NA)
        for c_idx, c in enumerate(cols.__dataclass_fields__):
            target_dtype = cols.pd_dtypes[c_idx]
            #this try except really seems to be a bug in pandas, iloc just
            #doesnt work
            try:
                init_vals_pd.iloc[:, [c_idx]] = init_vals_pd.iloc[:, [c_idx]].astype(target_dtype)
            except AttributeError:
                c = init_vals_pd.columns[c_idx]
                init_vals_pd.loc[:, [c]] = init_vals_pd.loc[:, [c]].astype(target_dtype)
        self._subject_intercept_detected  = subject_intercept_detected
        return init_vals_pd.reset_index(drop = True)
    
    def _unpack_init_vals(
        self,
    ):
        

        init_vals = [obj.optimization_init_val for obj in self.population_coeff]
        init_vals_pd = []
        cols = self.init_vals_pd_cols
        subject_intercept_detected = False
        #unpack the information related to each ODE parameter
        for pop_coeff in self.population_coeff:
            if pop_coeff.subject_level_intercept:
                subject_intercept_detected = True
            init_vals_pd.append(
                {   
                #the name of the ODE parameter
                cols.model_coeff: pop_coeff.coeff_name,
                #the name of the population component of the parameter's variance
                cols.log_name:pop_coeff.coeff_name + "_pop",
                cols.model_coeff_dep_var: None,
                cols.population_coeff: True,
                cols.model_error: False,
                cols.init_val: pop_coeff.optimization_init_val,
                cols.model_coeff_lower_bound: pop_coeff.optimization_lower_bound,
                cols.model_coeff_upper_bound: pop_coeff.optimization_upper_bound,
                cols.allometric: False,
                cols.allometric_norm_value: None,
                cols.subject_level_intercept: pop_coeff.subject_level_intercept,
                cols.subject_level_intercept_name: pop_coeff.subject_level_intercept_sd_name,
                cols.subject_level_intercept_sd_init_val: pop_coeff.subject_level_intercept_sd_init_val,
                cols.subject_level_intercept_init_vals_column_name: pop_coeff.subject_level_intercept_init_vals_column_name,
                cols.subject_level_intercect_sd_lower_bound: pop_coeff.subject_level_intercept_sd_lower_bound,
                cols.subject_level_intercect_sd_upper_bound: pop_coeff.subject_level_intercept_sd_upper_bound,
                }
            )
        if subject_intercept_detected or self.no_me_loss_needs_sigma:
            init_vals_pd.append(
                {
                    cols.model_coeff: self.model_error_sigma.coeff_name,
                    cols.log_name:self.model_error_sigma.coeff_name + '_const',
                    cols.model_coeff_dep_var: None,
                    cols.population_coeff: False,
                    cols.model_error: True,
                    cols.init_val: self.model_error_sigma.optimization_init_val,
                    cols.model_coeff_lower_bound: self.model_error_sigma.optimization_lower_bound,
                    cols.model_coeff_upper_bound: self.model_error_sigma.optimization_upper_bound,
                    cols.allometric: False,
                    cols.allometric_norm_value: None,
                    cols.subject_level_intercept: self.model_error_sigma.subject_level_intercept,
                    cols.subject_level_intercept_name: self.model_error_sigma.subject_level_intercept_sd_name,
                    cols.subject_level_intercept_sd_init_val: self.model_error_sigma.subject_level_intercept_sd_init_val,
                    cols.subject_level_intercept_init_vals_column_name: self.model_error_sigma.subject_level_intercept_init_vals_column_name,
                    cols.subject_level_intercect_sd_lower_bound: self.model_error_sigma.subject_level_intercept_sd_lower_bound,
                    cols.subject_level_intercect_sd_upper_bound: self.model_error_sigma.subject_level_intercept_sd_upper_bound,
                }
            )
        # unpack the dep vars for the population coeffs
        for model_coeff in self.dep_vars:
            coeff_dep_vars = self.dep_vars[model_coeff]
            init_vals.extend(
                [
                    coeff_obj.optimization_init_val
                    for coeff_obj in coeff_dep_vars
                    if isinstance(coeff_obj, ObjectiveFunctionColumn)
                ]
            )
            for coeff_dep_var in coeff_dep_vars:
                init_vals_pd.append(
                    {
                        cols.model_coeff: model_coeff,
                        cols.log_name: model_coeff + '__' + coeff_dep_var.column_name,
                        cols.model_coeff_dep_var: coeff_dep_var.column_name,
                        cols.population_coeff: False,
                        cols.model_error: False,
                        cols.init_val: coeff_dep_var.optimization_init_val,
                        cols.model_coeff_lower_bound: coeff_dep_var.optimization_lower_bound,
                        cols.model_coeff_upper_bound: coeff_dep_var.optimization_upper_bound,
                        cols.allometric: True
                        if coeff_dep_var.model_method == 'allometric'
                        else False,
                        cols.allometric_norm_value: coeff_dep_var.allometric_norm_value,
                        cols.subject_level_intercept: False,
                        cols.subject_level_intercept_name: None,
                        cols.subject_level_intercept_sd_init_val: None,
                        cols.subject_level_intercept_init_vals_column_name: None,
                        cols.subject_level_intercect_sd_lower_bound: None,
                        cols.subject_level_intercect_sd_upper_bound: None,
                    }
                )
        self.init_vals_pd = pd.DataFrame(init_vals_pd)
        self.n_optimized_coeff = len(init_vals)
        return np.array(init_vals, dtype=np.float64)


    def _unpack_upper_lower_bounds(self, model_error: PopulationCoeffcient):
        # pop coeff bounds
        bounds = [
            (obj.optimization_lower_bound, obj.optimization_upper_bound)
            for obj in self.population_coeff
        ]
        # then sigma
        sigma_case_1 = any(
            [obj.subject_level_intercept for obj in self.population_coeff]
        )
        sigma_case_2 = self.no_me_loss_needs_sigma
        if sigma_case_1 or sigma_case_2:
            bounds.append(
                (
                    model_error.optimization_lower_bound,
                    model_error.optimization_upper_bound,
                )
            )

        # then omega2s
        bounds.extend(
            [
                (
                    obj.subject_level_intercept_sd_lower_bound,
                    obj.subject_level_intercept_sd_upper_bound,
                )
                for obj in self.population_coeff
                if obj.subject_level_intercept
            ]
        )

        # then dep var bounds
        for model_coeff in self.dep_vars:
            coeff_dep_vars = self.dep_vars[model_coeff]
            bounds.extend(
                [
                    (obj.optimization_lower_bound, obj.optimization_upper_bound)
                    for obj in coeff_dep_vars
                    if isinstance(obj, ObjectiveFunctionColumn)
                ]
            )
        return deepcopy(bounds)

    def _unpack_validate_params(self, params):
        population_coeff = deepcopy(self.population_coeff)
        dep_vars = deepcopy(self.dep_vars)
        for coeff_obj in population_coeff:
            if coeff_obj.coeff_name not in (i for i in dep_vars):
                dep_vars[coeff_obj.coeff_name] = []
        # Ensure the dep vars correspond to the pop coeff in the correct order
        assert [i.coeff_name for i in population_coeff] == [i for i in dep_vars]
        # change how this in unpacked to be based on the length of `population_coeff`
        # population_coeff_alt = {}  # overwrite the list with a dict in the same order
        for idx, coeff_obj in enumerate(population_coeff):
            coeff_obj.optimization_history.append(params[idx])
        self.population_coeff = deepcopy(population_coeff)
        self.dep_vars = deepcopy(dep_vars)

    def _populate_model_betas(self, other_params):
        self.n_model_vars = len([i for i in self.dep_vars])
        self.n_dep_vars_per_model_var = {
            i: len(self.dep_vars[i]) for i in self.dep_vars
        }
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
                betas[model_param][param_col] = ObjectiveFunctionBeta(
                    column_name=param_col,
                    model_method=param_col_obj.model_method,
                    value=other_params[other_params_idx],
                    allometric_norm_value=param_col_obj.allometric_norm_value,
                )
                param_col_obj.optimization_history.append(
                    other_params[other_params_idx]
                )
                params.append(param_col)
                other_params_idx = other_params_idx + 1
        self.betas = deepcopy(betas)
        self.dep_vars = deepcopy(self.dep_vars)
        return deepcopy(betas)

    def _subject_iterator(self, data):
        # data_out = {}
        subject_id_c = self.groupby_col
        # data_out['subject_id_c'] = deepcopy(self.groupby_col)
        conc_at_time_c = self.conc_at_time_col
        # data_out['conc_at_time_c'] = deepcopy(self.conc_at_time_col)
        # data_out['pk_model_function'] = deepcopy(self.pk_model_function)
        pk_model_function = deepcopy(self.pk_model_function)
        verbose = self.verbose

        population_coeff = deepcopy(self.population_coeff)
        # data_out['betas'] = deepcopy(self.betas)
        # data_out['time_c'] = deepcopy(self.time_col)
        betas = deepcopy(self.betas)
        time_col = deepcopy(self.time_col)
        subs = data[subject_id_c].unique()
        for subject in subs:
            data_out = {}
            data_out["subject_id_c"] = subject_id_c
            data_out["conc_at_time_c"] = conc_at_time_c
            data_out["pk_model_function"] = pk_model_function
            data_out["time_c"] = time_col
            data_out["betas"] = deepcopy(betas)
            subject_filt = data[subject_id_c] == subject
            subject_data = data.loc[subject_filt, :].copy()
            initial_conc = subject_data[conc_at_time_c].values[0]
            subject_coeff = deepcopy(population_coeff)
            # subject_coeff = deepcopy(population_coeff)
            subject_coeff = {
                obj.coeff_name: obj.optimization_history[-1] for obj in subject_coeff
            }
            # subject_coeff_history = [subject_coeff]
            data_out["subject_coeff"] = deepcopy(subject_coeff)
            data_out["subject_data"] = deepcopy(subject_data)
            data_out["initial_conc"] = deepcopy(initial_conc)
            yield deepcopy(data_out)

    #@profile
    def _homongenize_timepoints(self, ode_data, subject_data, subject_id_c, time_col):
        data = ode_data.copy()
        data["tmp"] = 1
        time_mask_df = data.pivot(
            index=subject_id_c, columns=time_col, values="tmp"
        ).fillna(0)
        self.time_mask = time_mask_df.to_numpy().astype(bool)
        #diffrax saveat times, that is: `diffrax.diffeqsolve(saveat = self.global_tp_eval)`
        self.global_tp_eval = np.array(time_mask_df.columns.values, dtype=np.float64)
        #jax wants the saveat times as a list containing 
        # all times where the ODE should be eval'd and reported at
        # so these three below are not relevant
        self.global_t0_eval = self.global_tp_eval[0]
        self.global_tf_eval = self.global_tp_eval[-1]
        self.global_tspan_eval = np.array(
            [self.global_t0_eval, self.global_tf_eval], dtype=np.float64
        )
        # this assumes that all of the subjects have the ODE init
        #diffrax t0 and t1, that is:
        #`diffrax.diffeqsolve(t0 = self.global_tspan_init[0],t1 = self.global_tspan_init[0] )`
        self.global_tspan_init = np.array(
            [self.ode_t0_time_val, self.global_tf_eval], dtype=np.float64
        )

    def _unpack_prepare_thetas(
        self, model_param_dep_vars: pd.DataFrame, subject_data: pd.DataFrame
    ):
        thetas = pd.DataFrame()
        theta_data = pd.DataFrame()
        ivc = self.init_vals_pd_cols
        for idx, row in model_param_dep_vars.iterrows():
            coeff_name = row[ivc.model_coeff]
            theta_name = row[ivc.model_coeff_dep_var]
            theta_is_allometric = row[ivc.allometric]
            col = (coeff_name, theta_name)
            if col not in thetas.columns:
                thetas[col] = [np.nan]
                thetas[col] = thetas[col].astype(pd.Float64Dtype())
            thetas[col] = [row[ivc.init_val]]
            if col not in theta_data.columns:
                theta_data[col] = np.repeat(np.nan, len(subject_data))
                theta_data[col] = thetas[col].astype(pd.Float64Dtype())
            if theta_is_allometric:
                theta_data_col = safe_signed_log(
                    subject_data[theta_name].values / row[ivc.allometric_norm_value]
                )
            else:
                theta_data_col = subject_data[theta_name].values
            theta_data[col] = theta_data_col
        thetas.columns = (
            pd.MultiIndex.from_tuples(thetas.columns)
            if len(thetas.columns) > 0
            else thetas.columns
        )
        theta_data.columns = (
            pd.MultiIndex.from_tuples(theta_data.columns)
            if len(theta_data.columns) > 0
            else theta_data.columns
        )
        return thetas, theta_data

    
    #@profile
    def _unpack_prepare_pop_coeffs(self, model_params: pd.DataFrame):
        pop_coeffs = pd.DataFrame()
        subject_level_intercept_sds = pd.DataFrame(dtype=pd.Float64Dtype())
        subject_level_intercept_init_vals = pd.DataFrame(dtype=pd.Float64Dtype())
        ivc = self.init_vals_pd_cols
        for idx, row in model_params.iterrows():
            coeff_name = row[ivc.model_coeff]
            if coeff_name not in pop_coeffs.columns:
                pop_coeffs[coeff_name] = [np.nan]
                pop_coeffs[coeff_name] = pop_coeffs[coeff_name].astype(
                    pd.Float64Dtype()
                )
            pop_coeffs[coeff_name] = [row[ivc.init_val]]

            if row[ivc.subject_level_intercept]: 
                omega_name = row[ivc.subject_level_intercept_name]
                col = (coeff_name, omega_name)
                if col not in subject_level_intercept_sds.columns:
                    subject_level_intercept_sds[col] = [np.nan]
                    subject_level_intercept_sds[col] = subject_level_intercept_sds[
                        col
                    ].astype(pd.Float64Dtype())
                subject_level_intercept_sds[col] = [
                    row[ivc.subject_level_intercept_sd_init_val] 
                ]  # this is terrible, referencing the names like this
                
                init_vals_col = row[ivc.subject_level_intercept_init_vals_column_name]
                if init_vals_col in self.subject_data.columns:
                    subject_level_intercept_init_vals[col] = (
                        self.subject_data[init_vals_col]
                        .copy()
                        .astype(pd.Float64Dtype())
                    )
        subject_level_intercept_sds.columns = (
            pd.MultiIndex.from_tuples(subject_level_intercept_sds.columns)
            if len(subject_level_intercept_sds.columns) > 0
            else subject_level_intercept_sds.columns
        )
        subject_level_intercept_init_vals.columns = (
            pd.MultiIndex.from_tuples(subject_level_intercept_init_vals.columns)
            if len(subject_level_intercept_init_vals.columns) > 0
            else subject_level_intercept_init_vals.columns
        )

        # if there are subject level intercepts, then we need to have model error sigma to perform FO, FOCE etc.
        sigma_case_1 = len(subject_level_intercept_sds.columns) > 0
        sigma_case_2 = self.no_me_loss_needs_sigma
        if sigma_case_1 or sigma_case_2:
            coeff_name = self.model_error_sigma.coeff_name
            if coeff_name not in pop_coeffs.columns:
                pop_coeffs[coeff_name] = [np.nan]
                pop_coeffs[coeff_name] = pop_coeffs[coeff_name].astype(
                    pd.Float64Dtype()
                )
            pop_coeffs[coeff_name] = self.model_error_sigma.optimization_init_val

        return pop_coeffs, (
            subject_level_intercept_sds,
            subject_level_intercept_init_vals,
        )

    #@profile
    def _assemble_pred_matrices(self, data):
        subject_id_c = self.groupby_col
        conc_at_time_c = self.conc_at_time_col
        time_col = deepcopy(self.time_col)
        data = data.reset_index(drop=True).copy()
        self.data = data.copy()
        subject_data = data.drop_duplicates(subset=subject_id_c, keep="first").copy()
        subject_data_alt = data.loc[data[self.time_col] == 0, :].copy()
        self.subject_data = subject_data.copy()
        #construct ode_data 
        #ode_data is the data at which the ode will be solved
        #When the initial concentration is known (eg. absorption models)
        #the ode_data is just the data, when the initial concentration is 
        #unknown then the ode_data starts at the first tp with known conc
        if self.solve_ode_at_time_col is None:
            ode_data = data.copy()
        else:
            ode_data = data.loc[data[self.solve_ode_at_time_col], :]
        # ode data is 'fixed' from here forward
        self.ode_data = ode_data.copy()
        verbose = self.verbose

        self.unique_groups = subject_data[
            subject_id_c
        ].unique()  # list of unique subjects in order
        self.y_groups = ode_data[
            subject_id_c
        ].to_numpy()  # subjects in order, repeated n timepoints time per subject
        self.y = ode_data[conc_at_time_c].to_numpy(dtype=np.float64)

        # y0 is the fist y where ODE solutions are generated, this may or may not be the solution to the ODE at t0
        # In the case of a model without absorption if all timepoints before
        #the notation is confusing here b/c I am mixing the minimizer and ivp solver param names
        first_pred_t_df = ode_data.drop_duplicates(subset=subject_id_c, keep="first")
        self.subject_y0 = first_pred_t_df[conc_at_time_c].to_numpy(dtype=np.float64)
        self._subject_y0 = (first_pred_t_df[[conc_at_time_c] + [self.time_col]]
                            .copy()
                            )
        self.subject_y0_idx = np.array(first_pred_t_df.index.values, dtype=np.int64)
        ode_init_val_cols = [i.column_name for i in self.ode_t0_cols]
        #initial conditions for the ODE solver
        self.ode_t0_vals = subject_data[ode_init_val_cols].reset_index(drop=True).copy()
        self._ode_t0_vals = (subject_data[ode_init_val_cols + [self.time_col]]
                             .reset_index(drop=True).copy())
        ode_saveat_min_time = self._subject_y0[self.time_col].to_numpy()
        ode_init_cond_time = self._ode_t0_vals[self.time_col].to_numpy()
        self.ode_t0_vals_are_subject_y0 = np.all(
             ode_saveat_min_time == ode_init_cond_time
        )
        self.ode_t0_vals_are_subject_y0 = False
        
        
        # n_subs = len(subs)
        init_vals = self.init_vals_pd.copy()
        ivc = self.init_vals_pd_cols
        #init_val should never be null, but just to be safe
        init_vals[ivc.init_val] = init_vals[ivc.init_val].fillna(0.0)
        model_params = init_vals.loc[init_vals[ivc.population_coeff], :]
        self.n_population_coeff = len(model_params)
        model_param_dep_vars = init_vals.loc[
            (init_vals[ivc.population_coeff] == False)
            & (init_vals[ivc.model_error] == False),
            :,
        ]

        self._homongenize_timepoints(ode_data, subject_data, subject_id_c, time_col)
        thetas, theta_data = self._unpack_prepare_thetas(
            model_param_dep_vars, subject_data
        )
        pop_coeffs, subject_level_intercept_info = self._unpack_prepare_pop_coeffs(
            model_params
        )
        subject_level_intercept_sds = subject_level_intercept_info[0]
        subject_level_intercept_init_vals = subject_level_intercept_info[
            1
        ]  # this is so bad, fast tho
        self.subject_level_intercept_init_vals = (
            subject_level_intercept_init_vals.copy()
        )
        self.n_subject_level_intercept_sds = len(subject_level_intercept_sds.columns)

        self.subject_level_intercept_sds = deepcopy(subject_level_intercept_sds)
        self.init_pop_coeffs = deepcopy(pop_coeffs)
        self.init_thetas = deepcopy(thetas)
        # this is too many things to return in this manner
        return (
            pop_coeffs.copy(),
            subject_level_intercept_sds.copy(),
            thetas.copy(),
            theta_data.copy(),
        )
    

    #@profile
    def _generate_pk_model_coeff_vectorized(
        self, pop_coeffs, thetas, theta_data, expected_len_out=None
    ):
        use_jax = False
        expected_len_out = (
                len(self.subject_y0) if expected_len_out is None else expected_len_out
            )
        pop_coeff_cols = pop_coeffs.columns
        if not use_jax:
            
            model_coeffs = pd.DataFrame(dtype=pd.Float64Dtype())
            for c in pop_coeff_cols:
                pop_coeff = pop_coeffs[c].to_numpy(dtype = np.float64)
                theta = (
                    thetas[c].to_numpy(dtype = np.float64).flatten()
                    if c in thetas.columns
                    else np.zeros_like(pop_coeff)
                )
                X = (
                    theta_data[c].to_numpy(dtype = np.float64)
                    if c in theta_data.columns
                    else np.zeros_like(pop_coeff)
                )
                data_contribution = (X @ theta)
                out = np.exp(data_contribution + pop_coeff)   + 1e-6
                if len(out) != expected_len_out:
                    out = np.repeat(out, expected_len_out)
                if c not in model_coeffs.columns:
                    model_coeffs[c] = np.repeat(np.nan, expected_len_out)
                    model_coeffs[c] = model_coeffs[c].astype(pd.Float64Dtype())
                model_coeffs[c] = out
            if len(model_coeffs) != expected_len_out:
                tmp_coeff = np.tile(model_coeffs.values, (expected_len_out, 1))
                model_coeffs = pd.DataFrame(
                    tmp_coeff, columns=model_coeffs.columns, dtype=pd.Float64Dtype()
                )
        if use_jax:
            #wrap a version of above in a function
            #jit that function
            #store the func if this is the first time this code is run
            #use the stored or just-constructed jitted f to do the matrix multiplication
            raise NotImplementedError
            
        return model_coeffs.copy()

    #@profile
    def _solve_ivp_parallel(
        self, y0, args, tspan=None, teval=None, fun=None, method=None
    ):
        return solve_ivp(fun, tspan, y0, t_eval=teval, method=method, args=(*args,))

    def _solve_ivp_parallel2(
        self, y0, args, tspan=None, teval=None, ode_class: PKBaseODE = None, method=None
    ):
        pred_obj = solve_ivp(
            ode_class.ode, tspan, y0, t_eval=teval, method=method, args=(*args,)
        )
        pred_obj.y[0] = ode_class.mass_to_depvar(pred_obj.y[0], *args)
        return pred_obj
    
    @staticmethod
    def _solve_ivp_jax_worker(y0, args,
                       tspan=None,
                       teval=None,
                       dt0 = None,
                       ode_class: PKBaseODE = None,
                       diffrax_solver=None, 
                       diffrax_step_ctrl = None,
                       diffrax_max_steps = None
                       ):
        
        ode_term = diffrax.ODETerm(ode_class.diffrax_ode)
        
        solution = diffrax.diffeqsolve(
        terms = ode_term,
        solver = diffrax_solver,
        t0 = tspan[0],
        t1 = tspan[1],
        dt0 = dt0,
        y0 = y0,
        args=args,
        max_steps=diffrax_max_steps,
        saveat=diffrax.SaveAt(ts=teval), # Specify time points for output
        stepsize_controller=diffrax_step_ctrl
        )
        
        central_mass_trajectory = solution.ys[:, 0]
        concentrations = ode_class.diffrax_mass_to_depvar(
        central_mass_trajectory, 
        args # Pass the same parameter tuple
    )
        return solution.ys, concentrations
    
    def _compile_jax_ivp_solvers(self,
        ode_t0_vals=None,
        timepoints=None,
        **kwargs):
        ode_t0_vals = self.ode_t0_vals if ode_t0_vals is None else ode_t0_vals
        ode_t0_vals =  ode_t0_vals.to_numpy()
        maxsteps = 1000000
        
        #compile both types of solvers regardless of if we need them
        #consider moving this to __init__
        if not (self.jax_ivp_nonstiff_solver_is_compiled):
                diffrax_solver = diffrax.Tsit5()
                #Initial thoughts:
                #when the data is quite noisy to begin with
                #(eg. biomarker or drug level determinations)
                #I bet rtol and atol can be much higher
                #Later conclusion/hypothesis:
                #Upon researching further, this is not where
                #we should comprimise on precision, rather to speed
                #things up the tol of the outer optimizer should be 
                #increased as was done for the profiling optimizer. 
                diffrax_step_ctrl = diffrax.PIDController(rtol=1e-8, atol=1e-10)
                dt0 = 0.1
                
                partial_solve_ivp = partial(
                    self._solve_ivp_jax_worker,
                    ode_class=self.pk_model_class,
                    tspan=self.global_tspan_init,
                    teval=self.global_tp_eval if timepoints is None else timepoints,
                    diffrax_solver=diffrax_solver,
                    diffrax_step_ctrl = diffrax_step_ctrl,
                    dt0 = dt0, 
                    diffrax_max_steps = maxsteps
                    
                )
                
                vmapped_solve = jax.vmap(partial_solve_ivp, in_axes=(0, 0,) )
                jit_vmapped_solve = jax.jit(vmapped_solve)
                self.jax_ivp_nonstiff_jittable_ = vmapped_solve
                self.jax_ivp_nonstiff_compiled_solver_ = jit_vmapped_solve
                self.jax_ivp_nonstiff_solver_is_compiled = True
                print('Sucessfully complied non-stiff ODE solver')
        
        if not (self.jax_ivp_stiff_solver_is_compiled):
                diffrax_solver = diffrax.Kvaerno5()
                
                diffrax_step_ctrl = diffrax.PIDController(rtol=1e-8, atol=1e-10)
                dt0 = 0.1
                
                partial_solve_ivp = partial(
                    self._solve_ivp_jax_worker,
                    ode_class=self.pk_model_class,
                    tspan=self.global_tspan_init,
                    teval=self.global_tp_eval if timepoints is None else timepoints,
                    diffrax_solver=diffrax_solver,
                    diffrax_step_ctrl = diffrax_step_ctrl,
                    dt0 = dt0, 
                    diffrax_max_steps = maxsteps
                    
                )
                
                vmapped_solve = jax.vmap(partial_solve_ivp, in_axes=(0, 0,) )
                jit_vmapped_solve = jax.jit(vmapped_solve)
                self.jax_ivp_stiff_jittable_ = vmapped_solve
                self.jax_ivp_stiff_compiled_solver_ = jit_vmapped_solve
                self.jax_ivp_stiff_solver_is_compiled = True
                print('Sucessfully complied stiff ODE solver')
        
        if not (self.jax_ivp_pymcstiff_solver_is_compiled):
                diffrax_solver = diffrax.Tsit5()
                #Initial thoughts:
                #when the data is quite noisy to begin with
                #(eg. biomarker or drug level determinations)
                #I bet rtol and atol can be much higher
                #Later conclusion/hypothesis:
                #Upon researching further, this is not where
                #we should compromise on precision, rather to speed
                #things up the tol of the outer optimizer should be 
                #increased as was done for the profiling optimizer. 
                diffrax_step_ctrl = diffrax.ConstantStepSize()
                dt0 = 0.1
                
                partial_solve_ivp = partial(
                    self._solve_ivp_jax_worker,
                    ode_class=self.pk_model_class,
                    tspan=self.global_tspan_init,
                    teval=self.global_tp_eval if timepoints is None else timepoints,
                    diffrax_solver=diffrax_solver,
                    diffrax_step_ctrl = diffrax_step_ctrl,
                    dt0 = dt0, 
                    diffrax_max_steps = maxsteps
                    
                )
                
                vmapped_solve = jax.vmap(partial_solve_ivp, in_axes=(0, 0,) )
                jit_vmapped_solve = jax.jit(vmapped_solve)
                self.jax_ivp_pymcstiff_jittable_ = vmapped_solve
                self.jax_ivp_pymcstiff_compiled_solver_ = jit_vmapped_solve
                self.jax_ivp_pymcstiff_solver_is_compiled = True
                print('Sucessfully complied stiff PyMC ODE solver')
        
        
    
    def _solve_ivp_jax(self,
        model_coeffs,
        ode_t0_vals=None,
        time_mask=None,
        timepoints=None,
        stiff_ode = False,
        **kwargs):
        
        ode_t0_vals = self.ode_t0_vals if ode_t0_vals is None else ode_t0_vals
        time_mask = self.time_mask if time_mask is None else time_mask
        model_coeffs, ode_t0_vals = model_coeffs.to_numpy(), ode_t0_vals.to_numpy()
        maxsteps = 1000000
        
        #compile both types of solvers regardless of if we need them
        #consider moving this to __init__
        if not (self.jax_ivp_nonstiff_solver_is_compiled):
                diffrax_solver = diffrax.Tsit5()
                #Initial thoughts:
                #when the data is quite noisy to begin with
                #(eg. biomarker or drug level determinations)
                #I bet rtol and atol can be much higher
                #Later conclusion/hypothesis:
                #Upon researching further, this is not where
                #we should comprimise on precision, rather to speed
                #things up the tol of the outer optimizer should be 
                #increased as was done for the profiling optimizer. 
                diffrax_step_ctrl = diffrax.PIDController(rtol=1e-8, atol=1e-10)
                dt0 = 0.1
                
                partial_solve_ivp = partial(
                    self._solve_ivp_jax_worker,
                    ode_class=self.pk_model_class,
                    tspan=self.global_tspan_init,
                    teval=self.global_tp_eval if timepoints is None else timepoints,
                    diffrax_solver=diffrax_solver,
                    diffrax_step_ctrl = diffrax_step_ctrl,
                    dt0 = dt0, 
                    diffrax_max_steps = maxsteps
                    
                )
                
                vmapped_solve = jax.vmap(partial_solve_ivp, in_axes=(0, 0,) )
                jit_vmapped_solve = jax.jit(vmapped_solve)
                self.jax_ivp_nonstiff_jittable_ = vmapped_solve
                self.jax_ivp_nonstiff_compiled_solver_ = jit_vmapped_solve
                self.jax_ivp_nonstiff_solver_is_compiled = True
                print('Sucessfully complied non-stiff ODE solver')
        
        if not (self.jax_ivp_stiff_solver_is_compiled):
                diffrax_solver = diffrax.Kvaerno5()
                
                diffrax_step_ctrl = diffrax.PIDController(rtol=1e-8, atol=1e-10)
                dt0 = 0.1
                
                partial_solve_ivp = partial(
                    self._solve_ivp_jax_worker,
                    ode_class=self.pk_model_class,
                    tspan=self.global_tspan_init,
                    teval=self.global_tp_eval if timepoints is None else timepoints,
                    diffrax_solver=diffrax_solver,
                    diffrax_step_ctrl = diffrax_step_ctrl,
                    dt0 = dt0, 
                    diffrax_max_steps = maxsteps
                    
                )
                
                vmapped_solve = jax.vmap(partial_solve_ivp, in_axes=(0, 0,) )
                jit_vmapped_solve = jax.jit(vmapped_solve)
                self.jax_ivp_stiff_jittable_ = vmapped_solve
                self.jax_ivp_stiff_compiled_solver_ = jit_vmapped_solve
                self.jax_ivp_stiff_solver_is_compiled = True
                print('Sucessfully complied stiff ODE solver')
        
        if not (self.jax_ivp_pymcstiff_solver_is_compiled):
                diffrax_solver = diffrax.Tsit5()
                #Initial thoughts:
                #when the data is quite noisy to begin with
                #(eg. biomarker or drug level determinations)
                #I bet rtol and atol can be much higher
                #Later conclusion/hypothesis:
                #Upon researching further, this is not where
                #we should compromise on precision, rather to speed
                #things up the tol of the outer optimizer should be 
                #increased as was done for the profiling optimizer. 
                diffrax_step_ctrl = diffrax.ConstantStepSize()
                dt0 = 0.1
                
                partial_solve_ivp = partial(
                    self._solve_ivp_jax_worker,
                    ode_class=self.pk_model_class,
                    tspan=self.global_tspan_init,
                    teval=self.global_tp_eval if timepoints is None else timepoints,
                    diffrax_solver=diffrax_solver,
                    diffrax_step_ctrl = diffrax_step_ctrl,
                    dt0 = dt0, 
                    diffrax_max_steps = maxsteps
                    
                )
                
                vmapped_solve = jax.vmap(partial_solve_ivp, in_axes=(0, 0,) )
                jit_vmapped_solve = jax.jit(vmapped_solve)
                self.jax_ivp_pymcstiff_jittable_ = vmapped_solve
                self.jax_ivp_pymcstiff_compiled_solver_ = jit_vmapped_solve
                self.jax_ivp_pymcstiff_solver_is_compiled = True
                print('Sucessfully complied stiff PyMC ODE solver')
        
        #get the relevant solver
        if self.stiff_ode:
                jit_vmapped_solve = self.jax_ivp_stiff_compiled_solver_
        if not self.stiff_ode:
                jit_vmapped_solve = self.jax_ivp_nonstiff_compiled_solver_


            
        ode_t0_vals = np.array(ode_t0_vals, dtype = np.float64)
        model_coeffs = np.array(model_coeffs, dtype = np.float64)
        all_solutions_masses, all_concentrations = jit_vmapped_solve(
                    ode_t0_vals,
                    model_coeffs
                    )
        
        return all_solutions_masses, all_concentrations
        
        # . . .
        
    
    #@profile
    def _solve_ivp(
        self,
        model_coeffs,
        ode_t0_vals=None,
        time_mask=None,
        parallel=False,
        parallel_n_jobs=None,
        timepoints=None,
        stiff_ode = False,
    ):
        ode_t0_vals = self.ode_t0_vals if ode_t0_vals is None else ode_t0_vals
        time_mask = self.time_mask if time_mask is None else time_mask
        
        use_jax = True
        parallel = False
        parallel_n_jobs = 3
        if use_jax:
            masses, concs = self._solve_ivp_jax(
                model_coeffs = model_coeffs,
                ode_t0_vals = ode_t0_vals, 
                stiff_ode = stiff_ode
            )
            sol_full = np.array(masses, dtype = np.float64)
            sol_full[:,:,0] = concs
            
            sol_dep_var = concs
            sol_dep_var = np.concatenate(sol_dep_var)
            sol_full = np.vstack(sol_full)
        else:
            iter_obj = zip(model_coeffs.iterrows(), ode_t0_vals.iterrows())
            partial_solve_ivp = partial(
                self._solve_ivp_parallel2,
                ode_class=self.pk_model_class,
                tspan=self.global_tspan_init,
                teval=self.global_tp_eval if timepoints is None else timepoints,
                method=self.ode_solver_method,
            )
            if parallel:
                sol_full = Parallel(n_jobs=parallel_n_jobs)(
                    delayed(partial_solve_ivp)(ode_inits_idx_row[1], coeff_idx_row[1])
                    for coeff_idx_row, ode_inits_idx_row in iter_obj
                )     
            else:
                list_comp = True
                if list_comp:
                    sol_full = [
                        partial_solve_ivp(ode_inits_idx_row[1], coeff_idx_row[1])
                        for coeff_idx_row, ode_inits_idx_row in iter_obj
                    ]

                else:
                    sol_full = []
                    for coeff_idx_row, ode_inits_idx_row in iter_obj:
                        ode_sol = partial_solve_ivp(ode_inits_idx_row[1], coeff_idx_row[1])
                        sol_full.append(ode_sol)
                        
            sol_dep_var = [sol_i.y[0] for sol_i in sol_full]
            sol_dep_var = np.concatenate(sol_dep_var)
            sol_full = np.vstack([i.y.T for i in sol_full])
        if timepoints is None:
            sol_dep_var = sol_dep_var[time_mask.flatten()]
            sol_full = sol_full[time_mask.flatten()]
        return sol_dep_var, sol_full
    
    def _unpack_no_me_params(self, params, beta_data):
        pop_coeffs = pd.DataFrame(
                params[: self.n_population_coeff].reshape(1, -1),
                dtype=pd.Float64Dtype(),
                columns=self.init_pop_coeffs.columns,
            )
        thetas = pd.DataFrame(
            params[self.n_population_coeff :].reshape(1, -1),
            dtype=pd.Float64Dtype(),
            columns=self.init_thetas.columns,
        )
        model_coeffs = self._generate_pk_model_coeff_vectorized(
            pop_coeffs, thetas, beta_data
        )
        return [model_coeffs, thetas, beta_data]
    
    def _unpack_me_params(self, ):
        raise NotImplementedError
        
    #@profile
    @staticmethod
    def _me_objective_jax(params:Dict[str, jnp.array],
                          theta_data,
                          subject_level_effect_init_vals, 
                          n_subject_level_effects, 
                          n_population_coeff,
                          loss_function,
                          ):
        
        

        pop_coeffs = {i:params[i] 
                      for idx, i in enumerate(params) 
                      if idx < n_population_coeff}
        
        start_idx = n_population_coeff
        end_idx = start_idx + 1
        sigma = {i:params[i] 
                      for idx, i in enumerate(params) 
                      if idx >= start_idx and idx < end_idx}
        
        start_idx = end_idx
        end_idx = start_idx + n_subject_level_effects
        omegas = {i:params[i] 
                      for idx, i in enumerate(params) 
                      if idx >= start_idx and idx < end_idx}
        
        start_idx = end_idx
        thetas = {i:params[i] 
                      for idx, i in enumerate(params) 
                      if idx >= start_idx}
        
        
        error, _, preds = loss_function( #preds is a tuple containing the (dep_var_preds, full_preds)
                    pop_coeffs,
                    sigma,
                    omegas,
                    thetas,
                    theta_data,
                    self,
                    FO_b_i_apprx=subject_level_effect_init_vals,
                    stiff_ode = stiff_ode,
                )
    
    def _objective_function2(
        self,
        params,
        beta_data,
        subject_level_intercept_init_vals=None,
        parallel=None,
        parallel_n_jobs=None,
        stiff_ode = False,
    ):
        # If we do not need to unpack sigma (ie. when the loss is just SSE, MSE, Huber etc)
        if (self.n_subject_level_intercept_sds == 0) and (
            not self.no_me_loss_needs_sigma
        ):
            pop_coeffs = pd.DataFrame(
                params[: self.n_population_coeff].reshape(1, -1),
                dtype=pd.Float64Dtype(),
                columns=self.init_pop_coeffs.columns,
            )
            thetas = pd.DataFrame(
                params[self.n_population_coeff :].reshape(1, -1),
                dtype=pd.Float64Dtype(),
                columns=self.init_thetas.columns,
            )
            model_coeffs = self._generate_pk_model_coeff_vectorized(
                pop_coeffs, thetas, beta_data
            )
            preds, full_preds = self._solve_ivp(
                model_coeffs, parallel=parallel, parallel_n_jobs=parallel_n_jobs, stiff_ode = stiff_ode
            )
            error = self.no_me_loss_function(self.y, preds, **self.no_me_loss_params)
        # If we DO need to unpack sigma, even without omegas, the unpacking logic is the same (ie. log-likelihood w/ or w/o mixed effects)
        elif (self.n_subject_level_intercept_sds > 0) or self.no_me_loss_needs_sigma:
            n_pop_c = self.n_population_coeff
            pop_coeffs = pd.DataFrame(
                params[:n_pop_c].reshape(1, -1),
                dtype=pd.Float64Dtype(),
                columns=self.init_pop_coeffs.columns[:n_pop_c],
            )
            start_idx = n_pop_c
            end_idx = start_idx + 1
            sigma = pd.DataFrame(
                params[start_idx:end_idx].reshape(1, -1),
                dtype=pd.Float64Dtype(),
                columns=self.init_pop_coeffs.columns[start_idx:end_idx],
            )
            start_idx = end_idx
            end_idx = start_idx + self.n_subject_level_intercept_sds
            omegas = pd.DataFrame(
                params[start_idx:end_idx].reshape(1, -1),
                dtype=pd.Float64Dtype(),
                columns=self.subject_level_intercept_sds.columns,
            )
            start_idx = end_idx
            thetas = pd.DataFrame(
                params[start_idx:].reshape(1, -1),
                dtype=pd.Float64Dtype(),
                columns=self.init_thetas.columns,
            )

            if (self.no_me_loss_needs_sigma) and (
                self.n_subject_level_intercept_sds == 0
            ):
                model_coeffs = self._generate_pk_model_coeff_vectorized(
                    pop_coeffs, thetas, beta_data
                )
                preds, full_preds = self._solve_ivp(
                    model_coeffs, parallel=parallel, parallel_n_jobs=parallel_n_jobs, stiff_ode = stiff_ode
                )
                error = self.no_me_loss_function(
                    self.y, preds, sigma, **self.no_me_loss_params
                )
            else:
                error, _, preds = self.me_loss_function( #preds is a tuple containing the (dep_var_preds, full_preds)
                    pop_coeffs,
                    sigma,
                    omegas,
                    thetas,
                    beta_data,
                    self,
                    FO_b_i_apprx=subject_level_intercept_init_vals,
                    stiff_ode = stiff_ode,
                )
        # self.preds_opt_.append(preds)
        return error
    def generate_jb_dumpable_self(self,):
        unpickable_attr = ["_jit_vmapped_solve"]
        dump_obj = deepcopy(self)

        dump_obj.jax_ivp_nonstiff_compiled_solver_ = None
        dump_obj.jax_ivp_nonstiff_jittable_ = None
        dump_obj.jax_ivp_nonstiff_solver_is_compiled = False
        
        dump_obj.jax_ivp_stiff_compiled_solver_ = None
        dump_obj.jax_ivp_stiff_solver_is_compiled = False
        dump_obj.jax_ivp_stiff_jittable_ = None
        
        dump_obj.jax_ivp_pymcstiff_compiled_solver_ = None
        dump_obj.jax_ivp_pymcstiff_solver_is_compiled = False
        dump_obj.jax_ivp_pymcstiff_jittable_ = None
        
        return dump_obj
    
    def save_fitted_model(self, jb_file_name: str = None):
        
        dump_obj = self.generate_jb_dumpable_self()
        
        if dump_obj.fit_result_ is None:
            raise ValueError("The Model must be fit before saving a fitted model")
        save_dir = "logs/"
        if not os.path.exists("logs"):
            os.mkdir("logs")
        id_str = (
            jb_file_name
            if jb_file_name is not None
            else dump_obj._generate_fitted_model_name()
        )
        save_path = os.path.join(save_dir, f"fitted_model_{id_str}.jb")
        with open(save_path, "wb") as f:
            jb.dump(dump_obj, f)
        del(dump_obj)

    #@profile
    def predict2(
        self,
        data,
        predict_all_ode_outputs = False,
        parallel=None,
        parallel_n_jobs=None,
        timepoints=None,
        subject_level_prediction=True,
        predict_unknown_t0=False,
    ):
        #force recompilation of ivp solver in case the timepoints in the 
        #prediction data are not those the model was trained with
        self.jax_ivp_nonstiff_solver_is_compiled = False
        self.jax_ivp_stiff_solver_is_compiled = False
        self.jax_ivp_pymcstiff_solver_is_compiled = False
        
        if self.fit_result_ is None:
            raise ValueError("The Model must be fit before prediction")
        params = self.fit_result_["x"]
        _, _, _, beta_data = self._assemble_pred_matrices(data)
        ode_t0_vals_are_subject_y0_init_status = deepcopy(
            self.ode_t0_vals_are_subject_y0
        )
        if predict_unknown_t0:
            self.ode_t0_vals_are_subject_y0 = True
        if (self.n_subject_level_intercept_sds == 0) and (
            not self.no_me_loss_needs_sigma
        ):
            pop_coeffs = pd.DataFrame(
                params[: self.n_population_coeff].reshape(1, -1),
                dtype=pd.Float64Dtype(),
                columns=self.init_pop_coeffs.columns,
            )
            thetas = pd.DataFrame(
                params[self.n_population_coeff :].reshape(1, -1),
                dtype=pd.Float64Dtype(),
                columns=self.init_thetas.columns,
            )
            model_coeffs = self._generate_pk_model_coeff_vectorized(
                pop_coeffs, thetas, beta_data
            )
            preds, full_preds = self._solve_ivp(
                model_coeffs,
                parallel=parallel,
                parallel_n_jobs=parallel_n_jobs,
                timepoints=timepoints,
            )
            # hess_objective = self.no_me_loss_function
        elif (self.n_subject_level_intercept_sds > 0) or self.no_me_loss_needs_sigma:
            n_pop_c = self.n_population_coeff
            pop_coeffs = pd.DataFrame(
                params[:n_pop_c].reshape(1, -1),
                dtype=pd.Float64Dtype(),
                columns=self.init_pop_coeffs.columns[:n_pop_c],
            )
            start_idx = n_pop_c
            end_idx = start_idx + 1
            sigma = pd.DataFrame(
                params[start_idx:end_idx].reshape(1, -1),
                dtype=pd.Float64Dtype(),
                columns=self.init_pop_coeffs.columns[start_idx:end_idx],
            )
            start_idx = end_idx
            end_idx = start_idx + self.n_subject_level_intercept_sds
            omegas = pd.DataFrame(
                params[start_idx:end_idx].reshape(1, -1),
                dtype=pd.Float64Dtype(),
                columns=self.subject_level_intercept_sds.columns,
            )
            start_idx = end_idx
            thetas = pd.DataFrame(
                params[start_idx:].reshape(1, -1),
                dtype=pd.Float64Dtype(),
                columns=self.init_thetas.columns,
            )
            if (self.no_me_loss_needs_sigma) and (
                self.n_subject_level_intercept_sds == 0
            ):
                model_coeffs = self._generate_pk_model_coeff_vectorized(
                    pop_coeffs, thetas, beta_data
                )
                preds, full_preds = self._solve_ivp(
                    model_coeffs, parallel=parallel, parallel_n_jobs=parallel_n_jobs
                )
                subject_coeffs = model_coeffs
                # error = self.no_me_loss_function(self.y, preds, sigma, **self.no_me_loss_params)
            else:
                if self.b_i_approx is None:
                    error, b_i_approx, _ = self.me_loss_function(
                        pop_coeffs,
                        sigma,
                        omegas,
                        thetas,
                        beta_data,
                        self,
                        FO_b_i_apprx=self.subject_level_intercept_init_vals,
                        solve_for_omegas=True,
                    )
                else:
                    b_i_approx = self.b_i_approx
                b_i_approx = pd.DataFrame(
                    b_i_approx, dtype=pd.Float64Dtype(), columns=omegas.columns
                )
                pop_coeffs_i = pd.DataFrame(
                    dtype=pd.Float64Dtype(), columns=pop_coeffs.columns
                )
                assert len(pop_coeffs) == 1
                for c in pop_coeffs_i.columns:
                    if c in b_i_approx.columns and subject_level_prediction:
                        b_i_c = b_i_approx[c].values.flatten()
                        # b_i_c = np.repeat(0.0, len(b_i_approx))
                    else:
                        b_i_c = np.repeat(0.0, len(b_i_approx))
                    pop_coeffs_i[c] = (
                        np.repeat(pop_coeffs[c].values[0], len(b_i_approx)) + b_i_c
                    )
                model_coeffs = self._generate_pk_model_coeff_vectorized(
                    pop_coeffs_i, thetas, beta_data
                )
                preds, full_preds = self._solve_ivp(model_coeffs)
                self.b_i_approx = b_i_approx

            # CI95% construction
        self._fitted_subject_ode_params = model_coeffs.copy()
        
        self.ode_t0_vals_are_subject_y0 = ode_t0_vals_are_subject_y0_init_status
        if predict_all_ode_outputs:
            preds_out = full_preds
        else:
            preds_out = preds
        return preds_out

    def _validate_data_chronology(
        self, data: pd.DataFrame, update_data_chronology: bool = False
    ):
        if update_data_chronology:
            data = data.sort_values(by=[self.groupby_col, self.time_col]).copy()
            sorted_data = data
        else:
            sorted_data = data.sort_values(by=[self.groupby_col, self.time_col]).copy()
        if not np.all(sorted_data.values == data.values):
            error_str = f"""Incoming data should be sorted by `groupby_col` ({self.groupby_col}) then `time_col`({self.time_col}).
            Sort your pd.DataFrame data with `data.sort_values(by = [{self.groupby_col}, {self.time_col}])` or set the `fit` method's
            `update_data_chronology` argument to `True`.
            """
            raise ValueError(error_str)
        test_ode_t0_times = data.drop_duplicates(
            subset=self.groupby_col, keep="first"
        ).copy()[self.time_col]
        if isinstance(self.ode_t0_time_val, int) or isinstance(
            self.ode_t0_time_val, float
        ):
            declared_t0 = np.repeat(self.ode_t0_time_val, len(test_ode_t0_times))
        if not np.all(test_ode_t0_times == declared_t0):
            error_str = """The declared initial timepoint values (`ode_t0_time_val`) for ODE solving do not match the first timepoint seen 
            for each subject."""
            raise ValueError(error_str)
        return data

    #@profile
    def fit2(
        self,
        data,
        parallel=False,
        parallel_n_jobs=-1,
        n_iters_per_checkpoint=0,
        warm_start=False,
        fit_id: uuid.UUID = None,
        ci_level:np.float64 = 0.95, 
        stiff_ode = False,
    ) -> Self:
        self.stiff_ode = stiff_ode
        fit_id = uuid.uuid4() if fit_id is None else fit_id
        data = self._validate_data_chronology(data)
        pop_coeffs, omegas, thetas, theta_data = self._assemble_pred_matrices(data)
        subject_level_intercept_init_vals = self.subject_level_intercept_init_vals
        sigma_check_1 = len(omegas.values) > 0
        sigma_check_2 = self.no_me_loss_needs_sigma
        if sigma_check_1 or sigma_check_2:
            sigma = pop_coeffs.iloc[:, [-1]]
            pop_coeffs = pop_coeffs.iloc[:, :-1]
        init_params = [
            pop_coeffs.values,
        ]  # pop coeffs already includes sigma if needed, this is confusing
        init_params_jax = pop_coeffs.to_dict(orient = 'list')
        init_params_jax = {(i,f"{i}_pop"):jnp.array(init_params_jax[i]) for i in init_params_jax}
        param_names = [
            {
                "model_coeff": c,
                "log_name":c + "_pop",
                "population_coeff": True,
                "model_error": False,
                "subject_level_intercept": False,
                "coeff_dep_var": False,
                "model_coeff_dep_var": None,
                "subject_level_intercept_name": None,
            }
            for c in pop_coeffs.columns
        ]
        if sigma_check_1 or sigma_check_2:
            init_params.append(sigma.values.reshape(1, -1))
            sigma_jax = sigma.to_dict(orient = 'list')
            sigma_jax = {(i,f"{i}_const"):jnp.array(sigma_jax[i]) for i in sigma_jax}
            init_params_jax.update(sigma_jax)
            param_names.extend(
                [
                    {
                        "model_coeff": c,
                        "log_name":c + "_const",
                        "population_coeff": False,
                        "model_error": True,
                        "subject_level_intercept": False,
                        "coeff_dep_var": False,
                        "model_coeff_dep_var": None,
                        "subject_level_intercept_name": None,
                    }
                    for c in sigma.columns
                ]
            )
            model_has_subj_level_effects = True
            mlflow_loss = self.me_loss_function
        else:
            mlflow_loss = self.no_me_loss_function
            model_has_subj_level_effects = False
        if len(omegas.values) > 0:
            init_params.append(omegas.values)
            omega_jax = omegas.to_dict(orient = 'list')
            omega_jax = {i:jnp.array(omega_jax[i]) for i in omega_jax}
            init_params_jax.update(omega_jax)
            
            param_names.extend(
                [
                    {
                        "model_coeff": c_tuple[0],
                        "log_name": c_tuple[1],
                        "population_coeff": False,
                        "model_error": False,
                        "subject_level_intercept": True,
                        "coeff_dep_var": False,
                        "model_coeff_dep_var": None,
                        "subject_level_intercept_name": c_tuple[1],
                    }
                    for c_tuple in omegas.columns.to_list()
                ]
            )
        if len(thetas.values) > 0:
            init_params.append(thetas.values)
            theta_jax = thetas.to_dict(orient = 'list')
            theta_jax = {i:jnp.array(theta_jax[i]) for i in theta_jax}
            init_params_jax.update(theta_jax)
            param_names.extend(
                [
                    {
                        "model_coeff": c_tuple[0],
                        "log_name": c_tuple[0] + "__" + c_tuple[1],
                        "population_coeff": False,
                        "model_error": False,
                        "subject_level_intercept": False,
                        "coeff_dep_var": True,
                        "model_coeff_dep_var": c_tuple[1],
                        "subject_level_intercept_name": None,
                    }
                    for c_tuple in thetas.columns.to_list()
                ]
            )
        theta_data_jax = theta_data.to_dict(orient = 'list')
        theta_data_jax = {i:jnp.array(theta_data_jax[i]) for i in theta_data_jax}
        #it would be possible to construct this when the class is initialized from 
        #init vals pd, that would make it much easier to follow. 
        fit_result_summary = pd.DataFrame(
                param_names,
            )
        init_params = np.concatenate(init_params, axis=1, dtype=np.float64).flatten()
        fit_result_summary['init_val'] = init_params
        fit_result_summary['lower_bound'] = [i[0] for i in self.bounds]
        fit_result_summary['upper_bound'] = [i[1] for i in self.bounds]
        self.preds_opt_ = []
        debugging = True
        if debugging:
            return init_params_jax, theta_data_jax
        objective_function = partial(
            self._objective_function2,
            subject_level_intercept_init_vals=subject_level_intercept_init_vals,
            parallel=parallel,
            parallel_n_jobs=parallel_n_jobs,
        )
        self.fit_id = fit_id
        id_str = self._generate_fitted_model_name(ignore_fit_status=True)
        checkpoint_filename = f"{id_str}.jb"
        # checkpoint_filepath = os.path.join('logs', checkpoint_filename)
        #initalize mlflow
        self._initalize_mlflow_experiment()
        
        with mlflow.start_run(run_name=id_str) as run:
            ds = from_pandas(df = theta_data)
            mlflow.log_input(dataset=ds, context = 'training')
            
            init_mlf = from_pandas(df = fit_result_summary)
            mlflow.log_input(dataset=init_mlf, context = 'init_parms')
            for idx, row in fit_result_summary.iterrows():
                param_name = row['log_name']
                mlflow.log_param(f'{param_name}__idx', idx)
            
            
            init_mlf_alt = from_pandas(df = self.init_vals_pd)
            mlflow.log_input(dataset=init_mlf_alt, context = 'init_parms_alt')
            mlflow.log_table(data = self.init_vals_pd, artifact_file='init_vals_pd.json')
            non_subj_cols = ['init_val',
                             'model_coeff_lower_bound',
                             'model_coeff_upper_bound', 
                             
                             ]
            dep_var_cols = ['allometric', 
                             'allometric_norm_value']
            subj_eff_cols = [
                'subject_level_intercept_name', 
                'subject_level_intercept_sd_init_val', 
                'subject_level_intercept_init_vals_column_name', 
                'subject_level_intercect_sd_lower_bound', 
                'subject_level_intercect_sd_upper_bound'
            ]
            #subj_eff_cols = [i.replace('intercept', 'effect').replace('intercect', 'effect') 
            #                 for i in subj_eff_cols]
            tmp = self.init_vals_pd
            pop_coeffs_df = tmp.loc[tmp['population_coeff'], :]
            sigmas_df = tmp.loc[tmp['model_error'], :]
            dep_vars_df = tmp.loc[tmp['model_coeff_dep_var'].isnull() == False, :]
            subject_level_effect_df = tmp.loc[tmp['subject_level_intercept'], :]
            
            ode_coeffs = []
            dep_vars = []
            subject_level_eff = []
            for idx, row in self.init_vals_pd.iterrows():
                mlflow.log_param(f'opt_param_full_{idx}', row.to_dict())
                #logging if the row corresponds to a dependant variable for a ODE coeff 
                if row['model_coeff_dep_var'] is not None:
                    param_name = row['log_name']
                    loggables = non_subj_cols + dep_var_cols
                    [mlflow.log_param(f"{param_name}__{c}", row[c]) for c in loggables]
                    dep_vars.append(param_name)     
                if row['population_coeff']:
                    param_name = row['log_name']
                    loggables = non_subj_cols
                    [mlflow.log_param(f"{param_name}__{c}", row[c]) for c in loggables]
                    ode_coeffs.append(param_name)
                if row['model_error']:
                    param_name = row['log_name']
                    loggables = non_subj_cols
                    [mlflow.log_param(f"{param_name}__{c}", row[c]) for c in loggables]
                if row['subject_level_intercept']:
                    param_name = row['subject_level_intercept_name']
                    loggables = subj_eff_cols
                    [mlflow.log_param(f"{param_name}__{c}", row[c]) for c in loggables]
                    subject_level_eff.append(param_name)
            
            mlflow.log_param('ode_parameters', ode_coeffs)
            mlflow.log_param('fixed_effects', dep_vars)
            mlflow.log_param('mixed_effects', subject_level_eff)   

            
            ode_class_str = get_class_source_without_docstrings(self._pk_model_class)
            mlflow.log_text(ode_class_str, 'ode_definition.py')
            mlflow.log_param('ode_definition_hash',
                             generate_class_contents_hash(ode_class_str))
            mlflow.log_param('ode_class_name', self._pk_model_class.__name__)
            
            loss_str = get_function_source_without_docstrings_or_comments(mlflow_loss)
            mlflow.log_text(loss_str, 'loss_definition.py')
            mlflow.log_param('loss_definition_hash',
                             generate_class_contents_hash(loss_str))
            mlflow.log_param('loss_definition_name', mlflow_loss.__name__)

            mlflow.log_param('scipy_version', scipy.__version__)
            mlflow.log_param('scipy_minimize_method', self.minimize_method)
            mlflow.log_text(scipy.optimize.show_options('minimize', self.minimize_method, False), 'minimize_options.txt')
            
            mlflow.log_param('minimize_tol', self.optimizer_tol)
            mlflow.log_param('n_optimized_params', self.n_optimized_coeff)
            
            mlflow.log_param('model_has_subj_effects', model_has_subj_level_effects)
            
            mlf_callback = MLflowCallback(objective_name=mlflow_loss.__name__, 
                                          parameter_names=fit_result_summary['log_name'].values)
            fit_with_jax = False
            if fit_with_jax:
                raise NotImplementedError

            else:
                self.fit_result_ = optimize_with_checkpoint_joblib(
                    objective_function,
                    init_params,
                    n_checkpoint=n_iters_per_checkpoint,
                    checkpoint_filename=checkpoint_filename,
                    args=(theta_data,),
                    warm_start=warm_start,
                    minimize_method=self.minimize_method,
                    tol=self.optimizer_tol,
                    bounds=self.bounds,
                    callback = mlf_callback
                    
                )
            # hess_fun = nd.Hessian()
            mlflow.log_metric(f"final_{mlflow_loss.__name__}", self.fit_result_.fun)
            mlflow.log_metric(f"final_{self.loss_summary_name}", self.fit_result_.fun)
            mlflow.log_metric('final_nfev', self.fit_result_.nfev)
            mlflow.log_metric('final_nit', self.fit_result_.nit)
            mlflow.log_metric('final_success', bool(self.fit_result_.success))
            mlflow.log_text(str(self.fit_result_), 'OptimizeResult.txt')
            
            
            
            fit_result_summary["best_fit_param_val"] = pd.NA
            fit_result_summary["best_fit_param_val"] = fit_result_summary[
                "best_fit_param_val"
            ].astype(pd.Float64Dtype())
            fit_result_summary["best_fit_param_val"] = self.fit_result_.x
            if ci_level is not None:
                ci_df = self._generate_profile_ci(data=data)
                fit_result_summary = fit_result_summary.merge(
                    ci_df, how = 'left', right_index=True, left_index=True
                )
                ci_re = re.compile('^ci[0-9]{1,2}')
                ci_cols = [i for i in fit_result_summary.columns if bool(re.search(ci_re, i))]
                for idx, row in fit_result_summary.iterrows():
                    for c in ci_cols:
                        log_str = f"param_{row['log_name']}_{c}"
                        mlflow.log_metric(log_str, row[c])
            
            self.fit_result_summary_ = fit_result_summary.copy()
            mlflow.log_table(self.fit_result_summary_, 'fit_result_summary.json')
            
            # after fitting, predict2 to set self.ab_i_approx if the model was mixed effects
            #
            preds = self.predict2(data, parallel=parallel, parallel_n_jobs=parallel_n_jobs)
            pred_loss_sigma = (fit_result_summary
                               .loc[fit_result_summary['model_error'], 'best_fit_param_val'])
            pred_loss = neg2_log_likelihood_loss(self.y, preds, pred_loss_sigma)
            mlflow.log_metric('fit_predict_neg2ll', pred_loss)
            if len(omegas.values) > 0:
                mlflow.log_table(self.b_i_approx, 'subject_level_effects.json')

   
            mlflow.log_table(self._fitted_subject_ode_params, 'fitted_subject_ode_params_base.json')
            
            self._summary_pk_stats = (self._fitted_subject_ode_params
                                      .apply(self.pk_model_class.summary_calculations,
                                                                            axis = 1, result_type = 'expand'))
            #For some reason this writing and reloading is required to get the df to be serializable.
            #Related to: https://github.com/pandas-dev/pandas/issues/55490
            with BytesIO() as f:
                self._summary_pk_stats.to_csv(f,index=False)
                f.seek(0)
                self._summary_pk_stats = pd.read_csv(f, dtype = pd.Float64Dtype())
            mlflow.log_table(self._summary_pk_stats, 'fitted_subject_ode_params.json')
            self._summary_pk_stats_descr = self._summary_pk_stats.describe()
            mlflow.log_table(self._summary_pk_stats_descr, 'fitted_subject_ode_params_descr.json')
            for c in self._summary_pk_stats_descr.columns:
                tmp = self._summary_pk_stats_descr[c]
                [mlflow.log_metric(f"fitted_subject_{c}_{idx.replace('%', 'quantile')}", tmp[idx])
                 for idx in tmp.index[1:]] 
            
            
                
        return deepcopy(self)

    def _generate_profile_ci(self,data, parallel = True  ):
        if parallel:
            dump_obj = self.generate_jb_dumpable_self()
            n_jobs = int(np.min([multiprocessing.cpu_count(), len(dump_obj.fit_result_.x)]))
            res_list = Parallel(n_jobs=n_jobs)(delayed(construct_profile_ci)(dump_obj, data, idx,)
                                          for idx, val in enumerate(dump_obj.fit_result_.x))
            res_list = [i[0] for i in res_list]
        else:
            res_list = []
            for param_idx, param_val in enumerate(self.fit_result_.x):
                print(f"Profiling parameter:{param_idx}")
                res_list = construct_profile_ci(model_obj = self,
                                                        df = data,
                                                        param_index=param_idx,
                                                        result_list = res_list
                                                        )
            
        return pd.DataFrame(res_list,)
    
    def _generate_fitted_model_name(self, ignore_fit_status=False):
        if (self.fit_result_ is None) and not ignore_fit_status:
            raise ValueError(
                "The Model must be fit before generating a fitted model id"
            )
        else:
            return f"b-{str(self.batch_id)}_m-{str(self.model_name)}_f-{str(self.fit_id)}"


def fit_indiv_models(compartmental_model: CompartmentalModel, df: pd.DataFrame):
    df = df.copy()
    fits = []
    fit_res_dfs = []
    for sub in df["SUBJID"].unique():
        fit_df = df.loc[df["SUBJID"] == sub, :].copy()
        fit = compartmental_model.fit2(
            fit_df,
            checkpoint_filename=f"mod_fo_abs_indv{sub}_{now_str}.jb",
            n_iters_per_checkpoint=1,
            parallel=False,
            parallel_n_jobs=4,
        )
        fit_df["pred_y"] = fit.predict2(fit_df)
        fits.append(fit.fit_result_)
        fit_res_dfs.append(fit_df.copy())
    fit_res_df = pd.concat(fit_res_dfs)
