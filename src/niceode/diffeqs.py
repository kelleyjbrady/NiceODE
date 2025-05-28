from numba import njit
import abc
import numba
import warnings
import jax.numpy as jnp
import numpy as np

nca_docstring = """
    Relevant NCA Estimated Parameters:
        Non-Compartmental Analysis (NCA) provides model-independent estimates of
        drug exposure and disposition kinetics directly from concentration-time
        data. These estimates are valuable for summarizing data and informing
        initial parameter guesses or evaluating compartmental models. It is critical that 
        the units of the various inputs to the NCA estimation are compatible. For example, 
        when using NCA to estimate CL the y axis (concentration) of the conc:time profile 
        must have the same mass units as the dose used in the calculation (CL = Dose_IV / AUC_inf_IV). 
        Further, because it is fairly standard in PK literature to express volume/time as liters/hr 
        the units of time should be converted to hours and the units of volume should be converted to
        liters. You should always be cognizant and keep track of the units of your analysis 
        to ensure you are not generating nonsense.
        
        Key relevant NCA parameters include:

        - AUC_inf (Area Under the Concentration-Time Curve extrapolated to infinity):
            Represents total drug exposure. Calculated using the trapezoidal rule
            up to the last measurable concentration (AUC_last) plus an extrapolated
            area based on the terminal phase: AUC_inf = AUC_last + C_last / lambda_z.
            Units: concentration * time (e.g., mg*hr/L).

        - lambda_z (Terminal Elimination Rate Constant):
            Estimated from the slope of the terminal log-linear decline phase of the
            concentration-time curve via linear regression.
            Units: 1/time (e.g., 1/hr).

        - t_half_z (Terminal Half-life):
            The time required for the concentration to decrease by half during the
            terminal phase. Calculated as t_half_z = ln(2) / lambda_z.
            Units: time (e.g., hr).

        - CL (Total Clearance):
            The volume of plasma (or blood) cleared of drug per unit time. Represents
            the overall efficiency of drug elimination.
            * For Intravenous (IV) data: CL = Dose_IV / AUC_inf_IV.
            * For Extravascular (EV) data: Only Apparent Clearance (CL/F) can be
                estimated without knowing bioavailability (F): CL/F = Dose_EV / AUC_inf_EV.
            Units: volume/time (e.g., L/hr).

        - Vz (Volume of Distribution during Terminal Phase):
            An apparent volume relating plasma concentration to the amount of drug
            in the body during the terminal elimination phase.
            * For IV data: Vz = CL / lambda_z.
            * For EV data: Apparent Volume (Vz/F) is estimated: Vz/F = (CL/F) / lambda_z.
            Units: volume (e.g., L).

        - Vdss (Volume of Distribution at Steady State):
            The theoretical volume the drug would occupy at steady state, reflecting
            the extent of distribution at equilibrium. Usually calculated from IV data
            using Mean Residence Time (MRT): Vdss = CL * MRT_IV = CL * (AUMC_inf_IV / AUC_inf_IV).
            For multi-compartment drugs, often Vdss <= Vz.
            Units: volume (e.g., L).

        - MRT (Mean Residence Time):
            The average time drug molecules spend in the body. Calculated from IV data as
            MRT_IV = AUMC_inf_IV / AUC_inf_IV, where AUMC is the Area Under the First
            Moment Curve (Concentration * time).
            Units: time (e.g., hr).

        - Cmax:
            The maximum observed concentration in the measured data.
            Units: concentration (e.g., mg/L).

        - Tmax:
            The time at which Cmax is observed. Primarily relevant for extravascular routes.
            Units: time (e.g., hr).
    """

class OneCompartmentFODiffEq(object):
    def __init__(self):
        self.params = {'cl': {
            'name': 'clearance',
            'def': 'clearance rate',
        }, 
                       'vd': {
            'name': 'volume of distribution',
            'def': 'volume of distribution'  
        }
        }
    def diff_eq(self):  
        return first_order_one_compartment_model


#@njit
def first_order_one_compartment_model2(t, y, cl, vd):
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
    dCdt = -(cl/vd) * C  # Calculate the rate of change
    return [dCdt]


#@njit
def first_order_one_compartment_model(t, y, k):
   
    C = y[0]  # Extract concentration from the state vector
    dCdt = -(k) * C  # Calculate the rate of change
    return [dCdt]

@njit
def mm_one_compartment_model(t, y, Vmax, Km):
    C = y[0]
    dCdt = - (Vmax * C) / (Km + C)
    return [dCdt]

#@njit 
def parallel_elim_one_compartment_model(t, C, K, Vmax, Km):
    dCdt = (-K * C) - ((Vmax * C) / (Km + C))
    return [dCdt]

class PKBaseODE(abc.ABC):
    """
    Abstract Base Class for Pharmacokinetic ODE models.

    This class defines the structure for PK models described by ordinary
    differential equations. Subclasses must implement the `ode` method,
    which defines the differential equations for the masses in each
    compartment, and the `mass_to_depvar` method, which converts the
    predicted mass in the observed compartment (usually central) to the
    measured dependent variable (usually concentration).

    Attributes:
        None explicitly defined in the base class.

    Methods:
        ode(t, y, *params): Defines the ODE system.
        mass_to_depvar(pred_mass_central, *params): Converts mass to concentration.
    """
    def __init__(self, ):
        pass

    @abc.abstractmethod
    def ode(self, t, y, *params):
        """
        Defines the system of ordinary differential equations.

        Args:
            t (float): Current time point.
            y (list or np.ndarray): Array of current state variables
                (masses or amounts in compartments). The order depends
                on the specific model implementation.
            *params: Sequence of model parameters required by the ODEs
                (e.g., ka, cl, vd). Order must match subclass definition.

        Returns:
            list or np.ndarray: List of derivatives [dy/dt] corresponding
                to the order of state variables in `y`.
        """
        pass

    @abc.abstractmethod
    def mass_to_depvar(self, pred_mass_central, *params):
        """
        Converts the predicted mass in the central/observed compartment
        to the dependent variable (usually concentration).

        Args:
            pred_mass_central (float or np.ndarray): Predicted mass or amount
                in the central (or observed) compartment, typically the output
                from an ODE solver corresponding to the first state variable.
            *params: Sequence of model parameters required for conversion,
                typically including the volume of the observed compartment
                (e.g., vd or v1). Order must match subclass definition.

        Returns:
            float or np.ndarray: The corresponding dependent variable value(s)
                (e.g., concentration = mass / volume).
        """
        pass
    @staticmethod
    @abc.abstractmethod
    def ode_for_diffrax(t, y, *params):
        """
        Defines the system of ordinary differential equations.

        Args:
            t (float): Current time point.
            y (list or np.ndarray): Array of current state variables
                (masses or amounts in compartments). The order depends
                on the specific model implementation.
            *params: Sequence of model parameters required by the ODEs
                (e.g., ka, cl, vd). Order must match subclass definition.

        Returns:
            list or np.ndarray: List of derivatives [dy/dt] corresponding
                to the order of state variables in `y`.
        """
        pass
    @staticmethod
    @abc.abstractmethod
    def convert_state_to_depvar( pred_mass_central, *params):
        """
        Converts the predicted mass in the central/observed compartment
        to the dependent variable (usually concentration).

        Args:
            pred_mass_central (float or np.ndarray): Predicted mass or amount
                in the central (or observed) compartment, typically the output
                from an ODE solver corresponding to the first state variable.
            *params: Sequence of model parameters required for conversion,
                typically including the volume of the observed compartment
                (e.g., vd or v1). Order must match subclass definition.

        Returns:
            float or np.ndarray: The corresponding dependent variable value(s)
                (e.g., concentration = mass / volume).
        """
        pass

    
class OneCompartmentConc(PKBaseODE):
    def __init__(self, ):
        pass
    def ode(self, t, y, cl, vd):
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
        dCdt = -(cl/vd) * C  # Calculate the rate of change
        return [dCdt]
    def mass_to_depvar(self, pred_mass, cl, vd):
        return pred_mass
    
    @staticmethod
    def ode_for_diffrax(t, y, args_tuple):
        """
        JAX-compatible ODE function for Diffrax.
        y[0] is Concentration (C).
        args_tuple: (cl, vd)
        """
        cl, vd = args_tuple
        C = y[0]
        # Add epsilon for numerical stability if vd can be very close to zero
        # vd_safe = jnp.where(jnp.abs(vd) < 1e-9, 1e-9, vd) 
        dCdt = -(cl / vd) * C
        return jnp.array([dCdt])

    @staticmethod
    def convert_state_to_depvar(pred_y_state0, args_tuple):
        """
        JAX-compatible conversion. For this model, y[0] is already concentration.
        pred_y_state0: Predicted concentration trajectory (C).
        args_tuple: (cl, vd) - not used in this specific conversion but kept for signature consistency.
        """
        # _cl, _vd = args_tuple # Unpack if needed, not in this case
        return pred_y_state0

class OneCompartmentAbsorption(PKBaseODE):
    """
    One-compartment model with first-order absorption (Gut -> Central).
    Parameterized by Ka, Apparent Clearance (CL/F), Apparent Volume (Vd/F).

    Represents oral or other extravascular administration with first-order
    absorption and first-order elimination from a single compartment. That is, 
    oral or extravascular administration where drug enters the central compartment
    via a first-order absorption process from a hypothetical 'gut' or depot compartment.
    This is a common baseline model for extravascular PK data.

    Key Model Assumptions:
        - First-order absorption process (rate proportional to amount at absorption site).
        - Single, well-stirred central compartment.
        - First-order elimination process (rate proportional to amount in central compartment).
        - Instantaneous distribution within the compartment.
        - Constant parameters over time and concentration range.

    States (y):
        y[0]: Mass in Central Compartment (amount)
        y[1]: Mass in Gut/Absorption Compartment (amount)
        Order: [Central, Gut]

    Parameters (*params for ode/mass_to_depvar):
        ka (float): First-order absorption rate constant (1/time).
        cl_f (float): Apparent clearance (CL/F) from the central compartment (volume/time).
                      CL is the true clearance, F is bioavailability.
        vd_f (float): Apparent volume of distribution (Vd/F) of the central compartment (volume).
                      Vd is the true volume.

    Units and Output:
        - Ensure consistency in units (e.g., hours, Liters, mg).
        - Typical units: ka(1/hr), cl_f(L/hr), vd_f(L).
        - The ODE solver returns mass in compartments [Central, Gut] over time.
        - `mass_to_depvar` converts Central mass (y[0]) to concentration using Vd/F.
          Concentration = Mass / (Vd/F).

    Handling Dosing in Simulations:
        - Dosing is handled *externally* to the ODE function by the solver setup.
        - For a dose `Dose` at t=0: Set initial conditions `y0 = [0, Dose]`.
        - For multiple doses: At each dose time, add `Dose` to the current mass in
          the Gut compartment (y[1]) and restart/continue the simulation.

    Common Derived Parameters:
        - Elimination rate constant: ke = cl_f / vd_f (= CL/Vd, independent of F).
        - Half-life (terminal): t1/2 = ln(2) / ke = ln(2) * (vd_f / cl_f).
        - Area Under Curve (AUC): AUC_inf = Dose / cl_f = (F * Dose) / CL.

    Initial Conditions (y0 for ODE solver):
        Typically, for a single dose `Dose` at t=0: y0 = [0, Dose]

    Initial Parameter Estimates (for optimization):
        - ka: Related to Tmax (time of peak concentration). Guess ~1/Tmax.
        - cl_f: Estimate from NCA: CL/F = Dose / AUC_inf.
        - vd_f: Estimate from NCA: Vd/F = (CL/F) / lambda_z (where lambda_z ≈ ke). lambda_z
            is the terminal slope of the ln(conc):time profile. Alternatively: Vd/F ~ Dose / (AUC*lambda_z). 
            Often larger than physiological volumes. Check literature for similar drugs.

        Optimizer Bounds:
         - Lower Bounds: ka, cl_f, vd_f typically set > small positive value (e.g., 1e-9).
        - Upper Bounds: Less critical, but use plausible limits.
           - ka: Can be large (e.g., 10 or 100 1/hr), but usually < 50.
           - cl_f: Should generally not exceed physiological limits like cardiac output or relevant organ blood flow (e.g., < 100-200 L/hr for human hepatic/renal flow).
           - vd_f: Highly variable, but unlikely to be thousands of L/kg. Can be informed by Vz/F from NCA. Start relatively large if unsure (e.g., 1000 L).


    Common Covariate Associations (Fixed Effects):
        - cl_f: CL often related to weight (allometry), renal/hepatic function, age, genetics. F affected by gut factors.
        - vd_f: Vd often related to weight (allometry), body composition, age. F affects apparent value.
        - ka: Formulation, food, GI conditions (transit time, pH), age.

    Parameterization of Random Effects (Mixed Effects):
        In population modeling, inter-individual variability (IIV) is estimated:
        - Concept: Assumes each subject's parameter value (P_i) varies around the
                   typical population value (P_pop), often modeled using a log-normal
                   distribution: P_i = P_pop * exp(eta_i), where eta_i ~ N(0, omega^2).
        - Estimated Term: omega^2 (variance) or omega (SD) of IIV.
        - Common Parameters with IIV: CL/F, Vd/F, Ka often show significant IIV.
        - Initial Estimates for omega (SD): Start ~0.3 (30% CV). Range 0.1-0.6.
        - Bounds for omega^2 (Variance): Lower > 0. Upper relatively large (e.g., 1-4).


              
    When to Prefer This Model:
        1. Default Choice: Often the first model tried for extravascular data
           due to its relative simplicity.
        2. Post-Absorption Decline: The concentration-time profile (semi-log plot)
           shows a reasonably linear terminal phase (mono-exponential decline)
           after Cmax, suggesting distribution is rapid compared to elimination or
           that a peripheral compartment is negligible.
        3. Simplicity Preferred: When data is too sparse or variable to reliably
           support a more complex model (like a two-compartment model).
        4. Statistical Justification: Model selection criteria (AIC, BIC) do not
           significantly favor a more complex model, or a simpler model (e.g.,
           assuming instantaneous absorption if Tmax is very early) is clearly inadequate.

    Potential Fitting Difficulties:
        1. Estimating Ka: Can be difficult if absorption is very fast (ka >> ke),
           making the absorption phase poorly defined ("bolus-like" input). If
           absorption is very slow (ka << ke, "flip-flop" kinetics), the terminal
           slope reflects ka, not ke, potentially confounding estimates if not recognized.
        2. Confounding Ka and Ke/CL/Vd: Especially with sparse data around Cmax,
           it can be hard to separate the rates of input (ka) and disposition (ke or cl/vd).
        3. Apparent Parameters: Without IV data or known bioavailability (F), only
           apparent clearance (CL/F) and volume (Vd/F) can be estimated. Changes
           in F cannot be distinguished from changes in CL or Vd.
        4. Data Quality: Requires sufficient sampling during absorption, peak, and
           elimination phases for reliable parameter estimation.

    Advanced Fitting Strategies:
        1. NCA Guidance: Use Non-Compartmental Analysis (NCA) results robustly for
           initial estimates: lambda_z informs ke (or cl/vd), AUC informs cl/F,
           Tmax informs ka.
        2. Sequential Fitting (with IV data): If IV data is available, fit a
           one-compartment IV bolus model first to obtain reliable estimates (or priors)
           for CL and Vd. Then, fit the oral data using this model, fixing or using
           strong priors for CL and Vd, primarily estimating ka and potentially F.
        3. Fixing Ka: If absorption is known to be very rapid from formulation
           properties or previous studies, ka might be fixed to a large value.
        4. Population Modeling: In population PK (PopPK), relationships between F, CL,
           and Vd can sometimes be disentangled using data from multiple dose levels
           or studies, potentially leveraging covariate information.
           
    Potential Model Extensions:
        - Add compartments (Two-compartment absorption model).
        - Implement non-linear elimination (Michaelis-Menten).
        - Add transit compartments for absorption delay.
        - Include effect compartment for PK/PD modeling.

    Parameter Sensitivity (Qualitative):
        - Increasing `ka`: Faster absorption, earlier Tmax, potentially higher Cmax.
        - Increasing `cl_f`: Lower AUC, lower Cmax, faster decline (shorter t1/2).
        - Increasing `vd_f`: Lower Cmax, slower decline (longer t1/2), no change to AUC.
    
    
    """
    def __init__(self, ):
        pass

    def ode(self, t, y, ka, cl, vd):
        central_mass, gut_mass = y
        # Elimination rate constant ke = cl / vd
        dCMdt = ka * gut_mass - (cl / vd) * central_mass
        dGdt = -ka * gut_mass
        return [dCMdt, dGdt]

    def mass_to_depvar(self, pred_mass_central, ka, cl, vd):
        """Converts central compartment mass to concentration."""
        # Concentration = Mass / Volume
        depvar_unit_result = pred_mass_central / vd
        return depvar_unit_result
    
    @staticmethod
    def ode_for_diffrax(t, y, args_tuple):
        """
        JAX-compatible ODE function for Diffrax.
        y = [central_mass, gut_mass]
        args_tuple: (ka, cl, vd)
        """
        ka, cl, vd = args_tuple
        central_mass, gut_mass = y
        # vd_safe = jnp.where(jnp.abs(vd) < 1e-9, 1e-9, vd)
        dCMdt = ka * gut_mass - (cl / vd) * central_mass
        dGdt = -ka * gut_mass
        return jnp.array([dCMdt, dGdt])

    @staticmethod
    def convert_state_to_depvar(pred_y_state0, args_tuple):
        """
        JAX-compatible conversion from central mass to concentration.
        pred_y_state0: Predicted central_mass trajectory.
        args_tuple: (ka, cl, vd)
        """
        _ka, _cl, vd = args_tuple
        # vd_safe = jnp.where(jnp.abs(vd) < 1e-9, 1e-9, vd)
        depvar_unit_result = pred_y_state0 / vd
        return depvar_unit_result
        


def one_compartment_absorption(t, y, ka, cl, vd):
    """One-compartment model with first-order absorption and elimination."""
    C, A = y
    dCdt = ka * A - (cl/vd) * C
    dAdt = -ka * A
    return [dCdt, dAdt]

def one_compartment_absorption2(t, y, ka, ke):
    """One-compartment model with first-order absorption and elimination."""
    C, A = y
    dCdt = ka * A - ke * C
    dAdt = -ka * A
    return [dCdt, dAdt]

class OneCompartmentAbsorption2(PKBaseODE):
    """
    One-compartment model with first-order absorption (Gut -> Central).
    Parameterized by Ka, Elimination Rate Constant (Ke), Apparent Volume (Vd/F).

    Represents oral or other extravascular administration with first-order
    absorption and first-order elimination from a single compartment. This uses
    the elimination rate constant (Ke) directly.

    Key Model Assumptions:
        - First-order absorption process.
        - Single, well-stirred central compartment.
        - First-order elimination process (parameterized by Ke).
        - Instantaneous distribution within the compartment.
        - Constant parameters over time and concentration range.

    States (y):
        y[0]: Mass in Central Compartment (amount)
        y[1]: Mass in Gut/Absorption Compartment (amount)
        Order: [Central, Gut]

    Parameters (*params for ode/mass_to_depvar):
        ka (float): First-order absorption rate constant (1/time).
        ke (float): First-order elimination rate constant (1/time). Note: Ke = CL/Vd.
        vd_f (float): Apparent volume of distribution (Vd/F) of the central compartment (volume).

    Units and Output:
        - Ensure consistency in units (e.g., hours, Liters, mg).
        - Typical units: ka(1/hr), ke(1/hr), vd_f(L).
        - ODE solver returns mass in compartments [Central, Gut] over time.
        - `mass_to_depvar` converts Central mass (y[0]) to concentration using Vd/F.
          Concentration = Mass / (Vd/F).

    Handling Dosing in Simulations:
        - Dosing handled externally by solver setup.
        - Dose `Dose` at t=0: `y0 = [0, Dose]`.
        - Multiple doses: Add `Dose` to Gut compartment mass (y[1]) at dose time.

    Common Derived Parameters:
        - Half-life (terminal): t1/2 = ln(2) / ke.
        - Apparent Clearance (CL/F): cl_f = ke * vd_f.
        - Area Under Curve (AUC): AUC_inf = Dose / cl_f = Dose / (ke * vd_f).

    Initial Conditions (y0 for ODE solver):
        Typically, for a single dose `Dose` at t=0: y0 = [0, Dose]

    Initial Parameter Estimates (for optimization):
        - ka: Related to Tmax. Guess ~1/Tmax.
        - ke: Estimate from terminal slope of ln(C) vs t: ke ≈ lambda_z.
        - vd_f: Estimate from NCA: Vd/F = (Dose / AUC_inf) / ke = CL/F / ke.

        Optimizer Bounds:
         - Lower Bounds: ka, ke, vd_f typically set > small positive value (e.g., 1e-9).
        - Upper Bounds: Less critical, but use plausible limits.
           - ka: Can be large (e.g., 10 or 100 1/hr), but usually < 50.
           - vd(/F): Highly variable, but unlikely to be thousands of L/kg. Can be informed by Vz/F from NCA. Start relatively large if unsure (e.g., 1000 L).
           - ke: Usually < 10 1/hr. Related to cl/vd bounds.

    Common Covariate Associations (Fixed Effects):
        - ke: As CL/Vd, influenced by factors affecting both (weight, organ function, age). Often modeled via CL and Vd covariates.
        - vd_f: Vd often related to weight, body composition, age. F affects apparent value.
        - ka: Formulation, food, GI conditions (transit time, pH), age.

    Parameterization of Random Effects (Mixed Effects):
        In population modeling, inter-individual variability (IIV) is estimated:
        - Concept: Assumes each subject's parameter value (P_i) varies around the
                   typical population value (P_pop), often modeled using a log-normal
                   distribution: P_i = P_pop * exp(eta_i), where eta_i ~ N(0, omega^2).
        - Estimated Term: omega^2 (variance) or omega (SD) of IIV.
        - Common Parameters with IIV: Ke (or CL/F), Vd/F, Ka often show IIV.
        - Initial Estimates for omega (SD): Start ~0.3 (30% CV).
        - Bounds for omega^2 (Variance): Lower > 0. Upper relatively large (e.g., 1-4).

    
    When to Prefer This Model:
        1. Default Choice: Often the first model tried for extravascular data
           due to its relative simplicity.
        2. Post-Absorption Decline: The concentration-time profile (semi-log plot)
           shows a reasonably linear terminal phase (mono-exponential decline)
           after Cmax, suggesting distribution is rapid compared to elimination or
           that a peripheral compartment is negligible.
        3. Simplicity Preferred: When data is too sparse or variable to reliably
           support a more complex model (like a two-compartment model).
        4. Statistical Justification: Model selection criteria (AIC, BIC) do not
           significantly favor a more complex model, or a simpler model (e.g.,
           assuming instantaneous absorption if Tmax is very early) is clearly inadequate.

    Potential Fitting Difficulties:
        1. Estimating Ka: Can be difficult if absorption is very fast (ka >> ke),
           making the absorption phase poorly defined ("bolus-like" input). If
           absorption is very slow (ka << ke, "flip-flop" kinetics), the terminal
           slope reflects ka, not ke, potentially confounding estimates if not recognized.
        2. Confounding Ka and Ke/CL/Vd: Especially with sparse data around Cmax,
           it can be hard to separate the rates of input (ka) and disposition (ke or cl/vd).
        3. Apparent Parameters: Without IV data or known bioavailability (F), only
           apparent clearance (CL/F) and volume (Vd/F) can be estimated. Changes
           in F cannot be distinguished from changes in CL or Vd.
        4. Data Quality: Requires sufficient sampling during absorption, peak, and
           elimination phases for reliable parameter estimation.

    Advanced Fitting Strategies:
        1. NCA Guidance: Use Non-Compartmental Analysis (NCA) results robustly for
           initial estimates: lambda_z informs ke (or cl/vd), AUC informs cl/F,
           Tmax informs ka.
        2. Sequential Fitting (with IV data): If IV data is available, fit a
           one-compartment IV bolus model first to obtain reliable estimates (or priors)
           for CL and Vd. Then, fit the oral data using this model, fixing or using
           strong priors for CL and Vd, primarily estimating ka and potentially F.
        3. Fixing Ka: If absorption is known to be very rapid from formulation
           properties or previous studies, ka might be fixed to a large value.
        4. Population Modeling: In population PK (PopPK), relationships between F, CL,
           and Vd can sometimes be disentangled using data from multiple dose levels
           or studies, potentially leveraging covariate information.
    
    Potential Model Extensions:
        - Add compartments (Two-compartment absorption model).
        - Implement non-linear elimination (Michaelis-Menten).
        - Add transit compartments for absorption delay.
        - Include effect compartment for PK/PD modeling.
    
    Parameter Sensitivity (Qualitative):
        - Increasing `ka`: Faster absorption, earlier Tmax, potentially higher Cmax.
        - Increasing `ke`: Faster elimination rate, shorter t1/2, lower AUC (assuming fixed Vd/F), lower Cmax.
        - Increasing `vd_f`: Lower concentrations overall (C = Mass / (Vd/F)), lower Cmax, longer time to
            eliminate a given mass but t1/2 depends only on Ke in this param. Increases apparent
            CL (CL/F = ke * Vd/F) leading to lower AUC.
        
    
    """
    def __init__(self, ):
        pass

    def ode(self, t, y, ka, ke, vd):
        central_mass, gut_mass = y
        dCMdt = ka * gut_mass - ke * central_mass
        dGdt = -ka * gut_mass
        return [dCMdt, dGdt]

    def mass_to_depvar(self, pred_mass_central, ka, ke, vd):
        """Converts central compartment mass to concentration."""
        depvar_unit_result = pred_mass_central / vd
        return depvar_unit_result
    
    @staticmethod
    def ode_for_diffrax(t, y, args_tuple):
        """
        JAX-compatible ODE function for Diffrax.
        y = [central_mass, gut_mass]
        args_tuple: (ka, ke, vd) - assuming ke = cl/vd from context
        """
        ka, ke, _vd = args_tuple # vd is not directly used if ke is cl/vd
        central_mass, gut_mass = y
        dCMdt = ka * gut_mass - ke * central_mass
        dGdt = -ka * gut_mass
        return jnp.array([dCMdt, dGdt])

    @staticmethod
    def convert_state_to_depvar(pred_y_state0, args_tuple):
        """
        JAX-compatible conversion from central mass to concentration.
        pred_y_state0: Predicted central_mass trajectory.
        args_tuple: (ka, ke, vd)
        """
        _ka, _ke, vd = args_tuple
        # vd_safe = jnp.where(jnp.abs(vd) < 1e-9, 1e-9, vd)
        depvar_unit_result = pred_y_state0 / vd
        return depvar_unit_result

class OneCompartmentBolus_CL(PKBaseODE):
    """
    One-compartment IV Bolus model. Parameterized by Clearance (CL) and Volume (Vd).

    Simplest model for IV bolus injection, assuming instantaneous distribution
    and first-order elimination. Forms the basis for many PK calculations.

    Key Model Assumptions:
        - Instantaneous administration and distribution (bolus input).
        - Single, well-stirred central compartment.
        - First-order elimination process (rate proportional to amount).
        - Constant parameters over time and concentration range.

    States (y):
        y[0]: Mass in Central Compartment (amount)
        Order: [Central]

    Parameters (*params for ode/mass_to_depvar):
        cl (float): Clearance from the central compartment (volume/time).
        vd (float): Volume of distribution of the central compartment (volume).

    Units and Output:
        - Ensure consistency (e.g., hr, L, mg). Typical units: cl(L/hr), vd(L).
        - ODE solver returns mass in the Central compartment (y[0]) over time.
        - `mass_to_depvar` converts Central mass to concentration using Vd.
          Concentration = Mass / Vd.

    Handling Dosing in Simulations:
        - Dosing handled externally by solver setup.
        - For dose `Dose` at t=0: Set initial conditions `y0 = [Dose]`.
        - Multiple doses: At dose time, set/add `Dose` to Central compartment mass (y[0]).

    Common Derived Parameters:
        - Elimination rate constant: ke = cl / vd.
        - Half-life: t1/2 = ln(2) / ke = ln(2) * vd / cl.
        - Area Under Curve (AUC): AUC_inf = Dose / cl.
        - Initial Concentration (theoretical): C0 = Dose / vd.

    Initial Conditions (y0 for ODE solver):
        For a single IV bolus dose `Dose` at t=0: y0 = [Dose]

    Initial Parameter Estimates (for optimization):
        - cl: Estimate from NCA: CL = Dose / AUC_inf.
        - vd: Estimate from NCA: Vd = CL / lambda_z (this Vd is Vz). lambda_z is the slope of the 
            terminal phase of the the ln(conc):time profile. 
            Alternatively: Vd ≈ Dose / C0 (C0 defined in `Common Derived Parameters`).

        Optimizer Bounds:
         - Lower Bounds: cl, vd typically set > small positive value (e.g., 1e-9).
         - Upper Bounds: Plausible physiological limits (CL < blood flow, Vd < huge).

    Common Covariate Associations (Fixed Effects):
        - cl: Weight (allometry), renal/hepatic function, age, genetics.
        - vd: Weight (allometry), body composition, age.

    Parameterization of Random Effects (Mixed Effects):
        - Concept: P_i = P_pop * exp(eta_i), eta_i ~ N(0, omega^2).
        - Estimated Term: omega^2 (variance) or omega (SD) of IIV.
        - Common Parameters with IIV: CL and Vd typically show IIV.
        - Initial Estimates for omega (SD): Start ~0.3 (30% CV).
        - Bounds for omega^2 (Variance): Lower > 0. Upper relatively large (e.g., 1-4).
    
    When to Prefer This Model:
        1. Rapid IV Administration: Applicable for IV bolus or very short infusions.
        2. Mono-exponential Decline: Concentration-time profile (semi-log plot)
           shows a single, linear decline phase, indicating negligible distribution
           effects or very rapid distribution compared to elimination.
        3. Simplicity/Data Sparsity: When data is insufficient to support a more
           complex multi-compartment model, or for initial characterization.
        4. Foundational Analysis: Often used as a building block or reference for
           understanding basic PK parameters (CL, Vd, half-life).

    Potential Fitting Difficulties:
        1. Model Mismatch: The primary difficulty arises if the *true* PK is
           multi-compartmental. Forcing a one-compartment fit to bi-exponential data
           will lead to poor model fit (systematic residuals) and biased parameter
           estimates (e.g., Vd estimate will be closer to Vz or Vdss, not V1; CL might
           also be biased depending on weighting).
        2. Estimating Vd from C0: Extrapolating back to C0 can be inaccurate if
           early samples are missing or if there's rapid initial distribution.
        3. Weighting Issues: Choice of data weighting can influence parameter
           estimates, especially if the concentration range is wide.

    Advanced Fitting Strategies:
        1. Strong NCA Guidance: NCA results (CL, Vd from terminal phase, lambda_z)
           are often very good initial estimates for this model if it's appropriate.
        2. Residual Analysis: Closely examine residual plots (predicted vs observed)
           for systematic trends. If trends indicate a distribution phase is missed,
           a multi-compartment model should be considered.
        3. Informing Multi-Compartment Models: The CL and Vd estimated here (often
           representing CL and Vz/Vdss) can provide starting points or bounds when
           developing a two-compartment model (where CL is the same, but Vd splits
           into V1 and V2).
    
    Potential Model Extensions:
        - Add peripheral compartments (Two-compartment bolus model).
        - Implement non-linear elimination (Michaelis-Menten).
        - Include effect compartment for PK/PD modeling.

    Parameter Sensitivity (Qualitative):
        - Increasing `cl`: Faster decline (shorter t1/2), lower AUC.
        - Increasing `vd`: Slower decline (longer t1/2), lower C0/initial concentrations, no change to AUC.
    
    
    """
    def __init__(self, ):
        pass

    def ode(self, t, y, cl, vd):
        """ODE system for 1-cmt IV bolus (CL, Vd)."""
        central_mass = y[0]
        # Elimination rate constant ke = cl / vd
        dCMdt = -(cl / vd) * central_mass
        return [dCMdt]

    def mass_to_depvar(self, pred_mass_central, cl, vd):
        """Converts central compartment mass to concentration."""
        # Concentration = Mass / Volume
        depvar_unit_result = pred_mass_central / vd
        return depvar_unit_result
    
    @staticmethod
    def ode_for_diffrax(t, y, args_tuple):
        """
        JAX-compatible ODE function for Diffrax.
        y[0] is central_mass.
        args_tuple: (cl, vd)
        """
        cl, vd = args_tuple
        central_mass = y[0]
        # vd_safe = jnp.where(jnp.abs(vd) < 1e-9, 1e-9, vd)
        dCMdt = -(cl / vd) * central_mass
        return jnp.array([dCMdt])

    @staticmethod
    def convert_state_to_depvar(pred_y_state0, args_tuple):
        """
        JAX-compatible conversion from central mass to concentration.
        pred_y_state0: Predicted central_mass trajectory.
        args_tuple: (cl, vd)
        """
        _cl, vd = args_tuple
        # vd_safe = jnp.where(jnp.abs(vd) < 1e-9, 1e-9, vd)
        depvar_unit_result = pred_y_state0 / vd
        return depvar_unit_result

class OneCompartmentBolus_Ke(PKBaseODE):
    """
    One-compartment IV Bolus model. Parameterized by Elimination Rate Constant (Ke) and Volume (Vd).

    Alternative parameterization for IV bolus, assuming instantaneous distribution
    and first-order elimination defined by Ke.

    Key Model Assumptions:
        - Instantaneous administration and distribution (bolus input).
        - Single, well-stirred central compartment.
        - First-order elimination process (parameterized by Ke).
        - Constant parameters over time and concentration range.

    States (y):
        y[0]: Mass in Central Compartment (amount)
        Order: [Central]

    Parameters (*params for ode/mass_to_depvar):
        ke (float): First-order elimination rate constant (1/time). Note: Ke = CL/Vd.
        vd (float): Volume of distribution of the central compartment (volume).

    Units and Output:
        - Ensure consistency (e.g., hr, L, mg). Typical units: ke(1/hr), vd(L).
        - ODE solver returns mass in the Central compartment (y[0]) over time.
        - `mass_to_depvar` converts Central mass to concentration using Vd.
          Concentration = Mass / Vd.

    Handling Dosing in Simulations:
        - Dosing handled externally by solver setup.
        - Dose `Dose` at t=0: `y0 = [Dose]`.
        - Multiple doses: Set/add `Dose` to Central compartment mass (y[0]) at dose time.

    Common Derived Parameters:
        - Half-life: t1/2 = ln(2) / ke.
        - Clearance: cl = ke * vd.
        - Area Under Curve (AUC): AUC_inf = Dose / cl = Dose / (ke * vd).
        - Initial Concentration (theoretical): C0 = Dose / vd.

    Initial Conditions (y0 for ODE solver):
        For a single IV bolus dose `Dose` at t=0: y0 = [Dose]

    Initial Parameter Estimates (for optimization):
        - ke: Estimate from terminal slope of log(C) vs t: ke ≈ lambda_z.
        - vd: Estimate from NCA: Vd = (Dose / AUC_inf) / ke. Or Vd ≈ Dose / C0 (extrapolated).

    Optimizer Bounds:
        - Lower Bounds: ke, vd typically set > small positive value (e.g., 1e-9).
        - Upper Bounds: Use plausible physiological limits.
           - cl: Should not exceed cardiac output / relevant organ blood flow (e.g., < 100-200 L/hr for human).
           - vd: Highly variable. Informed by Vz from NCA. Start large if unsure (e.g., 1000 L).
           - ke: Usually < 10 1/hr. Related to cl/vd bounds.

    Common Covariate Associations (Fixed Effects):
        - ke: As CL/Vd, influenced by factors affecting both (weight, organ function, age). 
            Often modeled via CL and Vd covariates.
        - vd: Weight (allometry), body composition, age.

    Parameterization of Random Effects (Mixed Effects):
        In population modeling, inter-individual variability (IIV) is estimated:
        - Concept: P_i = P_pop * exp(eta_i), eta_i ~ N(0, omega^2).
        - Estimated Term: omega^2 (variance) or omega (SD) of IIV.
        - Common Parameters with IIV: Ke (or CL) and Vd typically show IIV.
        - Initial Estimates for omega (SD): Start ~0.3 (30% CV).
        - Bounds for omega^2 (Variance): Lower > 0. Upper relatively large (e.g., 1-4).

              
    When to Prefer This Model:
        1. Rapid IV Administration: Applicable for IV bolus or very short infusions.
        2. Mono-exponential Decline: Concentration-time profile (semi-log plot)
           shows a single, linear decline phase, indicating negligible distribution
           effects or very rapid distribution compared to elimination.
        3. Simplicity/Data Sparsity: When data is insufficient to support a more
           complex multi-compartment model, or for initial characterization.
        4. Foundational Analysis: Often used as a building block or reference for
           understanding basic PK parameters (CL, Vd, half-life).

    Potential Fitting Difficulties:
        1. Model Mismatch: The primary difficulty arises if the *true* PK is
           multi-compartmental. Forcing a one-compartment fit to bi-exponential data
           will lead to poor model fit (systematic residuals) and biased parameter
           estimates (e.g., Vd estimate will be closer to Vz or Vdss, not V1; CL might
           also be biased depending on weighting).
        2. Estimating Vd from C0: Extrapolating back to C0 can be inaccurate if
           early samples are missing or if there's rapid initial distribution.
        3. Weighting Issues: Choice of data weighting can influence parameter
           estimates, especially if the concentration range is wide.

    Advanced Fitting Strategies:
        1. Strong NCA Guidance: NCA results (CL, Vd from terminal phase, lambda_z)
           are often very good initial estimates for this model if it's appropriate.
        2. Residual Analysis: Closely examine residual plots (predicted vs observed)
           for systematic trends. If trends indicate a distribution phase is missed,
           a multi-compartment model should be considered.
        3. Informing Multi-Compartment Models: The CL and Vd estimated here (often
           representing CL and Vz/Vdss) can provide starting points or bounds when
           developing a two-compartment model (where CL is the same, but Vd splits
           into V1 and V2).
    
    Potential Model Extensions:
        - Add peripheral compartments (Two-compartment bolus model).
        - Implement non-linear elimination (Michaelis-Menten).
        - Include effect compartment for PK/PD modeling.

    Parameter Sensitivity (Qualitative):
        - Increasing `ke`: Faster decline (shorter t1/2), lower AUC (assuming fixed Vd).
        - Increasing `vd`: Lower C0/initial concentrations, no change to t1/2 (fixed Ke), lower AUC (as CL = Ke*Vd increases).
    
    
    """
    def __init__(self, ):
        pass

    def ode(self, t, y, ke, vd):
        """ODE system for 1-cmt IV bolus (Ke, Vd)."""
        central_mass = y[0]
        dCMdt = -ke * central_mass
        return [dCMdt]

    def mass_to_depvar(self, pred_mass_central, ke, vd):
        """Converts central compartment mass to concentration."""
        depvar_unit_result = pred_mass_central / vd
        return depvar_unit_result
    
    @staticmethod
    def ode_for_diffrax(t, y, args_tuple):
        """
        JAX-compatible ODE function for Diffrax.
        y[0] is central_mass.
        args_tuple: (ke, vd) - vd not used in ODE if ke is the rate constant
        """
        ke, _vd = args_tuple 
        central_mass = y[0]
        dCMdt = -ke * central_mass
        return jnp.array([dCMdt])

    @staticmethod
    def convert_state_to_depvar(pred_y_state0, args_tuple):
        """
        JAX-compatible conversion from central mass to concentration.
        pred_y_state0: Predicted central_mass trajectory.
        args_tuple: (ke, vd)
        """
        _ke, vd = args_tuple
        # vd_safe = jnp.where(jnp.abs(vd) < 1e-9, 1e-9, vd)
        depvar_unit_result = pred_y_state0 / vd
        return depvar_unit_result

class TwoCompartmentBolus(PKBaseODE):
    """
    Two-compartment IV Bolus model (Central, Peripheral).
    Parameterized by Clearance (cl), Volume Central (v1),
    Inter-compartmental Clearance (q), Volume Peripheral (v2).

    Models drug distribution between central and peripheral compartments following
    an IV bolus, with first-order elimination from the central compartment.

    Key Model Assumptions:
        - Instantaneous administration and distribution into V1 (bolus input).
        - Two well-stirred compartments (Central, Peripheral).
        - First-order transfer between compartments (governed by Q, V1, V2).
        - First-order elimination from the central compartment (governed by CL, V1).
        - Constant parameters over time and concentration range.

    States (y):
        y[0]: Mass in Central Compartment (amount)
        y[1]: Mass in Peripheral Compartment (amount)
        Order: [Central, Peripheral]

    Parameters (*params for ode/mass_to_depvar):
        cl (float): Clearance from central compartment (volume/time).
        v1 (float): Central volume (volume).
        q (float): Inter-compartmental clearance (volume/time).
        v2 (float): Peripheral volume (volume).

    Units and Output:
        - Ensure consistency (e.g., hr, L, mg). Typical units: cl, q (L/hr); v1, v2 (L).
        - ODE solver returns mass in compartments [Central, Peripheral] over time.
        - `mass_to_depvar` converts Central mass (y[0]) to concentration using V1.
          Concentration = Mass / V1.

    Handling Dosing in Simulations:
        - Dosing handled externally by solver setup.
        - Dose `Dose` at t=0: `y0 = [Dose, 0]`.
        - Multiple doses: Set/add `Dose` to Central compartment mass (y[0]) at dose time.

    Common Derived Parameters:
        - Micro-constants: k10=cl/v1, k12=q/v1, k21=q/v2.
        - Hybrid rate constants (alpha, beta): Derived from micro-constants, define the
                                             bi-exponential decline. alpha > beta.
        - Terminal half-life: t1/2,beta = ln(2) / beta (beta ≈ lambda_z).
        - Volume of distribution at steady state: Vdss = v1 + v2.
        - Area Under Curve (AUC): AUC_inf = Dose / cl.

    Initial Conditions (y0 for ODE solver):
        For a single IV bolus dose `Dose` at t=0: y0 = [Dose, 0]

    Initial Parameter Estimates (for optimization):
        - cl: Estimate from NCA: CL = Dose / AUC_inf.
        - v1: Often estimated initially from Vd ≈ Dose / C0 (from C(t) extrapolation),
              but V1 is typically smaller than Vd from NCA (Vdss or Vz). Start with
              a value smaller than Vdss (e.g., 1/3 to 1/2 Vdss), or based on physiology
              (e.g., plasma volume). Check literature. Challenging. Smaller than Vdss/Vz. 
              Use Dose / C0 cautiously.
        - q: Distribution speed. Start similar to CL or based on visual inspection/literature.
        - v2: Related to Vdss (Vd at steady state = V1 + V2). (1) Use NCA Vdss = Dose*AUMC/AUC^2, 
            (2) plug Vdss into: V2 ≈ Vdss - V1_guess.

        Optimizer Bounds:
         - Lower Bounds: cl, v1, q, v2 typically set > small positive value (e.g., 1e-9).
        - Upper Bounds: Use plausible limits.
           - cl: < Cardiac output / organ blood flow (e.g., < 100-200 L/hr).
           - v1: Typically > plasma volume but < Vz. Maybe < 500 L.
           - q: Highly variable, can exceed CL. Maybe < 500 L/hr.
           - v2: Can be large. Maybe < 2000 L. Informed by Vdss/Vz.

    Common Covariate Associations (Fixed Effects):
        - cl: Weight (allometry), renal/hepatic function, age, genetics.
        - v1, v2: Weight (allometry), body composition, age.
        - q: Less common, potentially cardiac output or weight.

    Parameterization of Random Effects (Mixed Effects):
        - Concept: P_i = P_pop * exp(eta_i), eta_i ~ N(0, omega^2).
        - Estimated Term: omega^2 (variance) or omega (SD) of IIV.
        - Common Parameters with IIV: CL and V1 often show IIV. Q and V2 variability may also exist.
        - Initial Estimates for omega (SD): Start ~0.3 (30% CV) for CL, V1. Maybe lower for Q, V2.
        - Bounds for omega^2 (Variance): Lower > 0. Upper relatively large (e.g., 1-4).

    
    When to Prefer This Model:
        1. Bi-exponential Decline: IV bolus data clearly shows a bi-exponential
           decline on a semi-log plot (distinct distribution phase followed by a
           terminal elimination phase).
        2. Physiological Rationale: Drug properties (lipophilicity, tissue binding)
           or known physiology suggest significant, non-instantaneous distribution
           into a peripheral compartment is likely.
        3. Poor 1-Cpt Fit: A one-compartment bolus model fails to capture the initial,
           steeper decline phase, leading to systematic prediction errors.
        4. Statistical Support: Model selection criteria (AIC, BIC, LRT) significantly
           favor this model over the one-compartment alternative.
        5. Not Absorption Route: Clearly preferred over absorption models when the
           route of administration is IV bolus (or a very short infusion).

    Potential Fitting Difficulties:
        1. Identifiability: Requires rich data, especially early after the bolus,
           to clearly define the distribution phase and separate V1, Q, and V2.
           Sparse early data makes these parameters hard to estimate precisely.
        2. Parameter Correlation: High correlation between V1, V2, and Q is very
           common, affecting optimization stability and estimate precision.
        3. Initial Estimates: Getting good starting values for V1, V2, Q can be
           less intuitive than for CL; poor guesses can lead to convergence failure
           or local minima.
        4. Local Minima: The complex interplay between parameters increases the risk
           of the optimizer settling in a local minimum.
        5. Over-parameterization: If the data lacks a clear distribution phase (true
           PK closer to 1-Cpt), fitting this model might yield unstable estimates
           (e.g., very large Q, very small V2, parameters hitting bounds).

    Advanced Fitting Strategies:
        1. Robust NCA Guidance: Use NCA extensively for initial guesses: CL from
           Dose/AUC; use Vdss (= V1+V2) from Dose*AUMC/AUC^2 and Vz (= CL/lambda_z)
           to inform the scale of volumes; use lambda_z itself. Note that V1 is typically
           less than Vz or Vdss.
        2. Graphical "Peeling": Method of residuals (feathering/peeling) on semi-log
           plots can provide rough initial estimates for the two exponential rates
           and intercepts, which can be converted to micro-constants (k10, k12, k21)
           and then to CL, V1, Q, V2 to start the optimization.
        3. Fixing Parameters: If CL is known very reliably from NCA or other studies,
           fixing it might help stabilize the estimation of V1, Q, V2. Similarly, if
           V1 (initial dilution volume) is expected to be close to plasma volume,
           it could potentially be fixed or constrained.
        4. Sequential Fitting Approaches: While less common for bolus, one could
           theoretically analyze the terminal phase first to inform CL and Vz, then
           use this information when fitting the full curve.
        5. Population Modeling: Greatly enhances the ability to estimate all parameters,
           especially the correlated distribution parameters, by leveraging data from
           multiple individuals.
    
    Potential Model Extensions:
        - Add more compartments (Three-compartment model).
        - Implement non-linear elimination (Michaelis-Menten) or distribution.
        - Model Target-Mediated Drug Disposition (TMDD) if applicable.
        - Include effect compartment for PK/PD modeling.

    Parameter Sensitivity (Qualitative):
        - Increasing `cl`: Faster terminal decline (beta), lower AUC. Little effect on initial distribution phase (alpha).
        - Increasing `v1`: Lower initial concentration (C0), may slightly slow initial decline, complex effect on terminal phase.
        - Increasing `q`: Faster distribution phase (larger alpha), faster transfer between compartments.
            Terminal phase (beta) may become faster or slower depending on relative rates. Lower initial peak.
        - Increasing `v2`: Slower distribution (smaller alpha), slower terminal phase (smaller beta, longer t1/2,beta), larger Vdss.
            Less impact on initial concentration.
    
    
    """
    def __init__(self, ):
        pass

    def ode(self, t, y, cl, v1, q, v2):
        """ODE system for 2-cmt IV bolus."""
        central_mass, peripheral_mass = y
        # Micro-rate constants (internal): k10=cl/v1, k12=q/v1, k21=q/v2
        dCMdt = (q / v2) * peripheral_mass - (q / v1) * central_mass - (cl / v1) * central_mass
        dPMdt = (q / v1) * central_mass - (q / v2) * peripheral_mass
        return [dCMdt, dPMdt]

    def mass_to_depvar(self, pred_mass_central, cl, v1, q, v2):
        """Converts central compartment mass to concentration."""
        # Concentration = Central Mass / Central Volume
        depvar_unit_result = pred_mass_central / v1
        return depvar_unit_result
    
    @staticmethod
    def ode_for_diffrax(t, y, args_tuple):
        """
        JAX-compatible ODE function for Diffrax.
        y = [central_mass, peripheral_mass]
        args_tuple: (cl, v1, q, v2)
        """
        cl, v1, q, v2 = args_tuple
        central_mass, peripheral_mass = y

        # It's good practice to ensure v1 and v2 are not zero before division.
        # This can be done by adding a small epsilon or using jnp.where,
        # or by ensuring valid parameters are passed.
        # For simplicity, direct division is used here, assuming valid v1, v2 > 0.
        # v1_safe = jnp.where(jnp.abs(v1) < 1e-9, 1e-9, v1)
        # v2_safe = jnp.where(jnp.abs(v2) < 1e-9, 1e-9, v2)

        k10 = cl / v1
        k12 = q / v1
        k21 = q / v2

        dCMdt = k21 * peripheral_mass - k12 * central_mass - k10 * central_mass
        dPMdt = k12 * central_mass - k21 * peripheral_mass
        return jnp.array([dCMdt, dPMdt])

    @staticmethod
    def convert_state_to_depvar(pred_y_state0, args_tuple):
        """
        JAX-compatible conversion from central mass to concentration.
        pred_y_state0: Predicted central_mass trajectory.
        args_tuple: (cl, v1, q, v2)
        """
        _cl, v1, _q, _v2 = args_tuple
        # v1_safe = jnp.where(jnp.abs(v1) < 1e-9, 1e-9, v1)
        depvar_unit_result = pred_y_state0 / v1
        return depvar_unit_result

class TwoCompartmentAbsorption(PKBaseODE):
    """
    Two-compartment model with first-order absorption (Gut -> Central -> Peripheral).
    Parameterized by Absorption Rate Const (ka), Clearance (cl), Volume Central (v1),
    Inter-compartmental Clearance (q), Volume Peripheral (v2).
    
    Models extravascular administration with absorption, distribution between central
    and peripheral compartments, and elimination from central.

    Key Model Assumptions:
        - First-order absorption process.
        - Two well-stirred compartments (Central, Peripheral) plus absorption site.
        - First-order transfer between compartments (Q/F, V1/F, V2/F).
        - First-order elimination from central compartment (CL/F, V1/F).
        - Instantaneous distribution within each compartment.
        - Constant parameters over time and concentration range.

    States (y):
        y[0]: Mass in Central Compartment (amount)
        y[1]: Mass in Peripheral Compartment (amount)
        y[2]: Mass in Gut/Absorption Compartment (amount)
        Order: [Central, Peripheral, Gut]

    Parameters (*params for ode/mass_to_depvar):
        ka (float): First-order absorption rate constant (1/time).
        cl_f (float): Apparent clearance (CL/F) from central compartment (volume/time).
        v1_f (float): Apparent central volume (V1/F) (volume).
        q_f (float): Apparent inter-compartmental clearance (Q/F) (volume/time).
        v2_f (float): Apparent peripheral volume (V2/F) (volume).
        Note: Apparent parameters due to unknown bioavailability F.

    Units and Output:
        - Ensure consistency (e.g., hr, L, mg). Typical units: ka(1/hr); cl_f, q_f (L/hr); v1_f, v2_f (L).
        - ODE solver returns mass in compartments [Central, Peripheral, Gut] over time.
        - `mass_to_depvar` converts Central mass (y[0]) to concentration using V1/F.
          Concentration = Mass / (V1/F).

    Handling Dosing in Simulations:
        - Dosing handled externally by solver setup.
        - Dose `Dose` at t=0: `y0 = [0, 0, Dose]`.
        - Multiple doses: Add `Dose` to Gut compartment mass (y[2]) at dose time.

    Common Derived Parameters:
        - Micro-constants (apparent): k10=cl_f/v1_f, k12=q_f/v1_f, k21=q_f/v2_f.
        - Hybrid rate constants (alpha, beta) and terminal half-life (t1/2,beta = ln(2)/beta).
        - Apparent volume at steady state: Vdss/F = v1_f + v2_f.
        - Area Under Curve (AUC): AUC_inf = Dose / cl_f = (F * Dose) / CL.

    Initial Conditions (y0 for ODE solver):
        Typically, for a single dose `Dose` at t=0: y0 = [0, 0, Dose]

    Initial Parameter Estimates (for optimization):
        - ka: Related to Tmax. Guess ~1/Tmax.
        - cl_f: Estimate from NCA: CL/F = Dose / AUC_inf.
        - v1_f: Often smaller than total Vd. Start with a plausible physiological volume 
            (e.g., plasma volume ~3-5L for humans) or a fraction of the total Vd estimate. 
            Literature for similar drugs is helpful. Note that: Vdss/F = V1/F + V2/F.
        - q_f: Distribution speed. Start similar to cl_f or based on literature/shape.
        - v2_f: Related to the extent of distribution. Estimate total Vdss/F or Vz/F
              from NCA (e.g., Vz/F = (CL/F, Vdss = Cl/F * MRT) / lambda_z). Initial guess:
              V2/F ≈ (Vz/F OR Vdss/F) - V1/F (ensure V1/F < Vz/F).

        Optimizer Bounds:
         - Lower Bounds: ka, cl_f, v1_f, q_f, v2_f typically set > small positive value (e.g., 1e-9).
         - - Upper Bounds: Use plausible limits.
           - ka: Typically < 50 1/hr.
           - cl(/F): < Cardiac output / organ blood flow (e.g., < 100-200 L/hr).
           - v1(/F): Typically larger than plasma volume but smaller than Vz/F. Maybe < 500 L.
           - q(/F): Highly variable, can exceed CL. Maybe < 500 L/hr.
           - v2(/F): Can be large, especially for lipophilic drugs. Maybe < 2000 L.
           Bounds often informed by NCA Vz/F and Vdss/F.

    Common Covariate Associations (Fixed Effects):
        - cl_f: CL often related to weight (allometry), renal/hepatic function, age, genetics. F affects apparent value.
        - v1_f, v2_f: V1, V2 often related to weight (allometry), body composition, age. F affects apparent value.
        - q_f: Q less commonly linked, potentially cardiac output or weight. F affects apparent value.
        - ka: Formulation, food, GI factors, age.

    Parameterization of Random Effects (Mixed Effects):
        In population modeling, inter-individual variability (IIV) is estimated:
        - Concept: P_i = P_pop * exp(eta_i), eta_i ~ N(0, omega^2).
        - Estimated Term: omega^2 (variance) or omega (SD) of IIV.
        - Common Parameters with IIV: CL/F, V1/F, Ka often show IIV. V2/F and Q/F variability also possible.
        - Initial Estimates for omega (SD): Start ~0.3 (30% CV) for CL/F, V1/F, Ka. Maybe lower for Q/F, V2/F.
        - Bounds for omega^2 (Variance): Lower > 0. Upper relatively large (e.g., 1-4).


    When to Prefer This Model:
        1. Bi-exponential Decline Post-Absorption: The concentration-time data
           (semi-log plot) clearly shows a bi-exponential decline after Cmax (steeper
           initial decline followed by shallower terminal phase), suggesting significant
           distribution kinetics not captured by a 1-Cpt model.
        2. Physiological Plausibility: The drug is known or expected to distribute
           significantly and non-instantaneously into tissues (e.g., lipophilic drugs,
           high tissue binding).
        3. Poor 1-Cpt Fit: A one-compartment absorption model provides a demonstrably poor
           fit, showing systematic deviations in residuals or visual predictive checks,
           especially during the post-peak phase.
        4. Statistical Improvement: Model selection criteria (AIC, BIC, LRT) favor
           this model over a one-compartment alternative, indicating the improved fit
           justifies the added complexity.
        5. IV Data Comparison: If corresponding IV data clearly requires a two-compartment
           model, it's highly likely the oral data will also benefit from modeling
           the same distribution process.

    Potential Fitting Difficulties:
        1. Identifiability: With potentially five parameters (plus F) estimated, needs
           rich data, especially around Cmax and during the distribution phase. Sparse
           data can make unique estimation hard.
        2. Parameter Correlation: High correlation is common between distribution parameters
           (v1, v2, q) and potentially with ka or cl/F, impacting estimate precision and
           optimization stability.
        3. Local Minima: Increased risk during optimization; good initial estimates and
           possibly multiple runs are important.
        4. Estimating Ka: Subject to issues if absorption is very fast (approaching bolus)
           or very slow (flip-flop kinetics).
        5. Distinguishing Distribution: Separating V1/F, V2/F, and Q/F requires good data
           post-Cmax; late sampling hinders this.
        6. Over-parameterization Risk: Applying this model to data adequately described
           by 1-Cpt can lead to unstable/unreliable estimates.

    Advanced Fitting Strategies:
        1. Strong NCA Guidance: Use NCA results (CL/F, Vz/F, Vdss/F, lambda_z, Tmax) to
           inform initial guesses for all parameters, recognizing the relationships
           (e.g., Vz/F > V1/F).
        2. Sequential Fitting (with IV data): Fit IV data first using a two-compartment
           bolus model to get CL, V1, Q, V2. Use these as fixed values or strong priors
           when fitting oral data to estimate ka and F (or ka if F is assumed/fixed).
        3. Fixing Parameters: If certain parameters are known reliably (e.g., F from a
           separate study, or distribution is very fast suggesting fixing Q to a large value),
           fixing them can stabilize the estimation of others.
        4. Population Modeling: Analyzing data from multiple individuals/dose levels together
           can significantly improve the identifiability and precision of all parameters,
           especially the distribution-related ones. Allows estimation of inter-individual
           variability and potential estimation of F.
    
    Potential Model Extensions:
        - Add more compartments (Three-compartment absorption model).
        - Implement non-linear elimination (MM) or distribution/absorption.
        - Add transit compartments for absorption delay.
        - Include effect compartment for PK/PD modeling.

    Parameter Sensitivity (Qualitative):
        - Increasing `ka`: Faster absorption, earlier Tmax, potentially higher Cmax.
        - Increasing `cl_f`: Lower AUC, faster terminal decline, potentially lower Cmax.
        - Increasing `v1_f`: Lower initial concentrations post-absorption, complex effects on profile shape.
        - Increasing `q_f`: Faster distribution phase, potentially lower peak concentration (Cmax).
        - Increasing `v2_f`: Slower distribution, slower terminal decline (longer t1/2,beta), larger Vdss/F.
    
    
    """
    def __init__(self, ):
        pass

    def ode(self, t, y, ka, cl, v1, q, v2):
        """ODE system for 2-cmt absorption."""
        central_mass, peripheral_mass, gut_mass = y

        # Ensure parameters that form rates are valid (avoid division by zero)
        # Volumes (v1, v2) should be positive. Clearances (cl, q) non-negative. ka non-negative.
        if v1 <= 0 or v2 <= 0:
             # Handle error appropriately - e.g., raise ValueError or return NaNs
             # Returning zeros might mask the issue during optimization
             # For simplicity here, we'll assume valid inputs but raise conceptually:
             raise ValueError("Volumes v1 and v2 must be positive.")
             # Alternatively, return array of NaNs matching state size:
             # return [np.nan, np.nan, np.nan]

        # Micro-rate constants (internal): k10=cl/v1, k12=q/v1, k21=q/v2
        k10 = cl / v1
        k12 = q / v1
        k21 = q / v2

        dCMdt = ka * gut_mass + k21 * peripheral_mass - k12 * central_mass - k10 * central_mass
        dPMdt = k12 * central_mass - k21 * peripheral_mass
        dGdt = -ka * gut_mass

        return [dCMdt, dPMdt, dGdt]

    def mass_to_depvar(self, pred_mass_central, ka, cl, v1, q, v2):
        """Converts central compartment mass to concentration."""
        # Ensure v1 is valid for division
        if v1 <= 0:
             # Handle error appropriately (raise or return NaN)
             raise ValueError("Central volume v1 must be positive for concentration calculation.")
             # return np.nan # Or np.full_like(pred_mass_central, np.nan)

        # Concentration = Central Mass / Central Volume
        depvar_unit_result = pred_mass_central / v1
        return depvar_unit_result
    
    @staticmethod
    def ode_for_diffrax(t, y, args_tuple):
        """
        JAX-compatible ODE function for Diffrax.
        y = [central_mass, peripheral_mass, gut_mass]
        args_tuple: (ka, cl, v1, q, v2)
        Note: Original ValueError checks for v1, v2 <= 0 are omitted
              as they are not JIT-friendly. Assumes valid positive volumes.
        """
        ka, cl, v1, q, v2 = args_tuple
        central_mass, peripheral_mass, gut_mass = y

        # v1_safe = jnp.where(jnp.abs(v1) < 1e-9, 1e-9, v1)
        # v2_safe = jnp.where(jnp.abs(v2) < 1e-9, 1e-9, v2)
        
        k10 = cl / v1
        k12 = q / v1
        k21 = q / v2

        dCMdt = ka * gut_mass + k21 * peripheral_mass - k12 * central_mass - k10 * central_mass
        dPMdt = k12 * central_mass - k21 * peripheral_mass
        dGdt = -ka * gut_mass
        return jnp.array([dCMdt, dPMdt, dGdt])

    @staticmethod
    def convert_state_to_depvar(pred_y_state0, args_tuple):
        """
        JAX-compatible conversion from central mass to concentration.
        pred_y_state0: Predicted central_mass trajectory.
        args_tuple: (ka, cl, v1, q, v2)
        Note: Original ValueError check for v1 <= 0 omitted.
        """
        _ka, _cl, v1, _q, _v2 = args_tuple
        # v1_safe = jnp.where(jnp.abs(v1) < 1e-9, 1e-9, v1)
        depvar_unit_result = pred_y_state0 / v1
        return depvar_unit_result

class OneCompartmentInfusion(PKBaseODE):
    """
    One-compartment model with zero-order infusion.
    Parameterized by Infusion Rate (R0), Clearance (cl), Volume (vd).

    Models drug administration via constant rate infusion into a single
    compartment with first-order elimination.

    Key Model Assumptions:
        - Constant, zero-order infusion rate (R0) during infusion period.
        - Single, well-stirred central compartment.
        - First-order elimination process (rate proportional to amount).
        - Instantaneous distribution within the compartment.
        - Constant parameters over time and concentration range.

    States (y):
        y[0]: Mass in Central Compartment (amount)
        Order: [Central]

    Parameters (*params for ode/mass_to_depvar):
        R0 (float): Zero-order infusion rate (amount/time). (Often fixed).
        cl (float): Clearance from the central compartment (volume/time).
        vd (float): Volume of distribution of the central compartment (volume).

    Units and Output:
        - Ensure consistency (e.g., hr, L, mg). Typical units: R0(mg/hr), cl(L/hr), vd(L).
        - ODE solver returns mass in Central compartment (y[0]) over time.
        - `mass_to_depvar` converts Central mass to concentration using Vd.
          Concentration = Mass / Vd.

    Handling Dosing in Simulations:
        - Dosing handled externally by solver setup.
        - Requires logic to set the `R0` parameter to the infusion rate during the
          infusion period and set `R0 = 0` after the infusion stops (t > Tinf).

    Common Derived Parameters:
        - Elimination rate constant: ke = cl / vd.
        - Half-life: t1/2 = ln(2) / ke = ln(2) * vd / cl.
        - Steady-state concentration (if Tinf >> t1/2): Css = R0 / cl.
        - Time to reach fraction (e.g., 90%) of Css: Approx 3.3 * t1/2.

    Initial Conditions (y0 for ODE solver):
        Typically, if starting at t=0 with the infusion: y0 = [0]

    Initial Parameter Estimates (for optimization):
        - R0: Usually known/fixed.
        - cl: Estimate from Css if reached: CL = R0 / Css. Else use NCA post-infusion.
        - vd: Estimate from post-infusion lambda_z: Vd = CL / lambda_z.

        Optimizer Bounds (if estimating CL, Vd):
         - Lower Bounds: cl, vd typically set > small positive value (e.g., 1e-9).
         - Upper Bounds: Plausible physiological limits (CL < blood flow, Vd < huge).

    Common Covariate Associations (Fixed Effects):
        - cl: Weight (allometry), renal/hepatic function, age, genetics.
        - vd: Weight (allometry), body composition, age.

    Parameterization of Random Effects (Mixed Effects):
        - Concept: P_i = P_pop * exp(eta_i), eta_i ~ N(0, omega^2).
        - Estimated Term: omega^2 (variance) or omega (SD) of IIV.
        - Common Parameters with IIV: CL and Vd typically show IIV.
        - Initial Estimates for omega (SD): Start ~0.3 (30% CV).
        - Bounds for omega^2 (Variance): Lower > 0. Upper relatively large (e.g., 1-4).
                                       
    When to Prefer This Model:
        1. Infusion Administration: Directly applicable for constant-rate IV infusion.
        2. Mono-exponential Post-Infusion Decline: Concentration decline after
           stopping the infusion is mono-exponential (linear on semi-log plot).
        3. Data Limitations: When data is insufficient (e.g., sparse sampling during
           or after infusion) to support a multi-compartment structure.
        4. Simplicity: As a simpler alternative to multi-compartment infusion models.

    Potential Fitting Difficulties:
        1. Model Mismatch: If true PK is multi-compartmental, fitting this model
           will result in poor fit during the initial phase or post-infusion phase,
           and biased parameters.
        2. Handling R0: Requires careful implementation in the solver loop to
           correctly switch R0 on and off at the infusion start/stop times (Tinf).
        3. Reaching Steady State: If the infusion duration (Tinf) is short relative
           to the drug's half-life (< 3-5 half-lives), true steady state (Css) is
           not reached, making CL estimation from Css impossible and requiring
           reliance on post-infusion data or more complex fitting.
        4. Sparse Post-Infusion Data: If sampling after the infusion stops is
           limited, estimating the terminal slope (lambda_z) and thus CL and Vd
           can be inaccurate.

    Advanced Fitting Strategies:
        1. Leverage Steady State: If Css is reliably measured, use CL = R0 / Css
           as a strong estimate or constraint for CL.
        2. Combine with Bolus Data: If IV bolus data is available, estimate CL and
           Vd from that data first using the bolus model, then use those estimates
           as strong priors or fixed values when fitting the infusion data (helps
           confirm model consistency).
        3. Sequential Infusion/Post-Infusion Fit: One could potentially fit the
           post-infusion decline first (treating it like a bolus starting from C_Tinf)
           to estimate ke (=lambda_z) and Vd, then fit the infusion phase using
           these estimates to potentially refine CL.
        4. Informing Multi-Compartment Infusion: Estimates of CL and Vd (as Vz/Vdss)
           can inform the development of a two-compartment infusion model.
           
    Potential Model Extensions:
        - Add peripheral compartments (Two-compartment infusion model).
        - Implement non-linear elimination (Michaelis-Menten).
        - Include effect compartment for PK/PD modeling.

    Parameter Sensitivity (Qualitative):
        - Increasing `R0`: Higher concentrations during infusion, higher Css.
        - Increasing `cl`: Lower concentrations during infusion, lower Css, faster decline post-infusion (shorter t1/2).
        - Increasing `vd`: Slower approach to Css, lower concentrations for a given mass, slower
            decline post-infusion (longer t1/2), no change to Css.
    
    
    """
    def __init__(self, ):
        pass

    def ode(self, t, y, R0, cl, vd):
        """ODE system for 1-cmt infusion."""
        central_mass = y[0]
        # Elimination rate constant ke = cl / vd
        # R0 is the infusion rate (amount/time)
        dCMdt = R0 - (cl / vd) * central_mass
        return [dCMdt]

    def mass_to_depvar(self, pred_mass_central, R0, cl, vd):
        """Converts central compartment mass to concentration."""
        # Concentration = Mass / Volume
        depvar_unit_result = pred_mass_central / vd
        return depvar_unit_result
    
    @staticmethod
    def ode_for_diffrax(t, y, args_tuple):
        """
        JAX-compatible ODE function for Diffrax.
        y[0] is central_mass.
        args_tuple: (R0, cl, vd)
        """
        R0, cl, vd = args_tuple
        central_mass = y[0]
        # vd_safe = jnp.where(jnp.abs(vd) < 1e-9, 1e-9, vd)
        dCMdt = R0 - (cl / vd) * central_mass
        return jnp.array([dCMdt])

    @staticmethod
    def convert_state_to_depvar(pred_y_state0, args_tuple):
        """
        JAX-compatible conversion from central mass to concentration.
        pred_y_state0: Predicted central_mass trajectory.
        args_tuple: (R0, cl, vd)
        """
        _R0, _cl, vd = args_tuple
        # vd_safe = jnp.where(jnp.abs(vd) < 1e-9, 1e-9, vd)
        depvar_unit_result = pred_y_state0 / vd
        return depvar_unit_result

class OneCompartmentBolusMM(PKBaseODE):
    """
    One-compartment IV Bolus model with Michaelis-Menten (saturable) elimination.
    Parameterized by Max Elimination Rate (vmax), Michaelis Constant (km),
    and Volume of Distribution (vd).

    Models non-linear, saturable elimination following IV bolus administration.

    Key Model Assumptions:
        - Instantaneous administration and distribution (bolus input).
        - Single, well-stirred central compartment.
        - Saturable elimination process following Michaelis-Menten kinetics.
        - Constant Vmax, Km, Vd parameters over time.

    States (y):
        y[0]: Mass in Central Compartment (amount)
        Order: [Central]

    Parameters (*params for ode/mass_to_depvar):
        vmax (float): Maximum rate of elimination (amount/time).
        km (float): Michaelis constant (concentration units); concentration at half Vmax.
        vd (float): Volume of distribution (volume).

    Units and Output:
        - Ensure consistency (e.g., hr, L, mg). Typical units: vmax(mg/hr), km(mg/L), vd(L).
        - ODE solver returns mass in the Central compartment (y[0]) over time.
        - `mass_to_depvar` converts Central mass to concentration using Vd.
          Concentration (C) = Mass / Vd. The elimination rate is Vmax*C / (Km + C).

    Handling Dosing in Simulations:
        - Dosing handled externally by solver setup.
        - Dose `Dose` at t=0: `y0 = [Dose]`.
        - Multiple doses: Set/add `Dose` to Central compartment mass (y[0]) at dose time.

    Common Derived Parameters:
        - Concentration-dependent Clearance: CL(C) = Vmax / (Km + C).
        - Low-Concentration Clearance (C << Km): CL_linear ≈ Vmax / Km.
        - Half-life is concentration-dependent and not constant.

    Initial Conditions (y0 for ODE solver):
        For a single IV bolus dose `Dose` at t=0: y0 = [Dose]

    Initial Parameter Estimates (for optimization):
        - vmax, km, vd: Challenging. Use literature/in vitro data if possible. Low-dose
                       data might give Vd and Vmax/Km ratio. Requires data spanning Km.

        Optimizer Bounds:
         - Lower Bounds: vmax, km, vd typically set > small positive value (e.g., 1e-9).
                  - Upper Bounds: Plausible limits.
           - vmax: Can be large, related to max enzyme/transporter capacity. Informed by
                   CL_linear * Km_guess. Start relatively large if needed.
           - km: Highly context dependent (concentration units). Can range from nM to mM.
                 Bounds should encompass expected therapeutic concentrations.
           - vd: Similar bounds as for linear Vd (e.g., < 1000 L).

    Common Covariate Associations (Fixed Effects):
        - vmax: Weight (allometry), organ function, genetics (enzymes/transporters), interactions.
        - km: Less commonly linked to simple covariates, potentially genetics (affinity changes).
        - vd: Weight (allometry), body composition, age.

    Parameterization of Random Effects (Mixed Effects):
        In population modeling, inter-individual variability (IIV) is estimated:
        - Concept: P_i = P_pop * exp(eta_i), eta_i ~ N(0, omega^2).
        - Estimated Term: omega^2 (variance) or omega (SD) of IIV.
        - Common Parameters with IIV: Vmax, Vd often show IIV. Km variability also possible.
        - Initial Estimates for omega (SD): Start ~0.3 (30% CV) for Vmax, Vd.
        - Bounds for omega^2 (Variance): Lower > 0. Upper relatively large (e.g., 1-4).
    
    When to Prefer This Model:
        1. Non-Linear Kinetics Observed: Clearance appears dose-dependent (decreases
           with increasing dose/concentration). AUC increases more than proportionally
           with dose. Half-life increases with dose. Semi-log plot of C vs t is
           concave (not a straight line).
        2. Saturation Mechanism Known: There is physiological or in vitro evidence
           suggesting a saturable enzyme or transporter is primarily responsible
           for elimination.
        3. Poor Linear Model Fit: Standard first-order models fail to describe the
           data adequately, especially across a range of doses or high concentrations.

    Potential Fitting Difficulties:
        1. Data Requirements: Accurately estimating both Vmax and Km requires data
           that spans concentrations both well below and near/above Km. If data is only
           at low concentrations (C << Km), kinetics appear linear (Rate ≈ (Vmax/Km)*C),
           and only the ratio Vmax/Km (apparent first-order CL) and Vd can be estimated.
           If data is only at high concentrations (C >> Km), kinetics appear zero-order
           (Rate ≈ Vmax), and Km is poorly identifiable.
        2. Parameter Correlation: Vmax and Km can be highly correlated, making unique
           estimation difficult without informative data across the right concentration range.
           Vd estimates can also influence Vmax and Km.
        3. Distinguishing from Multi-Compartment: With limited data, especially if
           sampling misses the early, high-concentration phase, non-linear elimination
           can sometimes be difficult to distinguish from multi-compartment linear kinetics.
        4. Identifiability of Vd: Estimating Vd simultaneously with Vmax and Km can
           be challenging. Sometimes Vd is fixed based on low-dose linear data or
           literature values.

    Advanced Fitting Strategies:
        1. Utilize Low-Dose Data: If data from low doses (expected C << Km) is
           available, fit a linear one-compartment model (`OneCompartmentBolus_CL`)
           first to get estimates for Vd and the apparent low-concentration clearance
           (CL_low ≈ Vmax/Km).
        2. Fix Vd: Use the Vd estimated from low-dose linear data (or literature)
           and fix it when fitting the MM model to higher-dose data. This reduces
           the number of estimated parameters to Vmax and Km, often improving stability.
        3. Use Vmax/Km Ratio: Use the CL_low ≈ Vmax/Km relationship from linear fits
           to inform initial guesses or constrain the ratio during MM fitting.
        4. In Vitro / In Silico Data: Incorporate information from in vitro enzyme
           kinetics (Km, Vmax for specific enzymes scaled to in vivo levels) as
           initial estimates or priors.
        5. Population Approach: Analyzing data from multiple dose levels simultaneously
           in a population context can significantly improve the ability to estimate
           MM parameters compared to fitting individual datasets separately.
    
    Potential Model Extensions:
        - Add peripheral compartments (Two-compartment MM model).
        - Implement non-linear absorption or distribution.
        - Include effect compartment for PK/PD modeling.

    Parameter Sensitivity (Qualitative):
        - Increasing `vmax`: Faster elimination, especially at higher concentrations (C > Km). Lower AUC.
        - Increasing `km`: Decreases saturation, makes kinetics more linear over a wider range.
            Increases concentrations needed to reach Vmax/2. Increases CL(C) at C >> Km.
        - Increasing `vd`: Lower concentrations overall (C=Mass/Vd), which indirectly slows elimination rate (Vmax*C/(Km+C)).
            Prolongs elimination time.
    
    
    """
    def __init__(self, ):
        pass

    def ode(self, t, y, vmax, km, vd):
        """ODE system for 1-cmt IV bolus with MM elimination."""
        central_mass = y[0]
        # Avoid division by zero if vd is zero (should not happen)
        if abs(vd) < 1e-12:
             concentration = 0.0 # Or handle as error
        else:
             concentration = central_mass / vd

        # Michaelis-Menten elimination rate: Vmax * C / (Km + C)
        # Need elimination in terms of mass: Vmax * (Mass/Vd) / (Km + Mass/Vd)
        # Multiply num/den by Vd: Vmax * Mass / (Km*Vd + Mass)
        denominator = (km * vd + central_mass)
        # Avoid division by zero if mass is zero AND km*vd is zero (highly unlikely)
        if abs(denominator) < 1e-12: # Use tolerance for floating point
             elimination_rate = 0.0
        else:
             # Ensure vmax and mass have compatible signs if needed (usually positive)
             elimination_rate = (vmax * central_mass) / denominator

        dCMdt = -elimination_rate
        return [dCMdt]

    def mass_to_depvar(self, pred_mass_central, vmax, km, vd):
        """Converts central compartment mass to concentration."""
        # Concentration = Mass / Volume
        # Avoid division by zero if vd is zero
        if abs(vd) < 1e-12:
            return np.zeros_like(pred_mass_central) if isinstance(pred_mass_central, np.ndarray) else 0.0
        depvar_unit_result = pred_mass_central / vd
        return depvar_unit_result
    
    @staticmethod
    def ode_for_diffrax(t, y, args_tuple):
        """
        JAX-compatible ODE function for Diffrax (Michaelis-Menten).
        y[0] is central_mass.
        args_tuple: (vmax, km, vd)
        """
        vmax, km, vd = args_tuple
        central_mass = y[0]

        # Avoid division by zero using jnp.where for vd if it's used to calculate concentration
        # concentration = jnp.where(jnp.abs(vd) < 1e-12, 0.0, central_mass / vd) 
        # The elimination term is directly formulated with central_mass and vd in denominator:
        
        denominator = km * vd + central_mass # Denominator for the elimination rate
        
        # Avoid division by zero for the elimination rate calculation
        elimination_rate = jnp.where(
            jnp.abs(denominator) < 1e-12, 
            0.0, 
            vmax * central_mass / denominator
        )
        
        dCMdt = -elimination_rate
        return jnp.array([dCMdt])

    @staticmethod
    def convert_state_to_depvar(pred_y_state0, args_tuple):
        """
        JAX-compatible conversion from central mass to concentration.
        pred_y_state0: Predicted central_mass trajectory.
        args_tuple: (vmax, km, vd)
        """
        _vmax, _km, vd = args_tuple
        
        # Handle potential division by zero for vd
        depvar_unit_result = jnp.where(
            jnp.abs(vd) < 1e-12,
            jnp.zeros_like(pred_y_state0), # Handles scalar or array pred_y_state0
            pred_y_state0 / vd
        )
        return depvar_unit_result
    

