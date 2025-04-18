from numba import njit
import abc


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
    def __init__(self):
        """Initializes the base class."""
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

class OneCompartmentAbsorption(PKBaseODE):
    """
    One-compartment model with first-order absorption (Gut -> Central).
    Parameterized by Ka, CL, Vd.

    Represents oral or extravascular administration where drug enters the
    central compartment via a first-order absorption process from a
    hypothetical 'gut' or depot compartment. Elimination is first-order
    from the central compartment.

    States (y):
        y[0]: Mass in Central Compartment (amount)
        y[1]: Mass in Gut/Absorption Compartment (amount)

    Parameters (*params for ode/mass_to_depvar):
        ka (float): First-order absorption rate constant (1/time).
        cl (float): Clearance from the central compartment (volume/time).
        vd (float): Volume of distribution of the central compartment (volume).

    Initial Conditions (y0 for ODE solver):
        Typically, for a single dose `Dose` at t=0:
        y0 = [0, Dose]
        If starting simulation after absorption has begun, estimate residual
        masses in both compartments.

    Initial Parameter Estimates (for optimization):
        - ka: Related to Tmax (time of peak concentration). Rough guess: 1/Tmax.
              Can be sensitive, try values like 0.5, 1.0, 2.0 (1/hr).
        - cl: Estimate from NCA: CL/F = Dose / AUC_inf (F=bioavailability).
              Guess F (e.g., 1) if unknown.
        - vd: Estimate from NCA: Vd/F = CL/F / lambda_z. Use terminal slope
              lambda_z. Or Vd/F ~ Dose / (AUC*lambda_z). Often larger than
              physiological volumes. Check literature for similar drugs.
    
    Optimizer Bounds:
         - Lower Bounds: All parameters (ka, cl, vd, ke) must be > 0. Use a small positive number (e.g., 1e-9).
         - Upper Bounds: Less critical, but use plausible limits.
           - ka: Can be large (e.g., 10 or 100 1/hr), but usually < 50.
           - cl(/F): Should generally not exceed physiological limits like cardiac output or relevant organ blood flow (e.g., < 100-200 L/hr for human hepatic/renal flow).
           - vd(/F): Highly variable, but unlikely to be thousands of L/kg. Can be informed by Vz/F from NCA. Start relatively large if unsure (e.g., 1000 L).
           - ke: Usually < 10 1/hr. Related to cl/vd bounds.

    Common Covariate Associations (Fixed Effects):
        These parameters may be influenced by subject attributes:
        - cl(/F): Often associated with body weight (allometric scaling), renal function
                  (e.g., Creatinine Clearance - CrCL) if renally excreted, hepatic
                  function markers if hepatically metabolized, age, genetic polymorphisms.
        - vd(/F): Often associated with body weight (allometric scaling), sometimes
                  body composition (fat mass for lipophilic drugs), age.
        - ka: Can be influenced by formulation differences, food effects, GI transit
              time/pH modifiers, age (esp. pediatrics/geriatrics).
        - F (Bioavailability, implicitly affecting cl/F, vd/F): Influenced by formulation,
              food, gut wall metabolism/transporters (genetics, drug interactions).

    Parameterization of Random Effects (Mixed Effects):
        In population modeling, inter-individual variability (IIV) is estimated:
        - Concept: Assumes each subject's parameter value (P_i) varies around the
                   typical population value (P_pop), often modeled using a log-normal
                   distribution: P_i = P_pop * exp(eta_i), where eta_i ~ N(0, omega^2).
        - Estimated Term: The variance (omega^2) or standard deviation (omega) of the
                         etas for each parameter exhibiting IIV.
        - Common Parameters with IIV: Typically CL/F, Vd/F, and Ka often show
                                     significant IIV.
        - Initial Estimates for omega (SD): A common starting guess is ~0.3 (approx.
                                          30% CV). Typical range might be 0.1 to 0.6.
        - Bounds for omega^2 (Variance): Lower bound > 0 (e.g., 1e-6). Upper bound
                                       can be relatively large (e.g., 1 or 4).

              
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
    Parameterized by Ka, Ke, Vd. (Ke = CL/Vd)

    Alternative parameterization of the 1-compartment absorption model.

    States (y):
        y[0]: Mass in Central Compartment (amount)
        y[1]: Mass in Gut/Absorption Compartment (amount)

    Parameters (*params for ode/mass_to_depvar):
        ka (float): First-order absorption rate constant (1/time).
        ke (float): First-order elimination rate constant (1/time).
        vd (float): Volume of distribution of the central compartment (volume).

    Initial Conditions (y0 for ODE solver):
        Typically, for a single dose `Dose` at t=0:
        y0 = [0, Dose]

    Initial Parameter Estimates (for optimization):
        - ka: Related to Tmax. Rough guess: 1/Tmax. Try 0.5, 1.0, 2.0 (1/hr).
        - ke: Estimate from terminal slope (lambda_z) of log(concentration) vs time.
              ke ≈ lambda_z for this model after absorption phase.
        - vd: Estimate from Dose / (ke * AUC). Or use CL estimate (CL = ke*vd)
              from NCA (CL/F = Dose/AUC) combined with ke estimate. Check literature.
    
    Optimizer Bounds:
         - Lower Bounds: All parameters (ka, cl, vd, ke) must be > 0. Use a small positive number (e.g., 1e-9).
         - Upper Bounds: Less critical, but use plausible limits.
           - ka: Can be large (e.g., 10 or 100 1/hr), but usually < 50.
           - cl(/F): Should generally not exceed physiological limits like cardiac output or relevant organ blood flow (e.g., < 100-200 L/hr for human hepatic/renal flow).
           - vd(/F): Highly variable, but unlikely to be thousands of L/kg. Can be informed by Vz/F from NCA. Start relatively large if unsure (e.g., 1000 L).
           - ke: Usually < 10 1/hr. Related to cl/vd bounds.

    Common Covariate Associations (Fixed Effects):
        These parameters may be influenced by subject attributes:
        - cl(/F): Often associated with body weight (allometric scaling), renal function
                  (e.g., Creatinine Clearance - CrCL) if renally excreted, hepatic
                  function markers if hepatically metabolized, age, genetic polymorphisms.
        - vd(/F): Often associated with body weight (allometric scaling), sometimes
                  body composition (fat mass for lipophilic drugs), age.
        - ka: Can be influenced by formulation differences, food effects, GI transit
              time/pH modifiers, age (esp. pediatrics/geriatrics).
        - F (Bioavailability, implicitly affecting cl/F, vd/F): Influenced by formulation,
              food, gut wall metabolism/transporters (genetics, drug interactions).

    Parameterization of Random Effects (Mixed Effects):
        In population modeling, inter-individual variability (IIV) is estimated:
        - Concept: Assumes each subject's parameter value (P_i) varies around the
                   typical population value (P_pop), often modeled using a log-normal
                   distribution: P_i = P_pop * exp(eta_i), where eta_i ~ N(0, omega^2).
        - Estimated Term: The variance (omega^2) or standard deviation (omega) of the
                         etas for each parameter exhibiting IIV.
        - Common Parameters with IIV: Typically CL/F, Vd/F, and Ka often show
                                     significant IIV.
        - Initial Estimates for omega (SD): A common starting guess is ~0.3 (approx.
                                          30% CV). Typical range might be 0.1 to 0.6.
        - Bounds for omega^2 (Variance): Lower bound > 0 (e.g., 1e-6). Upper bound
                                       can be relatively large (e.g., 1 or 4).
    
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

class OneCompartmentBolus_CL(PKBaseODE):
    """
    One-compartment IV Bolus model.
    Parameterized by Clearance (cl) and Volume of Distribution (vd).

    Simplest model for IV bolus injection. Assumes instantaneous distribution
    throughout a single compartment volume and first-order elimination.

    States (y):
        y[0]: Mass in Central Compartment (amount)

    Parameters (*params for ode/mass_to_depvar):
        cl (float): Clearance from the central compartment (volume/time).
        vd (float): Volume of distribution of the central compartment (volume).

    Initial Conditions (y0 for ODE solver):
        For a single IV bolus dose `Dose` at t=0:
        y0 = [Dose]
        If simulating a later dose in a series, y0 is the residual mass
        just before the new dose, and the dose is added: y0 = [ResidualMass + Dose].

    Initial Parameter Estimates (for optimization):
        - cl: Estimate from NCA: CL = Dose / AUC_inf.
        - vd: Estimate from NCA: Vd = CL / lambda_z, where lambda_z is the
              terminal slope. Alternatively, extrapolate C(t) to t=0 on a
              semi-log plot to get C0, then Vd ≈ Dose / C0. Check literature.
    
    Optimizer Bounds:
         - Lower Bounds: All parameters (cl, vd, ke) must be > 0 (e.g., 1e-9).
         - Upper Bounds: Use plausible physiological limits.
           - cl: Should not exceed cardiac output / relevant organ blood flow (e.g., < 100-200 L/hr for human).
           - vd: Highly variable. Informed by Vz from NCA. Start large if unsure (e.g., 1000 L).
           - ke: Usually < 10 1/hr. Related to cl/vd bounds.

    Common Covariate Associations (Fixed Effects):
        These parameters may be influenced by subject attributes:
        - cl: Associated with body weight (allometry), renal function (CrCL), hepatic
              function markers, age, genetic polymorphisms (relevant enzymes/transporters).
        - vd: Associated with body weight (allometry), body composition (lipophilicity), age.
        - ke: As a composite parameter (cl/vd), its direct covariate relationships
              are often modeled via CL and Vd separately.

    Parameterization of Random Effects (Mixed Effects):
        In population modeling, inter-individual variability (IIV) is estimated:
        - Concept: P_i = P_pop * exp(eta_i), where eta_i ~ N(0, omega^2).
        - Estimated Term: omega^2 (variance) or omega (SD) for parameters with IIV.
        - Common Parameters with IIV: CL and Vd typically exhibit IIV.
        - Initial Estimates for omega (SD): Common start ~0.3 (approx. 30% CV).
        - Bounds for omega^2 (Variance): Lower > 0 (e.g., 1e-6). Upper relatively
                                       large (e.g., 1 or 4).
    
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

class OneCompartmentBolus_Ke(PKBaseODE):
    """
    One-compartment IV Bolus model.
    Parameterized by Elimination Rate Constant (ke) and Volume of Distribution (vd).

    Alternative parameterization of the 1-compartment IV bolus model.

    States (y):
        y[0]: Mass in Central Compartment (amount)

    Parameters (*params for ode/mass_to_depvar):
        ke (float): First-order elimination rate constant (1/time).
        vd (float): Volume of distribution of the central compartment (volume).

    Initial Conditions (y0 for ODE solver):
        For a single IV bolus dose `Dose` at t=0:
        y0 = [Dose]

    Initial Parameter Estimates (for optimization):
        - ke: Estimate from the terminal slope (lambda_z) of log(concentration) vs time.
              For this model, ke = lambda_z.
        - vd: Estimate from NCA: Vd = Dose / (AUC_inf * ke). Alternatively,
              extrapolate C(t) to t=0 on semi-log plot to get C0, Vd ≈ Dose / C0.
              Check literature.
    
    Optimizer Bounds:
         - Lower Bounds: All parameters (cl, vd, ke) must be > 0 (e.g., 1e-9).
         - Upper Bounds: Use plausible physiological limits.
           - cl: Should not exceed cardiac output / relevant organ blood flow (e.g., < 100-200 L/hr for human).
           - vd: Highly variable. Informed by Vz from NCA. Start large if unsure (e.g., 1000 L).
           - ke: Usually < 10 1/hr. Related to cl/vd bounds.

    Common Covariate Associations (Fixed Effects):
        These parameters may be influenced by subject attributes:
        - cl: Associated with body weight (allometry), renal function (CrCL), hepatic
              function markers, age, genetic polymorphisms (relevant enzymes/transporters).
        - vd: Associated with body weight (allometry), body composition (lipophilicity), age.
        - ke: As a composite parameter (cl/vd), its direct covariate relationships
              are often modeled via CL and Vd separately.

    Parameterization of Random Effects (Mixed Effects):
        In population modeling, inter-individual variability (IIV) is estimated:
        - Concept: P_i = P_pop * exp(eta_i), where eta_i ~ N(0, omega^2).
        - Estimated Term: omega^2 (variance) or omega (SD) for parameters with IIV.
        - Common Parameters with IIV: CL and Vd typically exhibit IIV.
        - Initial Estimates for omega (SD): Common start ~0.3 (approx. 30% CV).
        - Bounds for omega^2 (Variance): Lower > 0 (e.g., 1e-6). Upper relatively
                                       large (e.g., 1 or 4).
              
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

class TwoCompartmentBolus(PKBaseODE):
    """
    Two-compartment IV Bolus model (Central, Peripheral).
    Parameterized by Clearance (cl), Volume Central (v1),
    Inter-compartmental Clearance (q), Volume Peripheral (v2).

    Models distribution between central (e.g., blood) and peripheral
    (e.g., tissue) compartments following IV bolus administration.
    Elimination occurs only from the central compartment.

    States (y):
        y[0]: Mass in Central Compartment (amount)
        y[1]: Mass in Peripheral Compartment (amount)

    Parameters (*params for ode/mass_to_depvar):
        cl (float): Clearance from the central compartment (volume/time).
        v1 (float): Volume of the central compartment (volume).
        q (float): Inter-compartmental clearance (volume/time).
        v2 (float): Volume of the peripheral compartment (volume).

    Initial Conditions (y0 for ODE solver):
        For a single IV bolus dose `Dose` at t=0:
        y0 = [Dose, 0]
        For later doses, add `Dose` to the residual mass in y[0].

    Initial Parameter Estimates (for optimization):
        - cl: Estimate from NCA: CL = Dose / AUC_inf.
        - v1: Often estimated initially from Vd ≈ Dose / C0 (from C(t) extrapolation),
              but V1 is typically smaller than Vd from NCA (Vdss or Vz). Start with
              a value smaller than Vdss (e.g., 1/3 to 1/2 Vdss), or based on physiology
              (e.g., plasma volume). Check literature.
        - q: Reflects the speed of distribution. Harder to estimate directly. Look
              at the shape of the initial decline (distribution phase). Can start
              with a value similar to CL or slightly higher/lower.
        - v2: Related to Vdss (Vd at steady state = V1 + V2). Estimate Vdss from
              NCA (Vdss = Dose * AUMC / AUC^2), then guess V2 = Vdss - V1 (initial guess).
              Ensure V1, V2, Q, CL are positive. Check literature.
    
    Optimizer Bounds:
         - Lower Bounds: All parameters (cl, v1, q, v2) must be > 0 (e.g., 1e-9).
         - Upper Bounds: Use plausible limits.
           - cl: < Cardiac output / organ blood flow (e.g., < 100-200 L/hr).
           - v1: Typically > plasma volume but < Vz. Maybe < 500 L.
           - q: Highly variable, can exceed CL. Maybe < 500 L/hr.
           - v2: Can be large. Maybe < 2000 L. Informed by Vdss/Vz.

    Common Covariate Associations (Fixed Effects):
        These parameters may be influenced by subject attributes:
        - cl: Associated with body weight (allometry), renal function (CrCL), hepatic
              function markers, age, genetics.
        - v1, v2: Associated with body weight (allometry), body composition, age.
        - q: Less commonly modeled with covariates, potentially related to cardiac
             output or body weight.

    Parameterization of Random Effects (Mixed Effects):
        In population modeling, inter-individual variability (IIV) is estimated:
        - Concept: P_i = P_pop * exp(eta_i), where eta_i ~ N(0, omega^2).
        - Estimated Term: omega^2 (variance) or omega (SD) for parameters with IIV.
        - Common Parameters with IIV: Typically CL and V1 show significant IIV.
                                     Variability on Q and V2 might also be estimated,
                                     depending on data richness.
        - Initial Estimates for omega (SD): Common start ~0.3 (approx. 30% CV) for CL, V1.
                                          May start lower for Q, V2.
        - Bounds for omega^2 (Variance): Lower > 0 (e.g., 1e-6). Upper relatively
                                       large (e.g., 1 or 4).
    
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

class TwoCompartmentAbsorption(PKBaseODE):
    """
    Two-compartment model with first-order absorption (Gut -> Central -> Peripheral).
    Parameterized by Absorption Rate Const (ka), Clearance (cl), Volume Central (v1),
    Inter-compartmental Clearance (q), Volume Peripheral (v2).

    Models drug absorption from a depot (e.g., gut) into a central compartment,
    distribution between the central and a peripheral compartment, and
    elimination from the central compartment. This model is more complex than
    a one-compartment model.

    States (y):
        y[0]: Mass in Central Compartment (amount)
        y[1]: Mass in Peripheral Compartment (amount)
        y[2]: Mass in Gut/Absorption Compartment (amount)

    Parameters (*params for ode/mass_to_depvar):
        ka (float): First-order absorption rate constant (1/time).
        cl (float): Clearance from the central compartment (volume/time).
        v1 (float): Volume of the central compartment (volume).
        q (float): Inter-compartmental clearance, governing distribution
                     speed between central and peripheral (volume/time).
        v2 (float): Volume of the peripheral compartment (volume).
        Note: For oral data, parameters involving volume or clearance are often
        estimated apparent values (e.g., cl/F, v1/F, q/F, v2/F) unless
        bioavailability (F) is known or estimated separately (e.g., from IV data).

    Initial Conditions (y0 for ODE solver):
        Typically, for a single dose `Dose` administered orally at t=0:
        y0 = [0, 0, Dose]  (Mass starts in the gut compartment)
        For subsequent doses in a regimen, the `Dose` is added to the
        residual mass in y[2] at the time of dosing.

    Initial Parameter Estimates (for optimization):
        Obtaining good initial estimates is crucial for model fitting.
        - ka: Related to Tmax (time of peak concentration). Rough initial guess:
              1/Tmax. Often requires adjustment. Try values like 0.5, 1.0, 2.0 (1/hr).
        - cl(/F): Estimate from NCA: CL/F = Dose / AUC_inf. (F=bioavailability)
        - v1(/F): Harder to estimate directly from oral data. Often smaller than
              total Vd. Start with a plausible physiological volume (e.g., plasma volume
              ~3-5L for humans) or a fraction of the total Vd estimate. Literature
              for similar drugs is helpful.
        - q(/F): Reflects distribution speed. Start with a value potentially similar
              to cl/F or based on literature/visual inspection of the distribution phase.
        - v2(/F): Related to the extent of distribution. Estimate total Vdss/F or Vz/F
              from NCA (e.g., Vz/F = (CL/F) / lambda_z). Initial guess:
              V2/F ≈ Vz/F - V1/F (ensure V1/F < Vz/F).
        Use NCA results and literature values as guidance. Ensure all parameters
        are positive. Test if initial guesses produce a curve shape roughly
        matching the data.
    
    Optimizer Bounds:
         - Lower Bounds: All parameters (ka, cl, v1, q, v2) must be > 0 (e.g., 1e-9).
         - Upper Bounds: Use plausible limits.
           - ka: Typically < 50 1/hr.
           - cl(/F): < Cardiac output / organ blood flow (e.g., < 100-200 L/hr).
           - v1(/F): Typically larger than plasma volume but smaller than Vz/F. Maybe < 500 L.
           - q(/F): Highly variable, can exceed CL. Maybe < 500 L/hr.
           - v2(/F): Can be large, especially for lipophilic drugs. Maybe < 2000 L.
           Bounds often informed by NCA Vz/F and Vdss/F.

    Common Covariate Associations (Fixed Effects):
        These parameters may be influenced by subject attributes:
        - cl(/F): Associated with body weight (allometry), renal function (CrCL), hepatic
                  function markers, age, genetics.
        - v1(/F), v2(/F): Associated with body weight (allometry), body composition, age.
        - q(/F): Less commonly modeled with covariates, but potentially related to
                 cardiac output or body weight.
        - ka: Influenced by formulation, food, GI conditions, age.
        - F (Implicit): Influenced by formulation, food, gut metabolism/transport.

    Parameterization of Random Effects (Mixed Effects):
        In population modeling, inter-individual variability (IIV) is estimated:
        - Concept: P_i = P_pop * exp(eta_i), where eta_i ~ N(0, omega^2).
        - Estimated Term: omega^2 (variance) or omega (SD) for parameters with IIV.
        - Common Parameters with IIV: Typically CL/F, V1/F, Ka often show significant
                                     IIV. V2/F and Q/F variability might also be estimated
                                     but can sometimes be harder to identify reliably.
        - Initial Estimates for omega (SD): Common start ~0.3 (approx. 30% CV) for CL,
                                          V1, Ka. May start lower for Q, V2.
        - Bounds for omega^2 (Variance): Lower > 0 (e.g., 1e-6). Upper relatively
                                       large (e.g., 1 or 4).

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

class OneCompartmentInfusion(PKBaseODE):
    """
    One-compartment model with zero-order infusion.
    Parameterized by Infusion Rate (R0), Clearance (cl), Volume (vd).

    Models drug administration via a constant rate infusion into a single
    compartment with first-order elimination.

    Note: Assumes infusion runs from t=0 up to Tinf (infusion duration).
    The ODE solver needs external logic to set R0=0 for t > Tinf.
    This class only defines the dynamics *while* R0 is potentially non-zero.

    States (y):
        y[0]: Mass in Central Compartment (amount)

    Parameters (*params for ode/mass_to_depvar):
        R0 (float): Zero-order infusion rate (amount/time). Should be set to 0
                    by the calling code for times after the infusion stops.
        cl (float): Clearance from the central compartment (volume/time).
        vd (float): Volume of distribution of the central compartment (volume).

    Initial Conditions (y0 for ODE solver):
        Typically, if starting at t=0 with the infusion:
        y0 = [0]
        If infusion follows a bolus, or starts later, y0 reflects mass at t=0.

    Initial Parameter Estimates (for optimization):
        - R0: Usually known from the dosing regimen (TotalDose / Tinf).
        - cl: If steady-state (Css) is reached: CL = R0 / Css. Otherwise, use
              NCA on post-infusion data (CL = Dose / AUC from Tinf to infinity)
              or literature.
        - vd: Estimate from post-infusion terminal slope (lambda_z): Vd = CL / lambda_z.
              Or Vd = R0 / (Css * lambda_z) if Css is reached.

    Optimizer Bounds (if estimating CL, Vd):
         - Lower Bounds: cl > 0, vd > 0 (e.g., 1e-9).
         - Upper Bounds: Plausible physiological limits.
           - cl: Not exceeding cardiac output / relevant organ blood flow (e.g., < 100-200 L/hr).
           - vd: Informed by Vz post-infusion. Start large if unsure (e.g., 1000 L).
         (R0 is usually fixed based on dose regimen).

    Common Covariate Associations (Fixed Effects):
        These parameters may be influenced by subject attributes:
        - cl: Associated with body weight (allometry), renal function (CrCL), hepatic
              function markers, age, genetics.
        - vd: Associated with body weight (allometry), body composition, age.
        - R0: Usually determined by dosing protocol, but variability in pump rate or
              delivery could potentially be modeled if relevant data exists (less common).

    Parameterization of Random Effects (Mixed Effects):
        In population modeling, inter-individual variability (IIV) is estimated:
        - Concept: P_i = P_pop * exp(eta_i), where eta_i ~ N(0, omega^2).
        - Estimated Term: omega^2 (variance) or omega (SD) for parameters with IIV.
        - Common Parameters with IIV: CL and Vd typically exhibit IIV. IIV on R0 is
                                     less common unless pump variability is a specific focus.
        - Initial Estimates for omega (SD): Common start ~0.3 (approx. 30% CV) for CL, Vd.
        - Bounds for omega^2 (Variance): Lower > 0 (e.g., 1e-6). Upper relatively
                                       large (e.g., 1 or 4).
                                       
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

class OneCompartmentBolusMM(PKBaseODE):
    """
    One-compartment IV Bolus model with Michaelis-Menten (saturable) elimination.
    Parameterized by Max Elimination Rate (vmax), Michaelis Constant (km),
    and Volume of Distribution (vd).

    Used when elimination process (e.g., enzyme metabolism, transport) can
    become saturated at higher concentrations.

    States (y):
        y[0]: Mass in Central Compartment (amount)

    Parameters (*params for ode/mass_to_depvar):
        vmax (float): Maximum rate of elimination (amount/time).
        km (float): Michaelis constant (concentration units). Concentration at
                     which elimination rate is half of Vmax.
        vd (float): Volume of distribution of the central compartment (volume).

    Initial Conditions (y0 for ODE solver):
        For a single IV bolus dose `Dose` at t=0:
        y0 = [Dose]

    Initial Parameter Estimates (for optimization):
        - vmax: Represents the maximum elimination capacity. Harder to estimate directly.
                Related to CL at low concentrations (CL_low_C ≈ Vmax/Km). Units are amount/time.
                Check literature for related enzyme kinetics or drugs.
        - km: Concentration at half-maximal elimination rate. Units are concentration.
              If data shows non-linear elimination (e.g., semi-log plot not straight),
              Km might be within the observed concentration range. Check literature.
        - vd: Volume relating mass to concentration (C = Mass/Vd). Might be estimated
              from low-dose data (where kinetics approximate first-order) using
              linear methods, or from literature.
        Estimation often requires data covering a range of concentrations, including
        those high enough to approach saturation. Initial guesses can be challenging.
    
    Optimizer Bounds:
         - Lower Bounds: vmax > 0, km > 0, vd > 0 (e.g., 1e-9).
         - Upper Bounds: Plausible limits.
           - vmax: Can be large, related to max enzyme/transporter capacity. Informed by
                   CL_linear * Km_guess. Start relatively large if needed.
           - km: Highly context dependent (concentration units). Can range from nM to mM.
                 Bounds should encompass expected therapeutic concentrations.
           - vd: Similar bounds as for linear Vd (e.g., < 1000 L).

    Common Covariate Associations (Fixed Effects):
        These parameters may be influenced by subject attributes:
        - vmax: Can be associated with body weight (allometry), organ function (liver/kidney
                depending on enzyme location), genetics (enzyme/transporter polymorphisms),
                drug interactions (induction/inhibition).
        - km: Generally considered less likely to vary systematically with simple covariates
              like weight, but could be affected by genetics or specific physiological states
              altering enzyme affinity (less common than Vmax effects).
        - vd: Associated with body weight (allometry), body composition, age.

    Parameterization of Random Effects (Mixed Effects):
        In population modeling, inter-individual variability (IIV) is estimated:
        - Concept: P_i = P_pop * exp(eta_i), where eta_i ~ N(0, omega^2). (Log-normal often
                   used, though normal on Km might sometimes be considered).
        - Estimated Term: omega^2 (variance) or omega (SD) for parameters with IIV.
        - Common Parameters with IIV: Vmax, Vd often show significant IIV. Km variability
                                     can sometimes be high or hard to estimate precisely.
        - Initial Estimates for omega (SD): Common start ~0.3 (approx. 30% CV) for Vmax, Vd.
                                          May need different starting guess for Km IIV.
        - Bounds for omega^2 (Variance): Lower > 0 (e.g., 1e-6). Upper relatively
                                       large (e.g., 1 or 4).
    
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
    

