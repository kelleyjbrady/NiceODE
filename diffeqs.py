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

    Combines first-order absorption with two-compartment distribution and
    elimination from the central compartment.

    States (y):
        y[0]: Mass in Central Compartment (amount)
        y[1]: Mass in Peripheral Compartment (amount)
        y[2]: Mass in Gut/Absorption Compartment (amount)

    Parameters (*params for ode/mass_to_depvar):
        ka (float): First-order absorption rate constant (1/time).
        cl (float): Clearance from the central compartment (volume/time).
        v1 (float): Volume of the central compartment (volume).
        q (float): Inter-compartmental clearance (volume/time).
        v2 (float): Volume of the peripheral compartment (volume).

    Initial Conditions (y0 for ODE solver):
        Typically, for a single dose `Dose` at t=0:
        y0 = [0, 0, Dose]
        For later doses, add `Dose` to the residual mass in y[2].

    Initial Parameter Estimates (for optimization):
        - ka: Related to Tmax. Rough guess: 1/Tmax. Try values like 0.5, 1.0, 2.0 (1/hr).
        - cl: Estimate from NCA: CL/F = Dose / AUC_inf. Guess F if unknown.
        - v1: Harder to estimate directly. Start with a plausible physiological volume
              (e.g., plasma volume) or fraction of Vd/F. Check literature.
        - q: Distribution speed. Start with value similar to CL/F or based on literature.
        - v2: Related to Vdss/F = (CL/F) * AUMC/AUC - MRT. Then V2/F = Vdss/F - V1/F.
              Estimate Vdss/F from NCA or literature. Ensure positive values.
        Check literature for similar drugs extensively.
    """
    def __init__(self, ):
        pass

    def ode(self, t, y, ka, cl, v1, q, v2):
        """ODE system for 2-cmt absorption."""
        central_mass, peripheral_mass, gut_mass = y
        # Micro-rate constants (internal): k10=cl/v1, k12=q/v1, k21=q/v2
        dCMdt = ka * gut_mass + (q / v2) * peripheral_mass - (q / v1) * central_mass - (cl / v1) * central_mass
        dPMdt = (q / v1) * central_mass - (q / v2) * peripheral_mass
        dGdt = -ka * gut_mass
        return [dCMdt, dPMdt, dGdt]

    def mass_to_depvar(self, pred_mass_central, ka, cl, v1, q, v2):
        """Converts central compartment mass to concentration."""
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
        - cl: Estimate from steady-state concentration (Css), if reached:
              CL = R0 / Css. If not reached, use NCA on post-infusion data
              (CL = Dose / AUC from Tinf to infinity) or literature.
        - vd: Estimate from terminal slope (lambda_z) post-infusion:
              Vd = CL / lambda_z. Or Vd = R0 / (Css * lambda_z). Check literature.
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
