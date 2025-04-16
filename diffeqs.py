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
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def ode(self,):
        pass
    
    @abc.abstractmethod
    def mass_to_depvar(self):
        pass

class OneCompartmentAbsorption(PKBaseODE):
    def __init__(self, ):
        pass
        
    def ode(self, t, y, ka, cl, vd):
        central_mass, gut_mass = y
        dCMdt = ka * gut_mass - (cl/vd) * central_mass
        dGdt = -ka * gut_mass
        return [dCMdt, dGdt]
    
    def mass_to_depvar(self, pred_mass, ka, cl, vd):
        depvar_unit_result = pred_mass/vd
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

