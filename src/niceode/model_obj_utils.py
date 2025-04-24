from utils import  get_function_args
from utils import PopulationCoeffcient, ObjectiveFunctionColumn
from copy import deepcopy

def template_CompartmentalModel_params(ode_func):
    # t and y must be the first 2 args to be compatible with many solvers
    ode_params =  get_function_args(ode_func)[2:]
    ode_params = [i.lower().strip() for i in ode_params]
    #output_size = determine_ode_output_size(ode_func)

    pop_coeffs = []
    dep_vars = {}
    for ode_param in ode_params:
        pop_coeffs.append(PopulationCoeffcient(ode_param, optimization_init_val=1))
        dep_vars[ode_param] = [
                                ObjectiveFunctionColumn(f'{ode_param}_indepvar_placeholder_1'), 
                                ObjectiveFunctionColumn(f'{ode_param}_indepvar_placeholder_2')
                            ]


    return {'pop_coeffs': deepcopy(pop_coeffs),'dep_vars': deepcopy(dep_vars),  }
