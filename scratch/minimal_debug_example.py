import numpy as np
from scipy.integrate import solve_ivp
from diffeqs import first_order_one_compartment_model  # Make sure this import is correct
from copy import deepcopy

fun = deepcopy(first_order_one_compartment_model)

def simple_test():
    t_span = (0, 10)
    y0 = [1.0]
    t_eval = np.linspace(0, 10, 100)
    sol = solve_ivp(fun, t_span, y0, t_eval=t_eval, args = (1,))
    print(sol)

if __name__ == "__main__":
    simple_test()