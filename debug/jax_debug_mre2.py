import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64']='True'
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Kvaerno5, PIDController, SaveAt, BacksolveAdjoint
import traceback

# 1. Fake, Simple Data (same as before)
init_params = {
    'k_absorption': jnp.array([1.5]),
    'k_elimination': jnp.array([0.8]),
    'omega_k_elim': jnp.array([0.2]),
}
y0 = jnp.array([100.0, 0.0]) # Initial condition for the ODE
ts = jnp.linspace(0., 10., 5) # Time points for evaluation

# 2. Simple ODE function for Diffrax
def ode_func(t, y, args):
    """A simple 1-compartment PK model. Now expects scalar args."""
    gut, central = y
    ka, ke = args  # ka and ke will now be scalar tracers
    dG_dt = -ka * gut
    dC_dt = ka * gut - ke * central
    # This will now correctly return a (2,) shape array
    return jnp.array([dG_dt, dC_dt])

# 3. Predictor that wraps diffeqsolve
def diffrax_predictor(ode_params_tuple):
    """Wraps diffeqsolve. Expects a tuple of scalar parameters."""
    solver = Kvaerno5()
    step_ctrl = PIDController(rtol=1e-6, atol=1e-6)
    adjoint = BacksolveAdjoint(solver=solver, stepsize_controller=step_ctrl)
    
    sol = diffeqsolve(
        terms=ODETerm(ode_func),
        solver=solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=0.1,
        y0=y0,
        args=ode_params_tuple, # Pass the tuple of scalar parameters
        saveat=SaveAt(ts=ts),
        stepsize_controller=step_ctrl,
        adjoint=adjoint
    )
    # Return just the central compartment predictions
    return sol.ys[:, 1]

# 4. Minimal Loss Function with Diffrax
def minimal_loss_with_diffrax(params):
    # Unpack params from the main dictionary
    ka = params['k_absorption']
    ke = params['k_elimination']
    omega_ke = params['omega_k_elim']

    jax.debug.print("--- MRE v2.1 --- params dictionary: {x}", x=params)

    # --- THIS IS THE FIX ---
    # Extract the SCALAR values from the parameter arrays.
    params_for_jac = (ka[0], ke[0])
    
    # Differentiate the predictor with respect to the scalar parameters
    jac_fn = jax.jacobian(diffrax_predictor)
    jacobian_tuple = jac_fn(params_for_jac)

    # Dummy loss depends on the jacobian and other params
    loss = jnp.sum(jacobian_tuple[0]) + jnp.sum(jacobian_tuple[1]) + jnp.sum(omega_ke**2)
    return loss, "hello from diffrax mre"

# 5. Run the test
print("--- Running MRE v2.1 with Diffrax ---")
value_and_grad_fn = jax.value_and_grad(minimal_loss_with_diffrax, has_aux=True)

try:
    (loss, aux), grads = value_and_grad_fn(init_params)
    print(f"\nSUCCESS!")
    print(f"Loss: {loss}")
    print(f"Grads: {grads}")
except Exception as e:
    print(f"\nError during MRE v2.1 value_and_grad:")
    traceback.print_exc()