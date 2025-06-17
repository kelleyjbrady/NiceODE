import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64']='True'
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Kvaerno5, PIDController, SaveAt, BacksolveAdjoint
import traceback

# 1. Fake, Simple Data using SCALAR arrays
init_params = {
    'k_absorption': jnp.array(1.5),
    'k_elimination': jnp.array(0.8),
    'omega_k_elim': jnp.array(0.2),
}
y0 = jnp.array([100.0, 0.0])
ts = jnp.linspace(0., 10., 5)

# 2. ODE function
def ode_func(t, y, args):
    gut, central = y
    ka, ke = args
    return jnp.array([-ka * gut, ka * gut - ke * central])

# 3. Predictor that wraps diffeqsolve
def diffrax_predictor(ode_params_tuple):
    solver = Kvaerno5()
    step_ctrl = PIDController(rtol=1e-6, atol=1e-6)
    adjoint = BacksolveAdjoint(solver=solver, stepsize_controller=step_ctrl)
    sol = diffeqsolve(
        terms=ODETerm(ode_func), solver=solver, t0=ts[0], t1=ts[-1], dt0=0.1, y0=y0,
        args=ode_params_tuple, saveat=SaveAt(ts=ts), stepsize_controller=step_ctrl,
        adjoint=adjoint
    )
    return sol.ys[:, 1]

# 4. The custom_vjp Solution
@jax.custom_vjp
def calculate_jacobian_for_fo(ode_params_tuple):
    """Calculates the jacobian with a manually defined gradient."""
    jac_fn = jax.jacobian(diffrax_predictor)
    return jac_fn(ode_params_tuple)

def jacobian_fwd(ode_params_tuple):
    return calculate_jacobian_for_fo(ode_params_tuple), None

def jacobian_bwd(residuals, g):
    # --- THIS IS THE FIX ---
    # The primal function takes ONE argument (a tuple).
    # So we must return a tuple of length ONE.
    # The single element of the returned tuple must be the gradient for the input,
    # which has the structure of a tuple of two things. So we return ((zero, zero),).
    return ((0.0, 0.0),)

calculate_jacobian_for_fo.defvjp(jacobian_fwd, jacobian_bwd)

# 5. The Top-Level Loss Function
def minimal_loss_final(params):
    ka = params['k_absorption']
    ke = params['k_elimination']
    omega_ke = params['omega_k_elim']
    
    params_for_jac = (ka, ke)
    jacobian_tuple = calculate_jacobian_for_fo(params_for_jac)

    loss = jnp.sum(jacobian_tuple[0]) + jnp.sum(jacobian_tuple[1]) + jnp.sum(omega_ke**2)
    return loss, "hello from working MRE"

# 6. Run the Final Test
print("--- Running Final MRE with Corrected custom_vjp ---")
value_and_grad_fn = jax.value_and_grad(minimal_loss_final, has_aux=True)

try:
    (loss, aux), grads = value_and_grad_fn(init_params)
    print(f"\nSUCCESS!")
    print(f"Loss: {loss}")
    print(f"Grads: {grads}")
except Exception:
    print(f"\nError during Final MRE value_and_grad:")
    traceback.print_exc()