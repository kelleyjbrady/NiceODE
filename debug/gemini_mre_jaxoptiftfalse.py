import jax
import jax.numpy as jnp
import finitediffx as fdx
import jaxopt
import diffrax
from functools import partial

#this file demonstrates that unrolling the inner optmizer is the same as 'rolling your own' with optax

# ===================================================================
# 1. Hardcoded Inputs & ODE Definition
# ===================================================================
opt_params = jnp.array([
    -0.69, 1.1, 3.5, # log-scale pop_coeff for ka, cl, vd
    -1.38,           # log(sigma^2)
    -0.51, 0.0, -1.2, 0.0, 0.0, -2.3
], dtype=jnp.float64)

# Static data for a single subject
observation_times = jnp.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0], dtype=jnp.float64)
padded_y_i = jnp.array([0.0, 1.5, 2.8, 4.5, 4.0, 2.5, 1.0, 0.5, 0.1], dtype=jnp.float64)
initial_b_i = jnp.zeros(3, dtype=jnp.float64)
data_contrib_i = jnp.zeros(3, dtype=jnp.float64)


def one_compartment_ode(t, y, args):
    ka, cl, vd = args
    k_el = cl / vd
    dG_dt = -ka * y[0]
    dC_dt = ka * y[0] - k_el * y[1]
    return jnp.array([dG_dt, dC_dt])

# ===================================================================
# 2. Diffrax Solver Setup (With Stiff Solver and Adjoint)
# ===================================================================
stiff_ode_solver = diffrax.Kvaerno5()
step_ctrl = diffrax.PIDController(rtol=1e-6, atol=1e-6)
adjoint_method = diffrax.RecursiveCheckpointAdjoint()


compiled_solver = partial(diffrax.diffeqsolve, 
                          solver=stiff_ode_solver, 
                          t0=observation_times[0], 
                          t1=observation_times[-1], 
                          dt0=0.1, 
                          saveat=diffrax.SaveAt(ts=observation_times),
                          stepsize_controller=step_ctrl,
                          adjoint=adjoint_method,
                          max_steps=8192)

# ===================================================================
# 3. Inner Loss and Differentiable Optimizer
# ===================================================================
def unpack_params(params):
    pop_coeff = jnp.exp(params[0:3])
    sigma2 = jnp.exp(params[3])
    omega_chol_vals = params[4:10]
    omega_lchol = jnp.zeros((3, 3), dtype=jnp.float64).at[jnp.tril_indices(3)].set(omega_chol_vals)
    omega_lchol = omega_lchol.at[jnp.diag_indices(3)].set(jnp.exp(jnp.diag(omega_lchol)))
    omega2 = omega_lchol @ omega_lchol.T
    return pop_coeff, jnp.array([sigma2]), omega2

def mre_inner_loss(b_i, pop_coeff, sigma2, omega2):
    model_coeffs = jnp.exp(jnp.log(pop_coeff) + b_i)
    y0 = jnp.array([100.0, 0.0])
    solution = compiled_solver(terms=diffrax.ODETerm(one_compartment_ode), y0=y0, args=model_coeffs)
    pred_y_i = solution.ys[:, 1]
    residuals_i = padded_y_i - pred_y_i
    loss_data = jnp.sum(residuals_i**2) / sigma2[0]
    inv_o2 = jnp.linalg.inv(omega2)
    loss_prior = b_i @ inv_o2 @ b_i
    return loss_data + loss_prior

def estimate_b_i_with_jaxopt(pop_coeff, sigma2, omega2):
    solver = jaxopt.LBFGSB(fun=mre_inner_loss, 
                           tol=1e-4, 
                           maxiter=100,
                           # --- THE KEY CHANGE ---
                           # Differentiate by unrolling the loop, not IFT
                           implicit_diff=False)
                           
    lower_bounds = jnp.full_like(initial_b_i, -5.0)
    upper_bounds = jnp.full_like(initial_b_i, 5.0)
    solution = solver.run(initial_b_i, bounds=(lower_bounds, upper_bounds), 
                          pop_coeff=pop_coeff, sigma2=sigma2, omega2=omega2)
    return solution.params

# ===================================================================
# 4. The Final MRE Test
# ===================================================================
def final_outer_loss(params):
    pop_coeff, sigma2, omega2 = unpack_params(params)
    b_i = estimate_b_i_with_jaxopt(pop_coeff, sigma2, omega2)
    inv_o2 = jnp.linalg.inv(omega2)
    return b_i @ inv_o2 @ b_i

print("--- Running Final MRE Comparison (Unrolling AD) ---")
fdx_grad_mre = fdx.fgrad(final_outer_loss)(opt_params)
jax_grad_mre = jax.grad(final_outer_loss)(opt_params)

print("\nfinitediffx (MRE):\n", fdx_grad_mre)
print("\njax.grad (MRE with Unrolling AD):\n", jax_grad_mre)