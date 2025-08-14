import jax
import jax.numpy as jnp
import finitediffx as fdx
import jaxopt
import diffrax
from functools import partial
import sympy

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

# Use sympy to robustly derive the Jacobians for the augmented ODE
t = sympy.Symbol('t')
ka, cl, vd = sympy.symbols('ka cl vd')
y0, y1 = sympy.symbols('y0 y1') # Gut, Central
y_sym = sympy.Matrix([y0, y1])
p_sym = sympy.Matrix([ka, cl, vd])
f_sym = sympy.Matrix([-ka * y0, ka * y0 - (cl / vd) * y1])
Jy_sym = f_sym.jacobian(y_sym)
Jp_sym = f_sym.jacobian(p_sym)

# Lambdify the symbolic expressions into JAX functions
F_LAMBDA = sympy.lambdify((t, y_sym, p_sym), f_sym, 'jax')
Jy_LAMBDA = sympy.lambdify((t, y_sym, p_sym), Jy_sym, 'jax')
Jp_LAMBDA = sympy.lambdify((t, y_sym, p_sym), Jp_sym, 'jax')

def augmented_ode_for_S(t, y_and_S, args):
    """Computes dynamics for the state (y) and sensitivities (S)"""
    y, s = y_and_S
    # THE FIX: Ensure the output is a 1D vector of shape (2,)
    f_val = F_LAMBDA(t, y, args)[:, 0]
    Jy_val = Jy_LAMBDA(t, y, args)
    Jp_val = Jp_LAMBDA(t, y, args)
    ds_dt = Jy_val @ s + Jp_val
    return f_val, ds_dt

# ===================================================================
# 2. Solver and Loss Function Setup
# ===================================================================
stiff_solver = diffrax.Kvaerno5()
step_ctrl = diffrax.PIDController(rtol=1e-6, atol=1e-6)
#adjoint_method = diffrax.BacksolveAdjoint(solver=stiff_solver, stepsize_controller=step_ctrl)
adjoint_method = diffrax.RecursiveCheckpointAdjoint()

compiled_solver_vanilla = partial(diffrax.diffeqsolve, 
                                  solver=stiff_solver, t0=0.0, t1=24.0, dt0=0.1, 
                                  saveat=diffrax.SaveAt(ts=observation_times),
                                  stepsize_controller=step_ctrl,
                                  adjoint=adjoint_method, max_steps=4096)

compiled_solver_augmented = partial(diffrax.diffeqsolve,
                                    solver=stiff_solver, t0=0.0, t1=24.0, dt0=0.1,
                                    saveat=diffrax.SaveAt(ts=observation_times),
                                    stepsize_controller=step_ctrl,
                                    max_steps=4096)

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
    solution = compiled_solver_vanilla(terms=diffrax.ODETerm(lambda t,y,args: F_LAMBDA(t,y,args)[:,0]), y0=y0, args=model_coeffs)
    pred_y_i = solution.ys[:, 1]
    residuals_i = padded_y_i - pred_y_i
    loss_data = jnp.sum(residuals_i**2) / sigma2[0]
    inv_o2 = jnp.linalg.inv(omega2)
    loss_prior = b_i @ inv_o2 @ b_i
    return loss_data + loss_prior

# ===================================================================
# 3. The Hybrid VJP Estimator
# ===================================================================
@jax.custom_vjp
def estimate_b_i_hybrid(pop_coeff, sigma2, omega2):
    return _estimate_b_i_fwd(pop_coeff, sigma2, omega2)[0]

def _estimate_b_i_fwd(pop_coeff, sigma2, omega2):
    solver = jaxopt.LBFGSB(fun=mre_inner_loss, tol=1e-4, maxiter=100)
    bounds = (jnp.full_like(initial_b_i, -5.0), jnp.full_like(initial_b_i, 5.0))
    estimated_b_i, _ = solver.run(initial_b_i, bounds=bounds, pop_coeff=pop_coeff, sigma2=sigma2, omega2=omega2)

    model_coeffs = jnp.exp(jnp.log(pop_coeff) + estimated_b_i)
    y0_aug = (jnp.array([100.0, 0.0]), jnp.zeros((2, 3)))
    sol_aug = compiled_solver_augmented(terms=diffrax.ODETerm(augmented_ode_for_S), y0=y0_aug, args=model_coeffs)
    S = sol_aug.ys[1][:, 1, :]
    
    S_wrt_b = S * model_coeffs[None, :]
    H_data = 2 * (S_wrt_b.T @ S_wrt_b) / sigma2[0]
    inv_omega2 = jnp.linalg.inv(omega2)
    H_prior = 2 * inv_omega2
    H_foce = H_data + H_prior
    
    pred_y_i = sol_aug.ys[0][:, 1]
    residuals = padded_y_i - pred_y_i
    
    primal_out = estimated_b_i
    residuals_for_bwd = (estimated_b_i, H_foce, S, residuals, model_coeffs, pop_coeff, sigma2, omega2)
    return primal_out, residuals_for_bwd

def _estimate_b_i_bwd(residuals_for_bwd, g_b_i):
    est_b_i, H_foce, S, res, model_coeffs, pop, sig2, om2 = residuals_for_bwd
    v = jax.scipy.linalg.solve(H_foce, g_b_i, assume_a='pos')
    
    S_wrt_b = S * model_coeffs[None, :]
    S_wrt_pc = S * model_coeffs[None, :]
    J_cross_pc = (2 / sig2[0]) * (S_wrt_pc.T @ S_wrt_b)
    implicit_grad_pc = -v @ J_cross_pc
    
    J_cross_s2 = 2 * (S_wrt_b.T @ res) / sig2[0]**2
    implicit_grad_s2 = -v @ J_cross_s2
    
    inv_om2 = jnp.linalg.inv(om2)
    grad_b_prior = 2 * inv_om2 @ est_b_i
    J_cross_o2 = jax.jacobian(lambda o, b: 2 * jnp.linalg.inv(o) @ b, argnums=0)(om2, est_b_i)
    implicit_grad_o2 = -v @ J_cross_o2
    
    explicit_grad_o2 = jax.grad(lambda o, b: b @ jnp.linalg.inv(o) @ b, argnums=0)(om2, est_b_i)
    
    # Transform gradients from natural scale back to log-scale for pop_coeff
    grad_log_pop_coeff = implicit_grad_pc * pop
    
    return grad_log_pop_coeff, jnp.array([implicit_grad_s2]), explicit_grad_o2 + implicit_grad_o2

estimate_b_i_hybrid.defvjp(_estimate_b_i_fwd, _estimate_b_i_bwd)

# ===================================================================
# 4. The Final MRE Test
# ===================================================================
def estimate_b_i_forward_only(pop_coeff, sigma2, omega2):
    return _estimate_b_i_fwd(pop_coeff, sigma2, omega2)[0]

def final_outer_loss_fdx(params):
    pop_coeff, sigma2, omega2 = unpack_params(params)
    b_i = estimate_b_i_forward_only(pop_coeff, sigma2, omega2)
    inv_o2 = jnp.linalg.inv(omega2)
    return b_i @ inv_o2 @ b_i

def final_outer_loss_vjp(params):
    pop_coeff, sigma2, omega2 = unpack_params(params)
    b_i = estimate_b_i_hybrid(pop_coeff, sigma2, omega2)
    inv_o2 = jnp.linalg.inv(omega2)
    return b_i @ inv_o2 @ b_i

print("--- Running Final Hybrid MRE Comparison ---")
fdx_grad_mre = fdx.fgrad(final_outer_loss_fdx)(opt_params)
jax_grad_mre = jax.grad(final_outer_loss_vjp)(opt_params)

print("\nfinitediffx (MRE):\n", fdx_grad_mre)
print("\njax.grad (Hybrid VJP):\n", jax_grad_mre)