import jax
import jax.numpy as jnp
import finitediffx as fdx
import jaxopt
import diffrax
from functools import partial
import sympy

#this file is an MRE for developing a custom vjp for an inner optmizer containing diffrax.diffeqsolve

# ===================================================================
# 1. Inputs & Full Augmented ODE Definition
# ===================================================================
opt_params = jnp.array([-0.69, 1.1, 3.5, -1.38, -0.51, 0.0, -1.2, 0.0, 0.0, -2.3], dtype=jnp.float64)
observation_times = jnp.array([0.0, 0.25, 0.57, 1.12, 2.02, 3.82, 5.1, 7.03, 9.05, 12.12, 24.37], dtype=jnp.float64)
padded_y_i = jnp.array([0.74, 2.84, 6.57, 10.5, 9.66, 8.58, 8.36, 7.47, 6.89, 5.94, 3.28], dtype=jnp.float64)
initial_b_i = jnp.zeros(3, dtype=jnp.float64)
data_contrib_i = jnp.zeros(3, dtype=jnp.float64)

# --- Sympy Derivation for 1st and 2nd order sensitivities ---
t, ka, cl, vd = sympy.symbols('t ka cl vd')
y0, y1 = sympy.symbols('y0 y1'); y_sym = sympy.Matrix([y0, y1])
p_sym = sympy.Matrix([ka, cl, vd]); n_p = len(p_sym)
f_sym = sympy.Matrix([-ka * y0, ka * y0 - (cl / vd) * y1])
Jy_sym = f_sym.jacobian(y_sym)
Jp_sym = f_sym.jacobian(p_sym)
dJp_dp_sym_list = [sympy.diff(Jp_sym, p) for p in p_sym]
dJy_dp_sym_list = [sympy.diff(Jy_sym, p) for p in p_sym]

# Lambdify all components
F_LAMBDA = sympy.lambdify((t, y_sym, p_sym), f_sym, 'jax')
Jy_LAMBDA = sympy.lambdify((t, y_sym, p_sym), Jy_sym, 'jax')
Jp_LAMBDA = sympy.lambdify((t, y_sym, p_sym), Jp_sym, 'jax')
dJp_dp_LAMBDA_list = [sympy.lambdify((t, y_sym, p_sym), term, 'jax') for term in dJp_dp_sym_list]
dJy_dp_LAMBDA_list = [sympy.lambdify((t, y_sym, p_sym), term, 'jax') for term in dJy_dp_sym_list]

def augmented_ode_for_S_and_H(t, y_S_H, args):
    y, S, H = y_S_H
    
    f_val = F_LAMBDA(t, y, args)[:, 0]
    Jy_val = Jy_LAMBDA(t, y, args)
    Jp_val = Jp_LAMBDA(t, y, args)
    dS_dt = Jy_val @ S + Jp_val
    
    # Dynamics for H (dHk/dt = Jy*Hk + (dJy/dpk)*S + dJp/dpk)
    dH_dt_list = []
    for k in range(n_p):
        H_k = H[k, :, :] # H is stored as a stack of matrices
        dJp_dpk = dJp_dp_LAMBDA_list[k](t, y, args)
        dJy_dpk_explicit = dJy_dp_LAMBDA_list[k](t, y, args)
        # The chain rule part (dJy/dy * dy/dpk) is zero for this simple model
        total_dJy_dpk = dJy_dpk_explicit 
        dHk_dt = Jy_val @ H_k + total_dJy_dpk @ S + dJp_dpk
        dH_dt_list.append(dHk_dt)
    
    dH_dt = jnp.stack(dH_dt_list)
    return f_val, dS_dt, dH_dt

# ===================================================================
# 2. Solvers and Unpacker
# ===================================================================
stiff_solver = diffrax.Kvaerno5()
step_ctrl = diffrax.PIDController(rtol=1e-7, atol=1e-7)
#djoint_method = diffrax.BacksolveAdjoint(solver=stiff_solver, stepsize_controller=step_ctrl)
adjoint_method = diffrax.RecursiveCheckpointAdjoint()

compiled_solver_vanilla = partial(diffrax.diffeqsolve, solver=stiff_solver, t0=0.0, t1=25.0, dt0=0.1, 
                                  saveat=diffrax.SaveAt(ts=observation_times), stepsize_controller=step_ctrl,
                                  adjoint=adjoint_method, max_steps=8192)
compiled_solver_augmented = partial(diffrax.diffeqsolve, solver=stiff_solver, t0=0.0, t1=25.0, dt0=0.1,
                                    saveat=diffrax.SaveAt(ts=observation_times),
                                    stepsize_controller=step_ctrl, max_steps=8192)


def unpack_params(params):
    pop_coeff, sigma2_val = jnp.exp(params[0:3]), jnp.exp(params[3])
    omega_chol_vals = params[4:10]
    L = jnp.zeros((3, 3)).at[jnp.tril_indices(3)].set(omega_chol_vals)
    L = L.at[jnp.diag_indices(3)].set(jnp.exp(jnp.diag(L)))
    return pop_coeff, jnp.array([sigma2_val]), L @ L.T

# ===================================================================
# 3. The Final Hybrid VJP
# ===================================================================
@jax.custom_vjp
def estimate_b_i_hybrid(pop_coeff, sigma2, omega2):
    return _estimate_b_i_fwd(pop_coeff, sigma2, omega2)[0]

def mre_inner_loss(b_i, pop_coeff, sigma2, omega2):
    model_coeffs = jnp.exp(jnp.log(pop_coeff) + data_contrib_i + b_i)
    y0 = jnp.array([319.992, 0])
    solution = compiled_solver_vanilla(terms=diffrax.ODETerm(lambda t,y,args: F_LAMBDA(t,y,args)[:,0]), y0=y0, args=model_coeffs)
    pred_y_i = solution.ys[:, 1]
    residuals_i = padded_y_i - pred_y_i
    loss_data = len(padded_y_i) * jnp.log(sigma2[0]) + jnp.sum(residuals_i**2) / sigma2[0]
    inv_o2 = jnp.linalg.inv(omega2)
    return loss_data + b_i @ inv_o2 @ b_i + jnp.linalg.slogdet(omega2)[1]

def _estimate_b_i_fwd(pop_coeff, sigma2, omega2):
    solver = jaxopt.LBFGSB(fun=mre_inner_loss, tol=1e-5, maxiter=200)
    bounds = (jnp.full_like(initial_b_i, -5.0), jnp.full_like(initial_b_i, 5.0))
    estimated_b_i, _ = solver.run(initial_b_i, bounds=bounds, pop_coeff=pop_coeff, sigma2=sigma2, omega2=omega2)

    model_coeffs = jnp.exp(jnp.log(pop_coeff) + data_contrib_i + estimated_b_i)
    # Solve 2nd-order system to get S and H
    y0_aug = (jnp.array([319.992, 0]), jnp.zeros((2, 3)), jnp.zeros((3, 2, 3)))
    sol_aug = compiled_solver_augmented(terms=diffrax.ODETerm(augmented_ode_for_S_and_H), y0=y0_aug, args=model_coeffs)
    S = sol_aug.ys[1][:, 1, :] # Sensitivities of the central compartment
    H_tensor = sol_aug.ys[2][:, :, 1, :] # Second-order sensitivities of the central comp.
    
    pred_y_i = sol_aug.ys[0][:, 1]
    residuals = padded_y_i - pred_y_i
    
    # Calculate FULL Hessian of inner loss
    S_wrt_b = S @ jnp.diag(model_coeffs)
    H_term_inner = jnp.einsum('tij,t->ij', H_tensor, residuals)
    H_data = (2 / sigma2[0]) * (S_wrt_b.T @ S_wrt_b - H_term_inner)
    inv_omega2 = jnp.linalg.inv(omega2)
    H_prior = 2 * inv_omega2
    H_foce = H_data + H_prior
    
    return estimated_b_i, (estimated_b_i, H_foce, S, H_tensor, residuals, model_coeffs, pop_coeff, sigma2, omega2)


def _estimate_b_i_bwd(residuals_for_bwd, g_b_i):
    est_b_i, H_foce, S, H, res, mc, pop, sig2, om2 = residuals_for_bwd
    v = jax.scipy.linalg.solve(H_foce, g_b_i, assume_a='pos')
    
    S_wrt_b = S @ jnp.diag(mc)
    S_wrt_pc = S @ jnp.diag(mc)
    
    # J_cross_pc with full second-order term
    H_wrt_b_pc = (H.transpose((0, 2, 1)) @ jnp.diag(mc)) @ jnp.diag(mc) # Simplified for MRE
    term1_final = jnp.einsum('tij,t->ij', H_wrt_b_pc, res)
    term2_final = S_wrt_b.T @ S_wrt_pc
    J_cross_pc = (2.0 / sig2[0]) * (term2_final - term1_final)
    grad_pop_coeff = -v @ J_cross_pc

    J_cross_s2 = 2.0 * (S_wrt_b.T @ res) / sig2[0]**2
    implicit_grad_s2 = -v @ J_cross_s2
    explicit_grad_s2 = (jnp.sum(res**2) / sig2[0]**2) * -1.0 + len(res) / sig2[0]
    grad_sigma2 = jnp.array([implicit_grad_s2 + explicit_grad_s2])
    
    def inner_prior_grad_b(b, o):
        return jax.grad(lambda b_in, o_in: b_in @ jnp.linalg.inv(o_in) @ b_in + jnp.linalg.slogdet(o_in)[1], argnums=0)(b, o)
    J_cross_o2 = jax.jacobian(inner_prior_grad_b, argnums=1)(est_b_i, om2)
    implicit_grad_o2 = -v @ J_cross_o2
    
    def explicit_outer_loss_o2(o, b): return b @ jnp.linalg.inv(o) @ b + jnp.linalg.slogdet(o)[1]
    explicit_grad_o2 = jax.grad(explicit_outer_loss_o2, argnums=0)(om2, est_b_i)
    grad_omega2 = implicit_grad_o2 + explicit_grad_o2

    return grad_pop_coeff, grad_sigma2, grad_omega2

estimate_b_i_hybrid.defvjp(_estimate_b_i_fwd, _estimate_b_i_bwd)

# ===================================================================
# 4. The Final MRE Test
# ===================================================================
def unpacked_loss(pop_coeff, sigma2, omega2):
    b_i = estimate_b_i_hybrid(pop_coeff, sigma2, omega2)
    inv_o2 = jnp.linalg.inv(omega2)
    return b_i @ inv_o2 @ b_i + jnp.linalg.slogdet(omega2)[1]

def final_outer_loss(params):
    pop_coeff, sigma2, omega2 = unpack_params(params)
    return unpacked_loss(pop_coeff, sigma2, omega2)

def final_outer_loss_fdx(params):
    pop_coeff, sigma2, omega2 = unpack_params(params)
    b_i = _estimate_b_i_fwd(pop_coeff, sigma2, omega2)[0]
    inv_o2 = jnp.linalg.inv(omega2)
    return b_i @ inv_o2 @ b_i + jnp.linalg.slogdet(omega2)[1]

print("--- Running Final Hybrid MRE Comparison ---")
fdx_grad_mre = fdx.fgrad(final_outer_loss_fdx)(opt_params)
jax_grad_mre = jax.grad(final_outer_loss)(opt_params)

print("\nfinitediffx (MRE):\n", fdx_grad_mre)
print("\njax.grad (Hybrid VJP):\n", jax_grad_mre)