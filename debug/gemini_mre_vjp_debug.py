import jax
import jax.numpy as jnp
import optax
import finitediffx as fdx

# ===================================================================
# 1. Hardcoded Inputs for the Minimal Reproducible Example
# ===================================================================
# Parameters to be optimized (a flat vector)
opt_params = jnp.array([
    0.47, 1.09, 3.55,  # pop_coeff for ka, cl, vd
    -1.386,            # log(sigma^2) -> so sigma^2 = 0.25
    -0.51, 0.0, -1.20,  # omega cholesky factors
    0.0, 0.0, -2.30
], dtype=jnp.float64)

# Static data for a single subject
padded_y_i = jnp.array([0.74, 2.84, 6.57, 10.5, 9.66, 8.58, 8.36, 7.47, 6.89, 5.94, 3.28], dtype=jnp.float64)
time_mask_y_i = jnp.ones_like(padded_y_i, dtype=bool)
data_contrib_i = jnp.zeros(3, dtype=jnp.float64)
initial_b_i = jnp.zeros(3, dtype=jnp.float64)
pop_coeff_w_bi_idx = jnp.array([0, 1, 2])

# ===================================================================
# 2. Helper Functions (Unpacking, Inner Loss)
# ===================================================================
def unpack_params(params):
    """Unpacks the flat vector into structured parameters."""
    pop_coeff = params[0:3]
    sigma2 = jnp.exp(params[3]) # Note: No squaring, exp(log(sigma^2))
    omega_chol_vals = params[4:10]
    
    omega_lchol = jnp.zeros((3, 3), dtype=jnp.float64).at[jnp.tril_indices(3)].set(omega_chol_vals)
    omega_lchol = omega_lchol.at[jnp.diag_indices(3)].set(jnp.exp(jnp.diag(omega_lchol)))
    omega2 = omega_lchol @ omega_lchol.T
    
    return pop_coeff, jnp.array([sigma2]), omega2 # Ensure sigma2 is shape (1,)

def toy_inner_loss(b_i, pop_coeff, sigma2, omega2):
    """The simplified inner loss using the linear model."""
    model_coeffs_i = jnp.exp(data_contrib_i + pop_coeff + b_i)
    
    A = jnp.eye(time_mask_y_i.shape[0], model_coeffs_i.shape[0])
    pred_y_i = A @ model_coeffs_i
    
    residuals_i = jnp.where(time_mask_y_i, padded_y_i - pred_y_i, 0.0)
    sum_sq_residuals = jnp.sum(residuals_i**2)
    
    loss_data = jnp.sum(time_mask_y_i) * jnp.log(sigma2[0]) + sum_sq_residuals / sigma2[0]
    
    L, _ = jax.scipy.linalg.cho_factor(omega2, lower=True)
    log_det_omega2 = 2 * jnp.sum(jnp.log(jnp.diag(L)))
    prior_penalty = b_i @ jax.scipy.linalg.cho_solve((L, True), b_i)
    
    return loss_data + log_det_omega2 + prior_penalty

# ===================================================================
# 3. The Custom VJP Estimator (IFT version)
# ===================================================================
def _estimate_b_i_impl(pop_coeff, sigma2, omega2):
    """Forward pass: finds b_i and calculates values needed for VJP."""
    obj_fn = lambda b_i: toy_inner_loss(b_i, pop_coeff, sigma2, omega2)
    
    optimizer = optax.adam(0.05)
    opt_state = optimizer.init(initial_b_i)
    grad_fn = jax.grad(obj_fn)

    def update_step(i, state):
        params, opt_state = state
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    estimated_b_i, _ = jax.lax.fori_loop(0, 10000, update_step, (initial_b_i, opt_state))

    # --- Analytical values for the Toy Problem ---
    model_coeffs = jnp.exp(data_contrib_i + pop_coeff + estimated_b_i)
    
    A = jnp.eye(time_mask_y_i.shape[0], pop_coeff.shape[0])
    S_simple = A @ jnp.diag(model_coeffs)
    H_data = 2 * (S_simple.T @ S_simple) / sigma2[0]
    
    inv_omega2 = jnp.linalg.inv(omega2)
    H_prior = 2 * inv_omega2
    H_foce = H_data + H_prior

    simple_preds = A @ model_coeffs
    simple_residuals = jnp.where(time_mask_y_i, padded_y_i - simple_preds, 0.0)

    return (estimated_b_i, H_foce, S_simple, simple_residuals, model_coeffs)

def _estimate_b_i_fwd(pop_coeff, sigma2, omega2):
    est_b_i, H_foce, S, res, model_coeffs = _estimate_b_i_impl(pop_coeff, sigma2, omega2)
    return est_b_i, (est_b_i, H_foce, S, res, model_coeffs, pop_coeff, sigma2, omega2)

def _estimate_b_i_bwd(residuals, g_b_i):
    """Backward pass: computes the VJP."""
    est_b_i, H_foce, S, res, model_coeffs, pop, sig2, om2 = residuals
    
    v = jax.scipy.linalg.solve(H_foce, g_b_i, assume_a='pos')
    
    # Implicit grad for pop_coeff
    S_wrt_pc = S * model_coeffs[None, :]
    J_cross_pc = (2 / sig2[0]) * (S_wrt_pc.T @ S) # H term is zero for linear model
    implicit_grad_pc = -v @ J_cross_pc
    
    # Implicit grad for sigma2
    J_cross_s2 = 2 * (S.T @ res) / sig2[0]**2
    implicit_grad_s2_scalar = -v @ J_cross_s2
    
    # Implicit grad for omega2
    def prior_grad_fn(b, o2): return jax.grad(lambda b,o2: b @ jnp.linalg.inv(o2) @ b, argnums=0)(b,o2)
    J_cross_o2 = jax.jacobian(prior_grad_fn, argnums=1)(est_b_i, om2)
    implicit_grad_o2 = -v @ J_cross_o2
    
    # This outer loss has no explicit dependency on pop_coeff or sigma2
    explicit_grad_pc = jnp.zeros_like(pop)
    explicit_grad_s2_scalar = 0.0

    # It does have an explicit dependency on omega2
    def explicit_omega_loss(o2, b): return b @ jnp.linalg.inv(o2) @ b
    explicit_grad_o2 = jax.grad(explicit_omega_loss, argnums=0)(om2, est_b_i)
    
    # Return tuple of gradients with shapes matching primal inputs
    return (implicit_grad_pc + explicit_grad_pc, 
            jnp.array([implicit_grad_s2_scalar + explicit_grad_s2_scalar]), # <-- THE FIX
            implicit_grad_o2 + explicit_grad_o2)

@jax.custom_vjp
def estimate_b_i_final(pop_coeff, sigma2, omega2):
    return _estimate_b_i_impl(pop_coeff, sigma2, omega2)[0]
estimate_b_i_final.defvjp(_estimate_b_i_fwd, _estimate_b_i_bwd)

# ===================================================================
# 4. The Final MRE Test
# ===================================================================
def final_outer_loss(params):
    pop_coeff, sigma2, omega2 = unpack_params(params)
    b_i = estimate_b_i_final(pop_coeff, sigma2, omega2)
    inv_o2 = jnp.linalg.inv(omega2)
    return b_i @ inv_o2 @ b_i

print("--- Running Final MRE Comparison ---")
fdx_grad_mre = fdx.fgrad(final_outer_loss)(opt_params)
jax_grad_mre = jax.grad(final_outer_loss)(opt_params)

print("\nfinitediffx (MRE):\n", fdx_grad_mre)
print("\njax.grad (VJP):\n", jax_grad_mre)