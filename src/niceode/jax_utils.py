import numpy as np
import jax.numpy as jnp


def FO_approx_ll_loss(
    pop_coeffs,
    sigma,
    omegas,
    thetas,
    theta_data,
    model_obj,
    y,
    y_groups_idx, 
    
    solve_for_omegas=False,
    
    **kwargs,
):
    # unpack some variables locally for clarity

    omegas_names = list(omegas.columns)
    omegas = omegas.to_numpy(
        dtype=np.float64
    ).flatten()  # omegas as SD, we want Variance, thus **2 below
    omegas2 = np.diag(
        omegas**2
    )  # FO assumes that there is no cov btwn the random effects, thus off diags are zero
    #this is not actually FO's assumption, but a simplification, 
    sigma = sigma.to_numpy(dtype=np.float64)[0]
    sigma2 = sigma**2
    n_individuals = len(model_obj.unique_groups)
    n_random_effects = len(omegas_names)

    # estimate model coeffs when the omegas are zero -- the first term of the taylor exapansion apprx
    model_coeffs = model_obj._generate_pk_model_coeff_vectorized(
        pop_coeffs, thetas, theta_data
    )
    # would be good to create wrapper methods inside of the model obj with these args (eg. `parallel = model_obj.parallel`)
    # prepopulated with partial
    preds, full_preds = model_obj._solve_ivp(
        model_coeffs,
        parallel=False,
    )
    residuals = y - preds

    # estimate jacobian
    # there is something going on with the way this is filtered to use fprime and
    # when there are multiple omegas
    apprx_fprime_jac = False
    central_diff_jac = True
    use_adaptive = True
    J = estimate_jacobian(
        pop_coeffs,
        thetas,
        theta_data,
        omegas_names,
        y,
        model_obj,
        use_fprime=apprx_fprime_jac,
        use_cdiff=central_diff_jac,
        use_adaptive=use_adaptive,
    )

    if np.all(J == 0):
        raise ValueError("All elements of the Jacobian are zero")
    # drop initial values from relevant arrays if `model_obj.ode_t0_vals_are_subject_y0`
    # perfect predictions can cause issues during optimization and also add no information to the loss
    # If there are any subjects with only one data point this will fail by dropping the entire subject
    if model_obj.ode_t0_vals_are_subject_y0:
        drop_idx = model_obj.subject_y0_idx
        J = np.delete(J, drop_idx, axis=0)
        residuals = np.delete(residuals, drop_idx)
        y = np.delete(y, drop_idx)
        y_groups_idx = np.delete(y_groups_idx, drop_idx)

    # Estimate the covariance matrix, then estimate neg log likelihood
    direct_det_cov = False
    per_sub_direct_neg_ll = False
    cholsky_cov = True
    neg2_ll = estimate_neg_log_likelihood(
        J,
        y_groups_idx,
        y,
        residuals,
        sigma2,
        omegas2,
        n_individuals,
        n_random_effects,
        model_obj,
        cholsky_cov=cholsky_cov,
        naive_cov_vec=direct_det_cov,
        naive_cov_subj=per_sub_direct_neg_ll,
    )

    # if predicting or debugging, solve for the optimal b_i given the first order apprx
    # perhaps b_i approx is off in the 2cmpt case bc the model was learned on t1+ w/
    # t0 as the intial condition but now t0 is in the data w/out a conc in the DV
    # col, this should make the resdiduals very wrong
    b_i_approx = np.zeros((n_individuals, n_random_effects))
    if solve_for_omegas:
        for sub_idx, sub in enumerate(model_obj.unique_groups):
            filt = y_groups_idx == sub
            J_sub = J[filt]
            residuals_sub = residuals[filt]
            try:
                # Ensure omegas2 is invertible (handle near-zero omegas)
                # Add a small value to diagonal for stability if needed, or use pinv
                min_omega_var = 1e-9  # Example threshold
                stable_omegas2 = np.diag(np.maximum(np.diag(omegas2), min_omega_var))
                omega_inv = np.linalg.inv(stable_omegas2)

                # Corrected matrix A
                A = J_sub.T @ J_sub + sigma2 * omega_inv
                # Right-hand side
                rhs = J_sub.T @ residuals_sub
                # Solve
                b_i_approx[sub_idx, :] = np.linalg.solve(A, rhs)

            except np.linalg.LinAlgError:
                print(
                    f"Warning: Linear algebra error (likely singular matrix) for subject {sub}. Setting b_i to zero."
                )

            except Exception as e:
                print(f"Error calculating b_i for subject {sub}: {e}")

    return neg2_ll, b_i_approx, (preds, full_preds)