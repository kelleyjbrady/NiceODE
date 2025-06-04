cp1805_prep



ode_param_models = []
for i in range(M_ode_params):
    model = lgb.LGBMRegressor(**lgbm_params_common)
    # "Fit" them initially to establish a starting prediction (e.g., our initial guess)
    model.fit(X_dummy_train, np.full(n_profiles_train, current_ode_param_estimates[i]))
    ode_param_models.append(model)

num_outer_iterations = 20 # Total outer loops
for outer_iter in range(num_outer_iterations):
    print(f"Outer Iteration: {outer_iter + 1}/{num_outer_iterations}")
    # Store the params predicted at the start of this iteration to pass to objectives
    params_at_iter_start = [model.predict(X_dummy_train)[0] for model in ode_param_models]

    for j in range(M_ode_params): # Iterate over each ODE parameter
        print(f" Optimizing Param {j} (Current val: {params_at_iter_start[j]:.4f})")
        
        # Create the specific objective for the j-th parameter
        # It uses the most recent estimates of other parameters
        # Note: This uses params_at_iter_start. For true coordinate descent,
        # you might use the *very latest* predictions if parameters are updated sequentially within the loop.
        # Using params_at_iter_start makes each parameter update based on the state at the start of the outer iter.
        
        other_params_for_obj = list(params_at_iter_start) # Copy

        custom_obj_for_j = make_custom_objective(
            param_index_to_optimize=j,
            all_current_params=other_params_for_obj, # Pass the current state of all params
            y_true_profiles=y_profiles_data[:n_profiles_train], # Use training subset
            initial_conditions_profiles=y_ics_data[:n_profiles_train],
            t_eval_points=t_eval_points
        )
        
        # Get the current model for parameter j
        current_model_j = ode_param_models[j]
        
        # Train/update this model for a few estimators
        # `init_model` allows for continued training (boosting)
        current_model_j.set_params(fobj=custom_obj_for_j, n_estimators=lgbm_params_common['n_estimators'])
        current_model_j.fit(
            X_dummy_train,
            y_dummy_targets_train, # Targets are not used by fobj directly
            init_model=current_model_j if outer_iter > 0 or j > 0 else None # Warm start
        )
        
        # Update the global estimate with the new prediction from this model
        current_ode_param_estimates[j] = current_model_j.predict(X_dummy_train)[0]
        print(f"  Updated Param {j} to: {current_ode_param_estimates[j]:.4f}")

    print(f" End of Outer Iter {outer_iter+1}: Current Pop Params: Alpha={current_ode_param_estimates[0]:.4f}, Beta={current_ode_param_estimates[1]:.4f}")
    # Optionally, evaluate overall loss on a validation set here

final_population_params = [model.predict(X_dummy_train)[0] for model in ode_param_models]
print(f"\nFinal Estimated Population Parameters: Alpha={final_population_params[0]:.4f}, Beta={final_population_params[1]:.4f}")
print(f"True population parameters were: Alpha={TRUE_ALPHA:.4f}, Beta={TRUE_BETA:.4f}")

# Final evaluation
final_avg_loss = 0
for i in range(n_profiles_total):
    y_pred_final = run_ivp_solver(final_population_params, y_ics_data[i], 
                                  (t_eval_points[0], t_eval_points[-1]), t_eval_points)
    final_avg_loss += calculate_profile_loss(y_pred_final, y_profiles_data[i])
print(f"Final average loss on all profiles: {final_avg_loss/n_profiles_total:.6f}")

