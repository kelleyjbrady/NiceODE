no_me_mod_k =  CompartmentalModel(
     ode_t0_cols=[ODEInitVals('DV')],
     population_coeff=[PopulationCoeffcient('cl', 20, ),
                       PopulationCoeffcient('vd', 50, ),
                       ],
     dep_vars= None, 
     model_error_sigma=PopulationCoeffcient('sigma',
                                            optimization_init_val=4, 
                                            optimization_lower_bound=0.000001, 
                                            optimization_upper_bound=20),
                              no_me_loss_function=neg_log_likelihood_loss, 
                              optimizer_tol=None, 
                              pk_model_function=first_order_one_compartment_model2, 
                              #ode_solver_method='BDF'
                              )