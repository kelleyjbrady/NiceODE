#%%
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64']='True'
import jax
import jax.numpy as jnp

# 1. Fake, Simple Data
# Use a dictionary for the parameters, as in your original code.
init_params = {
    'pop_A': jnp.array([1.5]),
    'pop_B': jnp.array([0.8]),
    'omega_A': jnp.array([0.2]),
    'sigma': jnp.array([0.1])
}
# Fake observation data (e.g., 2 subjects, 5 observations)
some_other_data = jnp.ones((2, 5))

# 2. Dummy Predictor (The stand-in for your ODE solver)
# It now correctly expects a tuple of two values.
def dummy_predictor(pop_params_values):
    """A dummy function that simulates the ODE solver."""
    # This unpacking will now work correctly.
    pA, pB = pop_params_values
    
    # Return a dummy "prediction" that is a function of the params.
    return pA * jnp.sin(pB * some_other_data)

# 3. Dummy Jacobian Estimator (This structure remains the same)
def dummy_jacobian_estimator(params_subset):
    """Replicates the structure of your _estimate_jacobian_jax."""
    keys = tuple(params_subset.keys())
    values = tuple(params_subset.values())
    
    # Differentiate the dummy predictor w.r.t its values
    jac_fn = jax.jacobian(dummy_predictor)
    jacobian_pytree = jac_fn(values)
    
    return dict(zip(keys, jacobian_pytree))

# 4. Minimal Loss Function (Now corrected)
def minimal_loss(params):
    """The simplified loss function to test the grad transformation."""
    # 1. Unpack params
    pop_A = params['pop_A']
    pop_B = params['pop_B']
    omega_A = params['omega_A']

    # --- DEBUG PRINT ---
    # This is the most important line for debugging our core issue.
    jax.debug.print("--- MRE --- params dictionary: {x}", x=params)

    # 2. Call the dummy jacobian estimator
    # --- THIS IS THE CORRECTED PART ---
    # We now pass both pop_A and pop_B, which dummy_predictor expects.
    params_for_jac = {'pop_A': pop_A, 'pop_B': pop_B}
    j = dummy_jacobian_estimator(params_for_jac)

    # 3. Calculate a dummy loss
    # The loss must depend on the jacobian and other params.
    # We extract values from the jacobian dict to use them.
    loss = jnp.sum(j['pop_A']) + jnp.sum(j['pop_B']) + jnp.sum(omega_A**2)
    
    return loss, ('hello world')
#%%
# 5. Try to Reproduce the Error
print("--- Running Forward Pass ---")
forward_loss = minimal_loss(init_params)
print(f"Forward pass loss: {forward_loss}")

print("\n--- Running value_and_grad ---")
# If the bug is present, the jax.debug.print inside minimal_loss will
# show an empty params dictionary here.
value_and_grad_fn = jax.value_and_grad(minimal_loss, has_aux = True)
try:
    (loss, aux_data), grads= value_and_grad_fn(init_params)
    
    print(f"Loss from grad call: {loss}")
    print(f"Grads: {grads}")
except Exception as e:
    print(f"\nError during value_and_grad: {e}")