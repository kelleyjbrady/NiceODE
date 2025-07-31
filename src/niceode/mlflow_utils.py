import inspect
import ast
import sys
import hashlib
import mlflow
from scipy.optimize import OptimizeResult
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax

class _DocstringRemover(ast.NodeTransformer):
    """
    An AST transformer that removes docstrings from ClassDef, FunctionDef,
    and AsyncFunctionDef nodes.
    """

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        # Check if the class has a docstring
        if ast.get_docstring(node, clean=False) is not None:
            # If it does, the first statement in its body is the docstring Expr node.
            # Remove it, but only if the body is not empty and the first item is an Expr.
            if node.body and isinstance(node.body[0], ast.Expr):
                node.body = node.body[1:]
            # If the body becomes empty, AST unparser might need a 'pass'
            # which it usually handles automatically.
        self.generic_visit(node)  # Continue to visit methods, nested classes, etc.
        return node

    def _remove_function_docstring(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.FunctionDef | ast.AsyncFunctionDef:
        """Helper to remove docstrings from function-like nodes."""
        if ast.get_docstring(node, clean=False) is not None:
            if node.body and isinstance(node.body[0], ast.Expr):
                node.body = node.body[1:]
        self.generic_visit(node)  # Visit the rest of the function body
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return self._remove_function_docstring(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        return self._remove_function_docstring(node)

def get_class_source_without_docstrings(cls_obj: type) -> str | None:
    """
    Retrieves the source code text of a Python class, excluding all docstrings
    (class-level and method-level).

    This function attempts to return the raw text definition as found in the
    source file, with docstrings removed. The formatting of the output
    code is determined by `ast.unparse` (Python 3.9+) or a similar
    tool if a fallback were implemented for older versions.

    Args:
        cls_obj: The class object to inspect.

    Returns:
        A string containing the source code of the class without docstrings.
        Returns None or an error message string if the source cannot be
        retrieved, parsed, or if `ast.unparse` is unavailable (on Python < 3.9
        without a fallback like 'astor').
    """
    if not inspect.isclass(cls_obj):
        # You might want to raise a TypeError here for programmatic use.
        return f"Error: Expected a class object, but got {type(cls_obj).__name__}."

    try:
        # Get the raw source code of the class.
        # inspect.getsource() usually returns a string that's directly parsable.
        source_text = inspect.getsource(cls_obj)
    except (TypeError, OSError) as e:
        # TypeError: cls_obj is a built-in class, module, or function.
        # OSError: source code cannot be retrieved (e.g., defined in C, REPL, or dynamically generated).
        return f"Error: Could not retrieve source for class '{cls_obj.__name__}'. Reason: {e}"

    try:
        # Parse the source text into an Abstract Syntax Tree (AST)
        parsed_tree = ast.parse(source_text)

        # Apply the transformation to remove docstrings
        transformer = _DocstringRemover()
        transformed_tree = transformer.visit(parsed_tree)

        # Ensure line numbers and column offsets are correct after transformation,
        # though ast.unparse typically handles this.
        ast.fix_missing_locations(transformed_tree)

        # Convert the modified AST back to source code.
        # ast.unparse is available in Python 3.9+.
        if sys.version_info >= (3, 9):
            return ast.unparse(transformed_tree)
        else:
            # For Python versions older than 3.9, ast.unparse is not available.
            # A common fallback is the 'astor' library.
            # If 'astor' is not available or desired, this limitation exists.
            try:
                import astor
                return astor.to_source(transformed_tree)
            except ImportError:
                return (
                    f"Error: For class '{cls_obj.__name__}', AST processing complete, but `ast.unparse` "
                    f"(Python 3.9+) or the 'astor' library is required to generate source code. "
                    f"Your Python version is {sys.version_info.major}.{sys.version_info.minor}."
                )

    except SyntaxError as e:
        return f"Error: Could not parse the source code for class '{cls_obj.__name__}'. Reason: {e}"
    except Exception as e:
        # Catch any other unexpected errors during AST processing or unparsing.
        return f"Error: An unexpected error occurred while processing class '{cls_obj.__name__}': {e}"

def generate_class_contents_hash(class_str:str):
    
    ode_bytes = class_str.encode('utf-8')
    #    b. Create a hash object (SHA256 is recommended)
    hasher = hashlib.sha256()
    #    c. Update the hasher with the bytes
    hasher.update(ode_bytes)
    #    d. Get the hexadecimal representation of the hash
    return hasher.hexdigest()


class _DocstringRemover(ast.NodeTransformer):
    """
    An AST transformer that removes docstrings from ClassDef, FunctionDef,
    and AsyncFunctionDef nodes.
    Docstrings are string literals that appear as the first statement in
    a definition's body.
    """

    def _process_node_with_docstring(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        """
        Helper to remove docstring from a node's body if ast.get_docstring confirms its presence.
        """
        # ast.get_docstring correctly identifies the docstring based on Python's rules.
        # 'clean=False' ensures we are just checking for existence, not processing it.
        if ast.get_docstring(node, clean=False) is not None:
            # If a docstring exists, the first element in the node's body
            # is expected to be an ast.Expr node containing the docstring.
            if node.body and isinstance(node.body[0], ast.Expr):
                node.body = node.body[1:]
            # If the body becomes empty (e.g., function with only a docstring),
            # ast.unparse or astor will typically insert 'pass' automatically.
        self.generic_visit(node)  # Continue to visit children of the node
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        # Though the request is for functions, making the remover general
        # doesn't hurt if it's reused.
        return self._process_node_with_docstring(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return self._process_node_with_docstring(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        return self._process_node_with_docstring(node)

def get_function_source_without_docstrings_or_comments(func_obj) -> str | None:
    """
    Retrieves the source code text of a Python function, excluding its
    docstring and most comments.

    Comments are typically removed as a side effect of parsing the code
    into an Abstract Syntax Tree (AST) and then unparsing it. Standard
    comments are not part of the AST structure that ast.unparse rebuilds.

    The formatting of the output code is determined by `ast.unparse`
    (Python 3.9+) or `astor` if used as a fallback.

    Args:
        func_obj: The function or method object to inspect.

    Returns:
        A string containing the source code of the function without its
        docstring and comments.
        Returns an error message string if the source cannot be retrieved,
        parsed, or if `ast.unparse` (or a fallback) is unavailable.
    """
    # inspect.isroutine checks for user-defined or built-in functions or methods.
    if not inspect.isroutine(func_obj):
        return f"Error: Expected a function or method, but got {type(func_obj).__name__}."

    try:
        source_text = inspect.getsource(func_obj)
    except (TypeError, OSError) as e:
        func_name = getattr(func_obj, '__name__', 'the provided object')
        return f"Error: Could not retrieve source for '{func_name}'. Reason: {e}"

    try:
        # Parse the source text into an Abstract Syntax Tree (AST)
        parsed_tree = ast.parse(source_text)

        # Apply the transformation to remove docstrings
        transformer = _DocstringRemover()
        transformed_tree = transformer.visit(parsed_tree)

        # Ensure line numbers and column offsets are correct after transformation.
        ast.fix_missing_locations(transformed_tree)

        # Convert the modified AST back to source code.
        # ast.unparse (Python 3.9+) and astor.to_source inherently omit
        # comments that are not part of the AST's structural representation.
        if sys.version_info >= (3, 9):
            return ast.unparse(transformed_tree)
        else:
            try:
                import astor
                # astor.to_source might add an extra newline at the end, strip it.
                return astor.to_source(transformed_tree).strip()
            except ImportError:
                func_name = getattr(func_obj, '__name__', 'the function')
                return (
                    f"Error: For '{func_name}', AST processing complete, but `ast.unparse` "
                    f"(Python 3.9+) or the 'astor' library (for Python < 3.9) is required to generate source code. "
                    f"Your Python version is {sys.version_info.major}.{sys.version_info.minor}. "
                    "Please install 'astor' (`pip install astor`) if using an older Python version."
                )

    except SyntaxError as e:
        func_name = getattr(func_obj, '__name__', 'the function')
        return f"Error: Could not parse the source code for '{func_name}'. Reason: {e}"
    except Exception as e:
        func_name = getattr(func_obj, '__name__', 'the function')
        return f"Error: An unexpected error occurred while processing '{func_name}': {e}"

class MLflowCallback:
    def __init__(self, objective_name:str,
                parameter_names,
                params_idx,
                omega_unpack_idx, 
                omega_diag_size,
                omega_diag_names,
                omega_ltri_nodiag_names,
                optimize_sigma_on_log_scale,
                optimize_omega_on_log_scale,
                init_params_for_scaling,
                total_n_params,
                opt_params_combined_params_idx,
                fixed_params_combined_params_idx,
                fixed_params,
                 use_full_omega:bool,
                 
                 
                 
                 ):
        self.iteration = 0
        self.objective_name = objective_name
        self.parameter_names = parameter_names
        self.params_idx = params_idx
        self.omega_unpack_idx = omega_unpack_idx
        self.omega_diag_size = omega_diag_size
        self.optimize_sigma_on_log_scale = optimize_sigma_on_log_scale
        self.optimize_omega_on_log_scale = optimize_omega_on_log_scale
        self.use_full_omega = use_full_omega
        self._cache_df = pd.DataFrame()
        self.omega_diag_names = omega_diag_names
        self.omega_ltri_nodiag_names = omega_ltri_nodiag_names
        self.init_params_for_scaling = init_params_for_scaling
        self.total_n_params = total_n_params
        self.opt_params_combined_params_idx = opt_params_combined_params_idx
        self.fixed_params_combined_params_idx = fixed_params_combined_params_idx
        self.fixed_params = fixed_params
        

    def __call__(self, intermediate_result:OptimizeResult):
        """
        Callback function to log metrics at each iteration.
        'xk' is the current parameter vector.
        'intermediate_result' is an OptimizeResult object (for some methods like 'trust-constr').
        """
        
        combined_params = np.zeros(self.total_n_params)
        combined_params[self.opt_params_combined_params_idx] = intermediate_result.x
        combined_params[self.fixed_params_combined_params_idx] = self.fixed_params
        self.iteration += 1
        current_fun_val = intermediate_result.fun
        mlflow.log_metric(self.objective_name,current_fun_val , step=self.iteration)
        uncentered_intermd_result = combined_params + self.init_params_for_scaling
        [mlflow.log_metric(f'param_{self.parameter_names[idx]}_value', val, step = self.iteration)
         for idx, val in enumerate(uncentered_intermd_result)]
        pop_idx = self.params_idx['pop']
        for idx in range(pop_idx[0], pop_idx[-1]):
            mlflow.log_metric(f'exp_param_{self.parameter_names[idx]}_value',
                              np.exp(uncentered_intermd_result[idx]),
                              step = self.iteration)
        
        if self.use_full_omega:
            omg, cor = self.reconstruct_omega(uncentered_intermd_result)
            self.log_vals_names(omg[0].flatten(), omg[1])
            self.log_vals_names(cor[0].flatten(), cor[1])
        else:
            if self.optimize_omega_on_log_scale:
                omega_idx = self.params_idx['omega']
                for idx in range(omega_idx[0], omega_idx[-1]):
                    mlflow.log_metric(f'exp_param_{self.parameter_names[idx]}_value',
                                    np.exp(uncentered_intermd_result[idx]),
                                    step = self.iteration)
            
        if self.optimize_sigma_on_log_scale:
            sigma_idx = self.params_idx['sigma']
            for idx in range(sigma_idx[0], sigma_idx[-1]):
                mlflow.log_metric(f'exp_param_{self.parameter_names[idx]}_value',
                                np.exp(uncentered_intermd_result[idx]),
                                step = self.iteration)
        
        
    def reconstruct_omega(self, uncentered_intermd_result):
        update_idx = self.omega_unpack_idx
        omegas_idx = self.params_idx['omega']
        omegas = uncentered_intermd_result[omegas_idx[0]:omegas_idx[-1]]
        omegas_lchol = np.zeros((self.omega_diag_size, self.omega_diag_size), dtype = np.float64)
        for idx, np_idx in enumerate(update_idx):
            omegas_lchol[np_idx] = omegas[idx]
        if self.optimize_omega_on_log_scale:
            omegas_diag = np.diag(omegas_lchol)
            omegas_diag = np.exp(omegas_diag)
            np.fill_diagonal(omegas_lchol, omegas_diag)
        omegas2 = omegas_lchol @ omegas_lchol.T
        omegas1_diag = np.diag(omegas2)
        omegas1_diag = np.sqrt(omegas1_diag)
        sd_matrix = np.outer(omegas1_diag, omegas1_diag)
        corr_matrix = np.copy(omegas2 / (sd_matrix + 1e-9))
        corr_log_idx = np.tril_indices_from(corr_matrix, k=-1)
        #omegas_log_idx = np.diag_indices_from(corr_matrix)
        np.fill_diagonal(omegas2, omegas1_diag)

        log_omegas = omegas1_diag

        
        log_corr = corr_matrix[corr_log_idx]

        
        return (log_omegas, self.omega_diag_names), (log_corr, self.omega_ltri_nodiag_names), 
    
    def log_vals_names(self, vals, names):
        iter_obj = zip(vals, names)
        for val, name in iter_obj:
            mlflow.log_metric(name, val, step=self.iteration)


class JaxMLflowCallbackFactory:
    _registered_class = None

    @classmethod
    def get_class(cls):
        """
        Defines, registers, and returns the JaxMLflowCallback class.
        Ensures registration only happens once by caching the class type.
        """
        # Return the cached class if it's already been created
        if cls._registered_class:
            return cls._registered_class

        # --- Define the class dynamically inside the factory method ---
        class _JaxMLflowCallback:
            def __init__(self, objective_fn, **kwargs):
                """
                Initializes the callback with the objective function and any
                static data it needs to run. All other keyword arguments
                from your original class are accepted via kwargs.
                """
                self.objective_fn = objective_fn
                self.iteration = 0
                # Store all other necessary parameters
                self.__dict__.update(kwargs)

            def __call__(self, params):
                """
                Callback for jaxopt. `params` is the JAX array of current optimized parameters.
                """
                self.iteration += 1

                # CHANGED: Reconstruct the full parameter vector using JAX operations
                combined_params = jnp.zeros(self.total_n_params)
                combined_params = combined_params.at[self.opt_params_combined_params_idx].set(params)
                combined_params = combined_params.at[self.fixed_params_combined_params_idx].set(self.fixed_params)
                
                # CHANGED: Recalculate the objective value (loss)
                current_fun_val = self.objective_fn(params,)# self.static_data)
                
                # mlflow.log_metric can handle JAX arrays, but converting to float is safest
                mlflow.log_metric(self.objective_name, float(current_fun_val), step=self.iteration)
                
                # Use JAX functions for transformations
                uncentered_intermd_result = combined_params + self.init_params_for_scaling
                
                # The logging logic can largely stay the same, but using jnp for math
                # We can convert to NumPy for iteration as it's cleaner
                for idx, val in enumerate(np.asarray(uncentered_intermd_result)):
                    mlflow.log_metric(f'param_{self.parameter_names[idx]}_value', val, step=self.iteration)

                pop_idx = self.params_idx['pop']
                for idx in range(pop_idx[0], pop_idx[-1]):
                    # Use jnp.exp
                    exp_val = jnp.exp(uncentered_intermd_result[idx])
                    mlflow.log_metric(f'exp_param_{self.parameter_names[idx]}_value', float(exp_val), step=self.iteration)

                if self.use_full_omega:
                    # This reconstruction function must now use jnp
                    omg, cor = self.reconstruct_omega_jax(uncentered_intermd_result)
                    self.log_vals_names(omg[0].flatten(), omg[1])
                    self.log_vals_names(cor[0].flatten(), cor[1])
                else:
                    if self.optimize_omega_on_log_scale:
                        omega_idx = self.params_idx['omega']
                        for idx in range(omega_idx[0], omega_idx[-1]):
                            mlflow.log_metric(f'exp_param_{self.parameter_names[idx]}_value',
                                            np.exp(uncentered_intermd_result[idx]),
                                            step = self.iteration)
                    
                if self.optimize_sigma_on_log_scale:
                    sigma_idx = self.params_idx['sigma']
                    for idx in range(sigma_idx[0], sigma_idx[-1]):
                        mlflow.log_metric(f'exp_param_{self.parameter_names[idx]}_value',
                                        np.exp(uncentered_intermd_result[idx]),
                                        step = self.iteration)
            
            def reconstruct_omega_jax(self, uncentered_intermd_result):
                #update_idx = self.omega_unpack_idx # Assuming this is pre-computed
                rows, cols = self.omega_unpack_idx
                omegas_idx = self.params_idx['omega']
                omegas = uncentered_intermd_result[omegas_idx[0]:omegas_idx[-1]]
                
                omegas_lchol = jnp.zeros((self.omega_diag_size, self.omega_diag_size))
                # Assuming update_idx is a tuple of (row_indices, col_indices)
                omegas_lchol = omegas_lchol.at[rows, cols ].set(omegas)
                
                if self.optimize_omega_on_log_scale:
                    omegas_diag = jnp.exp(jnp.diag(omegas_lchol))
                    omegas_lchol = omegas_lchol.at[jnp.diag_indices_from(omegas_lchol)].set(omegas_diag)
                    
                omegas2 = omegas_lchol @ omegas_lchol.T
                omegas1_diag = jnp.sqrt(jnp.diag(omegas2))
                sd_matrix = jnp.outer(omegas1_diag, omegas1_diag)
                corr_matrix = omegas2 / (sd_matrix + 1e-9)
                corr_log_idx = jnp.tril_indices_from(corr_matrix, k=-1)
                
                log_omegas = omegas1_diag
                log_corr = corr_matrix[corr_log_idx]
                
                return (log_omegas, self.omega_diag_names), (log_corr, self.omega_ltri_nodiag_names)
    
            def log_vals_names(self, vals, names):
                # Convert JAX array to NumPy for easy iteration
                for val, name in zip(np.asarray(vals), names):
                    mlflow.log_metric(name, val, step=self.iteration)

            # --- Pytree registration methods ---
            def _tree_flatten(self):
                # All attributes are considered static metadata.
                return (), self.__dict__

            @classmethod
            def _tree_unflatten(cls_local, aux_data, children):
                # Recreate the instance from the static metadata.
                instance = cls_local.__new__(cls_local)
                instance.__dict__.update(aux_data)
                return instance

        # --- Register the newly defined class with JAX ---
        jax.tree_util.register_pytree_node(
            _JaxMLflowCallback,
            _JaxMLflowCallback._tree_flatten,
            _JaxMLflowCallback._tree_unflatten,
        )

        # Cache the registered class and return it
        cls._registered_class = _JaxMLflowCallback
        return _JaxMLflowCallback

class JaxMLflowCallback:
    # CHANGED: Added objective_fn and any required static_data to the constructor
    def __init__(self, objective_fn,#, static_data,
                 objective_name:str,
                parameter_names,
                params_idx,
                omega_unpack_idx, 
                omega_diag_size,
                omega_diag_names,
                omega_ltri_nodiag_names,
                optimize_sigma_on_log_scale,
                optimize_omega_on_log_scale,
                init_params_for_scaling,
                total_n_params,
                opt_params_combined_params_idx,
                fixed_params_combined_params_idx,
                fixed_params,
                 use_full_omega:bool,
                ):
        
        self.objective_fn = objective_fn
        #self.static_data = static_data
        self.iteration = 0
        
        # It's good practice to ensure non-JAX arrays are standard NumPy arrays
        # and that parameters used in JAX functions are JAX arrays.
        self.fixed_params = jnp.asarray(fixed_params)
        
        self.objective_name = objective_name
        self.parameter_names = parameter_names
        self.params_idx = params_idx
        self.omega_unpack_idx = omega_unpack_idx
        self.omega_diag_size = omega_diag_size
        self.optimize_sigma_on_log_scale = optimize_sigma_on_log_scale
        self.optimize_omega_on_log_scale = optimize_omega_on_log_scale
        self.use_full_omega = use_full_omega
        self._cache_df = pd.DataFrame()
        self.omega_diag_names = omega_diag_names
        self.omega_ltri_nodiag_names = omega_ltri_nodiag_names
        self.init_params_for_scaling = init_params_for_scaling
        self.total_n_params = total_n_params
        self.opt_params_combined_params_idx = opt_params_combined_params_idx
        self.fixed_params_combined_params_idx = fixed_params_combined_params_idx


    # CHANGED: The signature now takes `params` directly
    def __call__(self, params):
        """
        Callback for jaxopt. `params` is the JAX array of current optimized parameters.
        """
        self.iteration += 1

        # CHANGED: Reconstruct the full parameter vector using JAX operations
        combined_params = jnp.zeros(self.total_n_params)
        combined_params = combined_params.at[self.opt_params_combined_params_idx].set(params)
        combined_params = combined_params.at[self.fixed_params_combined_params_idx].set(self.fixed_params)
        
        # CHANGED: Recalculate the objective value (loss)
        current_fun_val = self.objective_fn(params,)# self.static_data)
        
        # mlflow.log_metric can handle JAX arrays, but converting to float is safest
        mlflow.log_metric("objective_value", float(current_fun_val), step=self.iteration)
        
        # Use JAX functions for transformations
        uncentered_intermd_result = combined_params + self.init_params_for_scaling
        
        # The logging logic can largely stay the same, but using jnp for math
        # We can convert to NumPy for iteration as it's cleaner
        for idx, val in enumerate(np.asarray(uncentered_intermd_result)):
            mlflow.log_metric(f'param_{self.parameter_names[idx]}_value', val, step=self.iteration)

        pop_idx = self.params_idx['pop']
        for idx in range(pop_idx[0], pop_idx[-1]):
            # Use jnp.exp
            exp_val = jnp.exp(uncentered_intermd_result[idx])
            mlflow.log_metric(f'exp_param_{self.parameter_names[idx]}_value', float(exp_val), step=self.iteration)

        if self.use_full_omega:
            # This reconstruction function must now use jnp
            omg, cor = self.reconstruct_omega_jax(uncentered_intermd_result)
            self.log_vals_names(omg[0].flatten(), omg[1])
            self.log_vals_names(cor[0].flatten(), cor[1])
        # ... (rest of your logic for sigma/omega follows the same pattern) ...

    # CHANGED: This entire function is rewritten to use jnp
    def reconstruct_omega_jax(self, uncentered_intermd_result):
        update_idx = self.omega_unpack_idx # Assuming this is pre-computed
        omegas_idx = self.params_idx['omega']
        omegas = uncentered_intermd_result[omegas_idx[0]:omegas_idx[-1]]
        
        omegas_lchol = jnp.zeros((self.omega_diag_size, self.omega_diag_size))
        # Assuming update_idx is a tuple of (row_indices, col_indices)
        omegas_lchol = omegas_lchol.at[update_idx].set(omegas)
        
        if self.optimize_omega_on_log_scale:
            omegas_diag = jnp.exp(jnp.diag(omegas_lchol))
            omegas_lchol = omegas_lchol.at[jnp.diag_indices_from(omegas_lchol)].set(omegas_diag)
            
        omegas2 = omegas_lchol @ omegas_lchol.T
        omegas1_diag = jnp.sqrt(jnp.diag(omegas2))
        sd_matrix = jnp.outer(omegas1_diag, omegas1_diag)
        corr_matrix = omegas2 / (sd_matrix + 1e-9)
        corr_log_idx = jnp.tril_indices_from(corr_matrix, k=-1)
        
        log_omegas = omegas1_diag
        log_corr = corr_matrix[corr_log_idx]
        
        return (log_omegas, self.omega_diag_names), (log_corr, self.omega_ltri_nodiag_names)
    
    def log_vals_names(self, vals, names):
        # Convert JAX array to NumPy for easy iteration
        for val, name in zip(np.asarray(vals), names):
            mlflow.log_metric(name, val, step=self.iteration)