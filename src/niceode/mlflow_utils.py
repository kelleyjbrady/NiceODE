import inspect
import ast
import sys

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
