import jax, jax.numpy as jnp


def newton_rootsolve(
    func,
    guesses,
    params=[],
    jacfunc=None,
    tolfunc=None,
    rtol=1e-6,
    max_iter=100,
    careful_steps=1,
):
    """
    Solve the system f(X,p) = 0 for X, where both f and X can be vectors of arbitrary length and p is a set of fixed
    parameters passed to f. Broadcasts and parallelizes over an arbitrary number of initial guesses and parameter
    choices.

    Parameters
    ----------
    func: callable
        A JAX function of signature f(X,params) that implements the function we wish to rootfind, where X and params
        are arrays of shape (n,) and (n_p,) for dimension n and parameter number n_p. In general can return an array of
        shape (m,)
    guesses: array_like
        Shape (n,) or (N,n) array_like where N is the number of guesses + corresponding parameter choices
    params: array_like
        Shape (n,) or (N,n_p) array_like where N is the number of guesses + corresponding parameter choices
    jacfunc: callable, optional
        Function with the same signature as f that returns the Jacobian of f - will be computed with autodiff from f if
        not specified.
    rtol: float, optional
        Relative tolerance - iteration will terminate if relative change in all quantities is less than this value.
    atol: float, optional
        Absolute tolerance: iteration will terminate if the value computed by tolfunc goes below this value.
    careful_steps: int, optional
        Number of "careful" initial steps to take, gradually ramping up the step size in the Newton iteration

    Returns
    -------
    X: array_like
        Shape (N,n) array of solutions
    """
    guesses = jnp.array(guesses)
    params = jnp.array(params)
    if len(guesses.shape) < 2:
        guesses = jnp.atleast_2d(guesses).T
    if len(params.shape) < 2:
        params = jnp.atleast_2d(params).T

    if jacfunc is None:
        jac = jax.jacfwd(func)

    if tolfunc is None:

        def tolfunc(X, *params):
            return X

    def solve(guess, params):
        """Function to be called in parallel that solves the root problem for one guess and set of parameters"""

        def iter_condition(arg):
            """Iteration condition for the while loop: check if we are within desired tolerance."""
            X, dx, num_iter = arg
            fac = jnp.min(jnp.array([(num_iter + 1.0) / careful_steps, 1.0]))
            tol2, tol1 = tolfunc(X, *params), tolfunc(X - dx, *params)
            tolcheck = jnp.any(jnp.abs(tol1 - tol2) > rtol * jnp.abs(tol1) * fac)
            return jnp.any(jnp.abs(dx) > fac * rtol * jnp.abs(X)) & (num_iter < max_iter) & tolcheck

        def X_new(arg):
            """Returns the next Newton iterate and the difference from previous guess."""
            X, _, num_iter = arg
            fac = jnp.min(jnp.array([(num_iter + 1.0) / careful_steps, 1.0]))
            dx = -jnp.linalg.solve(jac(X, *params), func(X, *params)) * fac
            # need to reject steps that increase the residual...
            return X + dx, dx, num_iter + 1

        init_val = guess, 100 * guess, 0
        X, _, num_iter = jax.lax.while_loop(iter_condition, X_new, init_val)

        return jnp.where(num_iter < max_iter, X, X * jnp.nan)

    X = jax.vmap(solve)(guesses, params)
    return X


newton_rootsolve = jax.jit(
    newton_rootsolve, static_argnames=["func", "tolfunc", "jacfunc", "max_iter", "careful_steps"]
)
