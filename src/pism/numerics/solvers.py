import jax, jax.numpy as jnp
import sympy as sp


def jax_solver(T):
    func = sp.lambdify(unknowns + knowns, reduced_network, modules="jax")
    jacfunc = sp.lambdify(unknowns + knowns, jac, modules="jax")

    @jax.jit
    def solve_abundances(T):
        def f_numerical(X, knownvals):
            return jnp.array(func(X[0], X[1], X[2], knownvals[0], knownvals[1], knownvals[2]))

        def jac_numerical(X, knownvals):
            return jnp.array(jacfunc(X[0], X[1], X[2], knownvals[0], knownvals[1], knownvals[2]))

        def solve_for_T(T, _):
            x0 = jnp.array([0.5, 1e-5, 1e-5])
            knownvals = jnp.array([T, 1, 0.24])

            X = x0

            # can use
            for _ in range(100):
                J = jac_numerical(X, knownvals)
                f = f_numerical(X, knownvals)
                dx = -jnp.linalg.solve(J, f)  # don't actually have to invert!
                X += dx

            return X

        X = jax.vmap(solve_for_T)(Tval, None)
        return X

    return solve_abundances(T)
