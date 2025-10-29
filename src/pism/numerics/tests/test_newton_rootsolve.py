import numpy as np
import jax, jax.numpy as jnp
from ..solvers import newton_rootsolve


def test_newton_rootsolve(N=10**5):
    """Test: solve the system x^p = a for various choices of p and a and check solution

    NOTE: this newton iteration does not converge in general, so we will not catch all possible bugs that might return
    nonfinite values...
    """
    import numpy as np

    p = 0.1 + np.random.rand(N) * 10
    a = 0.1 + np.random.rand(N)
    params = jnp.c_[p, a]
    guess = jnp.ones(N)

    exact = np.atleast_2d(a ** (1.0 / p)).T

    def func(x, *params):
        return x ** params[0] - params[1]

    func = jax.jit(func)

    sol = newton_rootsolve(func, guess, params)
    converged = jnp.all(jnp.isfinite(sol), axis=1)
    assert converged.sum() > 0.9 * N
    assert jnp.all(jnp.isclose(sol[converged], exact[converged], rtol=1e-5, atol=0))
