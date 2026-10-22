from functools import partial

import jax
import jax.numpy as jnp
from pytest import approx

from spmlex import profile_ll, profile_ll_diagnostics


def test_profile_ll_forward(discrete, linear_factor, simulated_data, supp_grid):

    ll, probs = profile_ll(linear_factor, simulated_data, supp_grid)

    mean_est = (supp_grid * probs.flatten()).sum()
    mean_dgp = (discrete.supp * discrete.weights).sum()

    assert jnp.isscalar(ll), "Log-likelihood (ll) is not a scalar."
    assert probs.shape[0] == supp_grid.shape[0], (
        "Shape of probs does not match the shape of supp."
    )
    assert mean_est == approx(mean_dgp, abs=1e-1), (
        "Estimated distribution doesn't not fit mean well"
    )


def test_profile_ll_grad(linear_factor, simulated_data, supp_grid):

    @jax.jit
    @partial(jax.value_and_grad, has_aux=True)
    def val_and_grad(model):
        return profile_ll(model, simulated_data, supp_grid)

    (ll, probs), g = val_and_grad(linear_factor)

    assert jnp.isscalar(ll), "Log-likelihood (ll) is not a scalar."
    assert probs.shape[0] == supp_grid.shape[0], (
        "Shape of probs does not match the shape of supp."
    )
    assert isinstance(g, type(linear_factor)), (
        "Gradient (g) is not an instance of LinearFactor."
    )
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(g))
    assert jnp.allclose(ll, profile_ll(linear_factor, simulated_data, supp_grid)[0])


def test_profile_ll_diagnostics_returns_timing_breakdown(
    linear_factor, simulated_data, supp_grid
):
    diagnostics = profile_ll_diagnostics(linear_factor, simulated_data, supp_grid)
    ll, probs = profile_ll(linear_factor, simulated_data, supp_grid)

    assert diagnostics.nobs == simulated_data["outcomes"].shape[0]
    assert diagnostics.nsupp == supp_grid.shape[0]
    assert diagnostics.ngroups == 1
    assert diagnostics.converged
    assert diagnostics.options.differentiation == "implicit"
    assert diagnostics.log_components_seconds >= 0.0
    assert diagnostics.solve_seconds >= 0.0
    assert diagnostics.value_seconds >= 0.0
    assert diagnostics.total_seconds >= 0.0
    assert len(diagnostics.groups) == 1
    assert diagnostics.groups[0].solve_seconds >= 0.0
    assert diagnostics.groups[0].density_offset == 1e-6 / diagnostics.nobs
    assert jnp.allclose(diagnostics.objective, ll)
    assert jnp.allclose(diagnostics.probabilities, probs)
    assert diagnostics.gradient is not None
    assert diagnostics.gradient_seconds >= 0.0
    assert diagnostics.gradient_objective_error < 1e-12
