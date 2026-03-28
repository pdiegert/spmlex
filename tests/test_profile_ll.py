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
    assert probs.shape[0] == supp_grid.shape[0], "Shape of probs does not match the shape of supp."
    assert mean_est == approx(mean_dgp, abs=1e-1), "Estimated distribution doesn't not fit mean well"


def test_profile_ll_grad(linear_factor, simulated_data, supp_grid):

    @jax.jit
    @partial(jax.value_and_grad, has_aux=True)
    def val_and_grad(model):
        return profile_ll(model, simulated_data, supp_grid)

    (ll, probs), g = val_and_grad(linear_factor)

    assert jnp.isscalar(ll), "Log-likelihood (ll) is not a scalar."
    assert probs.shape[0] == supp_grid.shape[0], "Shape of probs does not match the shape of supp."
    assert isinstance(g, type(linear_factor)), "Gradient (g) is not an instance of LinearFactor."


def test_profile_ll_diagnostics_returns_timing_breakdown(linear_factor, simulated_data, supp_grid):
    diagnostics = profile_ll_diagnostics(linear_factor, simulated_data, supp_grid)
    ll, probs = profile_ll(linear_factor, simulated_data, supp_grid)

    assert diagnostics["nobs"] == simulated_data["outcomes"].shape[0]
    assert diagnostics["nsupp"] == supp_grid.shape[0]
    assert diagnostics["ngroups"] == 1

    forward = diagnostics["forward"]
    assert forward["lclk_seconds"] >= 0.0
    assert forward["mixsolve_seconds"] >= 0.0
    assert forward["negloglik_seconds"] >= 0.0
    assert forward["total_seconds"] >= 0.0
    assert len(forward["groups"]) == 1
    assert forward["groups"][0]["mixsolve_seconds"] >= 0.0
    assert jnp.allclose(forward["objective"], ll)
    assert jnp.allclose(forward["probs"], probs)

    gradient = diagnostics["gradient"]
    assert gradient is not None
    assert gradient["lclk_seconds"] >= 0.0
    assert gradient["clk_jac_seconds"] >= 0.0
    assert gradient["mixsolve_seconds"] >= 0.0
    assert gradient["ll_jac_seconds"] >= 0.0
    assert gradient["total_seconds"] >= 0.0
    assert len(gradient["groups"]) == 1
