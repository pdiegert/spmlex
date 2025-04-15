from pytest import fixture

import jax
import jax.numpy as jnp

import equinox as eqx

class Discrete(eqx.Module):
    supp: jnp.ndarray
    weights: jnp.ndarray

    def simulate(self, *, rng, n: int):

        component_indices = jax.random.choice(
            rng, jnp.arange(len(self.supp)), shape=(n,), p=self.weights
        )

        samples = self.supp[component_indices]

        return {"latent": samples}


class LinearFactor(eqx.Module):
    coef: jnp.ndarray
    log_std_e: jnp.ndarray

    def simulate(self, data, *, rng):
        latent = data["latent"]
        nobs, *_ = latent.shape
        nperiod = self.log_std_e.shape[0]

        coef = jnp.concat([jnp.array([1.0]), self.coef]) # normalize the first coefficient to 1

        # Compute the outcomes
        e = jax.random.normal(rng, shape=(nobs, nperiod)) * jnp.exp(self.log_std_e)
        outcomes = latent[:, None] * coef[None, :] + e

        return {"outcomes": outcomes}

    def lclk(self, data, supp):
        outcomes = data["outcomes"]  # (obs, period)

        coef = jnp.concat([jnp.array([1.0]), self.coef]) # normalize the first coefficient to 1

        deviation = (
            outcomes[:, :, None] - supp[None, None, :] * coef[None, :, None]
        )

        lclk_ = -0.5 * (
            jnp.log(2 * jnp.pi)
            + 2 * self.log_std_e.sum()
            + ((deviation / jnp.exp(self.log_std_e)[None, :, None]) ** 2).sum(axis=1)
        )

        return lclk_

@fixture
def discrete():
    """
    Fixture to create a Discrete instance with default parameters.
    """
    supp = jnp.array([-3.0, 0., 5.0])
    weights = jnp.array([0.3, 0.5, 0.2])

    return Discrete(supp=supp, weights=weights)


@fixture
def linear_factor():
    """
    Fixture to create a LinearFactor instance with default parameters.
    """
    coef = jnp.array([1.5, 0.3])
    log_std_e = jnp.array([0.0, 0.2, -0.3])

    return LinearFactor(coef=coef, log_std_e=log_std_e)


@fixture
def simulated_data(discrete: Discrete, linear_factor):
    """
    Fixture to create simulated data from DGP
    """

    # 1. Define mixture model parameters
    nobs = 4000

    # 2. Generate data from the mixture
    key = jax.random.PRNGKey(40)
    
    key, subkey = jax.random.split(key)
    data = discrete.simulate(rng = subkey, n = nobs)
    
    key, subkey = jax.random.split(key)
    data = linear_factor.simulate(data, rng = subkey)

    return data

@fixture
def supp_grid():
    return jnp.linspace(-6.0, 6.0, 75)