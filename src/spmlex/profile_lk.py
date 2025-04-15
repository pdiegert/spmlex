__all__ = ["profile_ll"]

from typing import Protocol, Any

import jax
from jax import jacfwd, custom_jvp, jit
from jax import flatten_util
import jax.numpy as np
import numpy as onp

from mixsqpx import mixsolve

class LogLikelihood(Protocol):
    def __call__(
        self,
        theta,
        data: dict[str, np.ndarray],
        supp: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray: ...


class ModelWithLCLK(Protocol):
    def lclk(
        self, data: dict[str, np.ndarray], supp: np.ndarray, **kwargs: Any
    ) -> np.ndarray: ...


@jit
def negloglik_numeric(probs: np.ndarray, lclk_: np.ndarray) -> np.ndarray:
    ll_ = jax.nn.logsumexp(lclk_ + np.log(probs)[None, :], axis=1).mean()
    return -ll_


@jit
def ll_jac(probs, clk, clk_jac, ll_jac_probs, ll_hess_probs):
    lambd = -(ll_jac_probs + 1)

    nobs_j, nsupp = clk.shape

    m = np.vstack(
        [
            np.hstack([ll_hess_probs, np.eye(nsupp)]),
            np.hstack([np.diag(lambd), np.diag(probs)]),
        ]
    )

    wgts = 1 / (clk @ probs)  # (obs, )

    # (obs, supp, supp)
    gg = (
        np.eye(nsupp)[None, :, :]
        - (probs[None, None, :] * clk[:, :, None] * wgts[:, None, None])
    ) * wgts[:, None, None]

    b = jax.tree_util.tree_map(
        lambda clk_jac_g: (
            -np.einsum("ikl, il... -> ik...", gg, clk_jac_g).mean(axis=0)
        ),
        clk_jac,
    )

    b, _ = jax.tree_util.tree_flatten(b)
    b = [b_.reshape((nsupp, -1)) for b_ in b]
    b = np.hstack(b)
    _, nparam = b.shape
    b = np.vstack([b, np.zeros((nsupp, nparam))])

    jac_p = np.linalg.solve(-m, b)[:nsupp]

    clk_jac, _ = jax.tree_util.tree_flatten(clk_jac)
    clk_jac = [clk_jac_g.reshape((nobs_j, nsupp, -1)) for clk_jac_g in clk_jac]
    clk_jac = np.concatenate(clk_jac, axis=2)

    jac_ll = -wgts[:, None, None] * (
        clk_jac * probs[None, :, None] + jac_p[None, :, :] * clk[:, :, None]
    )
    jac_ll = jac_ll.sum(axis=1).mean(axis=0)

    return jac_ll


@jit
@jacfwd
def clk_jac(model: ModelWithLCLK, data: dict[str, np.ndarray], supp: np.ndarray):
    return np.exp(model.lclk(data, supp))


@custom_jvp
def profile_ll(
    model,
    data,
    supp,
    endog_dummies: onp.ndarray | None = None,
    weights: onp.ndarray | None = None,
    cpp: bool = True,
):

    endog_dummies, weights = _default_groups(data, endog_dummies, weights)

    lclk_ = model.lclk(data, supp)
    probs = [mixsolve(lclk_[i, :])[0] for i in endog_dummies.T]
    probs_stacked = np.column_stack(probs)

    def error_case(x):
        return jax.lax.stop_gradient(np.inf), jax.lax.stop_gradient(
            np.zeros_like(probs_stacked)
        )

    def success_case(x):
        neg_lls_ = np.array(
            [
                weight_i * negloglik_numeric(probs_i, lclk_[i, :])
                for weight_i, probs_i, i in zip(weights, probs, endog_dummies.T)
            ]
        )
        return neg_lls_.sum(), probs_stacked

    has_nans = np.any(np.isnan(probs_stacked))
    return jax.lax.cond(has_nans, error_case, success_case, None)


@profile_ll.defjvp
def profile_ll_jac(primals, tangents):
    model, data, supp, endog_dummies, weights, cpp = primals
    model_dot, *_ = tangents
    model_dot, _ = flatten_util.ravel_pytree(model_dot)

    endog_dummies, weights = _default_groups(data, endog_dummies, weights)

    lclk_ = model.lclk(data, supp)
    clk_ = np.exp(lclk_)
    clk_jac_ = clk_jac(model, data, supp)

    nobs, _ = lclk_.shape

    def _jvp_group(i):
        probs_i, ll_jac_probs_i, ll_hess_probs_i, *_ = mixsolve(lclk_[i, :])
        clk_jac_i = jax.tree_util.tree_map(lambda theta_g: theta_g[i], clk_jac_)
        theta_jac_i = ll_jac(
            probs_i, clk_[i], clk_jac_i, ll_jac_probs_i, ll_hess_probs_i
        )

        primal_out_i = (i.sum() / nobs) * negloglik_numeric(probs_i, lclk_[i, :])
        tangent_out_i = (i.sum() / nobs) * theta_jac_i

        return primal_out_i, tangent_out_i, probs_i

    jvp_groups = [_jvp_group(i) for i in endog_dummies.T]

    primal_out = sum([jvp_group[0] for jvp_group in jvp_groups])
    tangent_out = sum([jvp_group[1] for jvp_group in jvp_groups]) @ model_dot
    probs_out = np.column_stack([jvp_group[2] for jvp_group in jvp_groups])

    return (primal_out, probs_out), (tangent_out, probs_out)


def _default_groups(data, endog_dummies, weights):
    """
    Create a default group structure for the endog_dummies.
    """
    if endog_dummies is None:
        nobs = _get_nobs(data)
        endog_dummies = onp.ones((nobs, 1)).astype(onp.bool_)
        weights = onp.ones((nobs, 1)) / nobs

    return endog_dummies, weights


def _get_nobs(data):
    vals = list(data.values())[0]
    if isinstance(vals, dict):
        return _get_nobs(vals)
    else:
        return vals.shape[0]

