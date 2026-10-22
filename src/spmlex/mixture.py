"""Rust mixture solves with local implicit simplex derivatives."""

from functools import partial

import jax
import jax.numpy as jnp
from mixsqpx import MixSolveResult, mixsolve

from .options import MixtureSolverOptions


def regularized_log_components(log_components: jax.Array, offset: float) -> jax.Array:
    if offset == 0:
        return log_components
    shift = jnp.max(log_components, axis=1, keepdims=True)
    return shift + jnp.logaddexp(log_components - shift, jnp.log(offset))


def normalized_weights(weights: jax.Array) -> jax.Array:
    # The common scale cancels exactly; its derivative is unnecessary.
    scale = jax.lax.stop_gradient(jnp.max(weights))
    scaled = weights / scale
    return scaled / jnp.sum(scaled)


def rust_solve(
    log_components, observation_weights, offset, options: MixtureSolverOptions
) -> MixSolveResult:
    return mixsolve(
        log_components,
        observation_weights=observation_weights,
        density_offset=offset,
        num_iter_em=options.num_iter_em,
        tol_svd=options.tol_svd,
        control_sqp=options.control_sqp,
        convtol_active_set=options.convtol_active_set,
        zero_threshold_solution=options.zero_threshold_solution,
        zero_threshold_searchdir=options.zero_threshold_searchdir,
        suffdecr_linesearch=options.suffdecr_linesearch,
        stepsizereduce=options.stepsizereduce,
        minstepsize=options.minstepsize,
        identity_contrib_increase=options.identity_contrib_increase,
        maxiter_sqp=options.maxiter_sqp,
        maxiter_activeset=options.maxiter_activeset,
        return_info=True,
    )


def _reject_solution(probabilities, status, active, inactive, simplex):
    raise RuntimeError(
        f"Rust mixture solve failed certification: status={int(status)}, "
        f"active={float(active):.3g}, inactive={float(inactive):.3g}, simplex={float(simplex):.3g}."
    )


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def solve_mixture_weights(log_components, observation_weights, offset, options):
    result = rust_solve(log_components, observation_weights, offset, options)
    probabilities, info = result.probabilities, result.info
    return jax.lax.cond(
        (info.status != 0) | ~jnp.all(jnp.isfinite(probabilities)),
        lambda: jax.pure_callback(
            _reject_solution,
            jax.ShapeDtypeStruct(probabilities.shape, probabilities.dtype),
            probabilities,
            info.status,
            info.active_residual,
            info.inactive_residual,
            info.simplex_residual,
        ),
        lambda: probabilities,
    )


def active_ratios(log_components, probabilities):
    active = probabilities > 0
    shift = jnp.max(
        jnp.where(active[None, :], log_components, -jnp.inf), axis=1, keepdims=True
    )
    scaled = jnp.exp(jnp.where(active[None, :], log_components - shift, -jnp.inf))
    return scaled / (scaled @ probabilities)[:, None]


def mixture_weight_tangent(
    log_components, weights, probabilities, logs_dot, weights_dot
):
    """Differentiate KKT equations on the current, nondegenerate active face."""
    normalized, normalized_dot = jax.jvp(normalized_weights, (weights,), (weights_dot,))
    ratios = active_ratios(log_components, probabilities)
    mixture_dot = jnp.sum(ratios * probabilities[None, :] * logs_dot, axis=1)
    score_dot = -jnp.sum(
        normalized[:, None] * ratios * (logs_dot - mixture_dot[:, None])
        + normalized_dot[:, None] * ratios,
        axis=0,
    )
    mass_hessian = ratios.T @ (normalized[:, None] * ratios)
    active = (probabilities > 0).astype(probabilities.dtype)
    active_hessian = active[:, None] * mass_hessian * active[None, :] + jnp.diag(
        1 - active
    )
    kkt = jnp.block(
        [
            [active_hessian, active[:, None]],
            [active[None, :], jnp.zeros((1, 1), dtype=probabilities.dtype)],
        ]
    )
    rhs = jnp.concatenate(
        [-active * score_dot, jnp.zeros(1, dtype=probabilities.dtype)]
    )
    return jnp.linalg.solve(kkt, rhs)[:-1]


@solve_mixture_weights.defjvp
def _solve_mixture_weights_jvp(offset, options, primals, tangents):
    log_components, weights = primals
    logs_dot, weights_dot = tangents
    probabilities = solve_mixture_weights(log_components, weights, offset, options)
    regularized, regularized_dot = jax.jvp(
        lambda logs: regularized_log_components(logs, offset),
        (log_components,),
        (logs_dot,),
    )
    return probabilities, mixture_weight_tangent(
        regularized, weights, probabilities, regularized_dot, weights_dot
    )


def log_mixture(log_components, probabilities):
    active = probabilities > 0
    log_mass = jnp.where(active, jnp.log(jnp.where(active, probabilities, 1)), -jnp.inf)
    return jax.nn.logsumexp(log_components + log_mass[None, :], axis=1)


def mixture_value(log_components, weights, probabilities):
    """Ordinary chain rule, including sensitivity through fitted masses."""
    return -jnp.dot(
        normalized_weights(weights), log_mixture(log_components, probabilities)
    )


@jax.custom_jvp
def envelope_value(log_components, weights, probabilities):
    return mixture_value(log_components, weights, probabilities)


@envelope_value.defjvp
def _envelope_value_jvp(primals, tangents):
    logs, weights, probabilities = primals
    logs_dot, weights_dot, _ = tangents
    normalized, normalized_dot = jax.jvp(normalized_weights, (weights,), (weights_dot,))
    value = envelope_value(*primals)
    responsibilities = active_ratios(logs, probabilities) * probabilities[None, :]
    tangent = -jnp.sum(normalized[:, None] * responsibilities * logs_dot)
    tangent -= jnp.dot(normalized_dot, log_mixture(logs, probabilities))
    return value, tangent
