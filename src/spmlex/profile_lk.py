"""Grouped, weighted profile likelihoods with implicit or envelope gradients."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from mixsqpx import MixSolveInfo

from .mixture import (
    envelope_value,
    mixture_value,
    normalized_weights,
    regularized_log_components,
    rust_solve,
    solve_mixture_weights,
)
from .options import ProfileOptions


class ModelWithLCLK(Protocol):
    def lclk(self, data: Any, supp: jax.Array) -> jax.Array: ...


@dataclass(frozen=True)
class GroupDiagnostics:
    group_index: int
    nobs: int
    density_offset: float
    aggregation_weight: float
    objective: float
    solver_info: MixSolveInfo
    solve_seconds: float
    value_seconds: float

    @property
    def converged(self) -> bool:
        return int(self.solver_info.status) == 0


@dataclass(frozen=True)
class ProfileDiagnostics:
    nobs: int
    nsupp: int
    ngroups: int
    options: ProfileOptions
    objective: jax.Array
    probabilities: jax.Array
    groups: tuple[GroupDiagnostics, ...]
    log_components_seconds: float
    solve_seconds: float
    value_seconds: float
    total_seconds: float
    gradient: Any | None
    gradient_seconds: float | None
    gradient_objective_error: float | None

    @property
    def converged(self) -> bool:
        return all(group.converged for group in self.groups)


def _reject_weights(weights):
    raise ValueError("Weights must be finite and nonnegative with positive total.")


@jax.custom_jvp
def _checked_weights(weights):
    valid = jnp.all(jnp.isfinite(weights) & (weights >= 0)) & jnp.any(weights > 0)
    return jax.lax.cond(
        valid,
        lambda: weights,
        lambda: jax.pure_callback(
            _reject_weights, jax.ShapeDtypeStruct(weights.shape, weights.dtype), weights
        ),
    )


@_checked_weights.defjvp
def _checked_weights_jvp(primals, tangents):
    return _checked_weights(primals[0]), tangents[0]


def _prepare_inputs(log_components, endog_dummies, observation_weights, group_weights):
    logs = jnp.asarray(log_components)
    if logs.ndim != 2 or min(logs.shape) == 0:
        raise ValueError("Log components must have nonempty shape (nobs, nsupp).")
    if logs.dtype != jnp.float64:
        raise TypeError("Profile likelihoods require float64 log components.")
    nobs = logs.shape[0]
    if endog_dummies is None:
        indices = (np.arange(nobs),)
    else:
        masks = np.asarray(endog_dummies)
        if masks.dtype != bool or masks.ndim != 2 or masks.shape[0] != nobs:
            raise ValueError(
                "endog_dummies must be a concrete boolean (nobs, ngroups) partition."
            )
        if (
            masks.shape[1] == 0
            or not np.all(masks.sum(axis=1) == 1)
            or not np.all(masks.any(axis=0))
        ):
            raise ValueError(
                "Groups must be nonempty and each observation must belong to exactly one group."
            )
        indices = tuple(np.flatnonzero(mask) for mask in masks.T)
    weights = (
        jnp.ones(nobs, dtype=logs.dtype)
        if observation_weights is None
        else jnp.asarray(observation_weights, dtype=logs.dtype)
    )
    if weights.shape != (nobs,):
        raise ValueError("observation_weights must have shape (nobs,).")
    weights = _checked_weights(weights)
    logs = jnp.where((weights[:, None] == 0) & ~jnp.isfinite(logs), 0.0, logs)
    if group_weights is None:
        normalized = normalized_weights(weights)
        aggregation = jnp.stack([jnp.sum(normalized[index]) for index in indices])
    else:
        aggregation = jnp.asarray(group_weights, dtype=logs.dtype)
        if aggregation.shape != (len(indices),):
            raise ValueError("group_weights must have shape (ngroups,).")
        aggregation = normalized_weights(_checked_weights(aggregation))
    return logs, weights, indices, aggregation


@partial(jax.jit, static_argnums=(3,))
def _group_value(logs, weights, probabilities, options):
    regularized = regularized_log_components(
        logs, options.density_offset(logs.shape[0])
    )
    if options.differentiation == "envelope":
        return envelope_value(regularized, weights, probabilities)
    return mixture_value(regularized, weights, probabilities)


def profile_mixture(
    log_components: jax.Array,
    endog_dummies: np.ndarray | None = None,
    *,
    observation_weights: jax.Array | None = None,
    group_weights: jax.Array | None = None,
    options: ProfileOptions = ProfileOptions(),
) -> tuple[jax.Array, jax.Array]:
    """Return negative mean profile likelihood and (nsupp, ngroups) masses.

    Groups are a concrete partition closed over by a JIT-compiled objective.
    Observation weights enter each Rust solve; group weights determine the
    outer aggregation and default to each group's share of observation weight.
    The inverse-sample-size offset uses the number of supplied rows per group.
    """
    logs, weights, indices, aggregation = _prepare_inputs(
        log_components, endog_dummies, observation_weights, group_weights
    )
    values, probabilities = [], []
    for index in indices:
        group_logs, group_weights_ = logs[index], weights[index]
        masses = solve_mixture_weights(
            group_logs,
            group_weights_,
            options.density_offset(len(index)),
            options.solver,
        )
        probabilities.append(masses)
        values.append(_group_value(group_logs, group_weights_, masses, options))
    return jnp.dot(aggregation, jnp.stack(values)), jnp.column_stack(probabilities)


def profile_ll(
    model: ModelWithLCLK,
    data: Any,
    supp: jax.Array,
    endog_dummies: np.ndarray | None = None,
    *,
    observation_weights: jax.Array | None = None,
    group_weights: jax.Array | None = None,
    options: ProfileOptions = ProfileOptions(),
) -> tuple[jax.Array, jax.Array]:
    """Profile model.lclk with full implicit differentiation by default."""
    return profile_mixture(
        model.lclk(data, supp),
        endog_dummies,
        observation_weights=observation_weights,
        group_weights=group_weights,
        options=options,
    )


def _timed_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = perf_counter()
    value = fn()
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value, perf_counter() - start


def _diagnose(
    log_components,
    endog_dummies,
    observation_weights,
    group_weights,
    options,
    gradient_call,
    log_seconds,
):
    start = perf_counter()
    logs, weights, indices, aggregation = _prepare_inputs(
        log_components, endog_dummies, observation_weights, group_weights
    )
    groups, values, probabilities = [], [], []
    for group_index, index in enumerate(indices):
        offset = options.density_offset(len(index))
        result, solve_seconds = _timed_call(
            lambda: rust_solve(logs[index], weights[index], offset, options.solver)
        )
        value, value_seconds = _timed_call(
            lambda: _group_value(
                logs[index], weights[index], result.probabilities, options
            )
        )
        probabilities.append(result.probabilities)
        values.append(value)
        groups.append(
            GroupDiagnostics(
                group_index=group_index,
                nobs=len(index),
                density_offset=offset,
                aggregation_weight=float(aggregation[group_index]),
                objective=float(value),
                solver_info=result.info,
                solve_seconds=solve_seconds,
                value_seconds=value_seconds,
            )
        )
    objective = jnp.dot(aggregation, jnp.stack(values))
    objective.block_until_ready()
    gradient = gradient_seconds = objective_error = None
    if gradient_call is not None and all(group.converged for group in groups):
        (gradient_value, gradient), gradient_seconds = _timed_call(gradient_call)
        objective_error = float(jnp.abs(gradient_value - objective))
    return ProfileDiagnostics(
        nobs=logs.shape[0],
        nsupp=logs.shape[1],
        ngroups=len(indices),
        options=options,
        objective=objective,
        probabilities=jnp.column_stack(probabilities),
        groups=tuple(groups),
        log_components_seconds=log_seconds,
        solve_seconds=sum(group.solve_seconds for group in groups),
        value_seconds=sum(group.value_seconds for group in groups),
        total_seconds=log_seconds + perf_counter() - start,
        gradient=gradient,
        gradient_seconds=gradient_seconds,
        gradient_objective_error=objective_error,
    )


def profile_ll_diagnostics(
    model: ModelWithLCLK,
    data: Any,
    supp: jax.Array,
    endog_dummies: np.ndarray | None = None,
    *,
    observation_weights: jax.Array | None = None,
    group_weights: jax.Array | None = None,
    options: ProfileOptions = ProfileOptions(),
    include_gradient: bool = True,
) -> ProfileDiagnostics:
    """Inspect the same criterion and report each Rust convergence certificate."""
    logs, log_seconds = _timed_call(lambda: model.lclk(data, supp))

    def criterion(point):
        return profile_ll(
            point,
            data,
            supp,
            endog_dummies,
            observation_weights=observation_weights,
            group_weights=group_weights,
            options=options,
        )[0]

    gradient_call = (
        (lambda: jax.jit(jax.value_and_grad(criterion))(model))
        if include_gradient
        else None
    )
    return _diagnose(
        logs,
        endog_dummies,
        observation_weights,
        group_weights,
        options,
        gradient_call,
        log_seconds,
    )


def profile_mixture_diagnostics(
    log_components: jax.Array,
    endog_dummies: np.ndarray | None = None,
    *,
    observation_weights: jax.Array | None = None,
    group_weights: jax.Array | None = None,
    options: ProfileOptions = ProfileOptions(),
    include_gradient: bool = False,
) -> ProfileDiagnostics:
    def criterion(logs):
        return profile_mixture(
            logs,
            endog_dummies,
            observation_weights=observation_weights,
            group_weights=group_weights,
            options=options,
        )[0]

    gradient_call = (
        (lambda: jax.jit(jax.value_and_grad(criterion))(log_components))
        if include_gradient
        else None
    )
    return _diagnose(
        log_components,
        endog_dummies,
        observation_weights,
        group_weights,
        options,
        gradient_call,
        0.0,
    )
