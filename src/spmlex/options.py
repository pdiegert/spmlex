"""Static, hashable controls for profiled mixture likelihoods."""

from dataclasses import dataclass, field
import math
from typing import Literal


@dataclass(frozen=True)
class MixtureSolverOptions:
    num_iter_em: int = 10
    tol_svd: float = 1e-10
    control_sqp: float = 1e-9
    convtol_active_set: float = 1e-12
    zero_threshold_solution: float = 1e-12
    zero_threshold_searchdir: float = 1e-14
    suffdecr_linesearch: float = 1e-2
    stepsizereduce: float = 0.75
    minstepsize: float = 1e-12
    identity_contrib_increase: float = 10.0
    maxiter_sqp: int = 1000
    maxiter_activeset: int = 100

    def __post_init__(self) -> None:
        for value in (
            self.tol_svd,
            self.zero_threshold_solution,
            self.zero_threshold_searchdir,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "SVD and zero thresholds must be finite and nonnegative."
                )
        for value in (self.control_sqp, self.convtol_active_set):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Convergence tolerances must be finite and positive.")
        if not 0 < self.minstepsize <= 1:
            raise ValueError("minstepsize must lie in (0, 1].")
        if not 0 < self.stepsizereduce < 1 or not 0 < self.suffdecr_linesearch < 1:
            raise ValueError(
                "Line search reduction and sufficient decrease must lie in (0, 1)."
            )
        if (
            not math.isfinite(self.identity_contrib_increase)
            or self.identity_contrib_increase <= 1
        ):
            raise ValueError(
                "identity_contrib_increase must be finite and greater than one."
            )
        for value in (self.num_iter_em, self.maxiter_sqp, self.maxiter_activeset):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("Iteration budgets must be nonnegative integers.")
        if self.maxiter_activeset == 0:
            raise ValueError("maxiter_activeset must be positive.")


@dataclass(frozen=True)
class ProfileOptions:
    differentiation: Literal["implicit", "envelope"] = "implicit"
    density_offset_scale: float = 1e-6
    density_offset_rule: Literal["inverse_sample_size", "constant"] = (
        "inverse_sample_size"
    )
    solver: MixtureSolverOptions = field(default_factory=MixtureSolverOptions)

    def __post_init__(self) -> None:
        if self.differentiation not in ("implicit", "envelope"):
            raise ValueError("differentiation must be 'implicit' or 'envelope'.")
        if self.density_offset_rule not in ("inverse_sample_size", "constant"):
            raise ValueError("Unknown density offset rule.")
        if (
            not math.isfinite(self.density_offset_scale)
            or self.density_offset_scale < 0
        ):
            raise ValueError("density_offset_scale must be finite and nonnegative.")

    def density_offset(self, nobs: int) -> float:
        if nobs <= 0:
            raise ValueError("A mixture group must contain observations.")
        if self.density_offset_rule == "constant":
            return self.density_offset_scale
        return self.density_offset_scale / nobs
