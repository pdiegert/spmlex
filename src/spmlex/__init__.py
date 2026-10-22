__all__ = [
    "profile_ll",
    "profile_mixture",
    "profile_ll_diagnostics",
    "profile_mixture_diagnostics",
    "ProfileOptions",
    "MixtureSolverOptions",
    "ProfileDiagnostics",
    "GroupDiagnostics",
]

import jax
from .options import MixtureSolverOptions, ProfileOptions
from .profile_lk import (
    GroupDiagnostics,
    ProfileDiagnostics,
    profile_ll,
    profile_ll_diagnostics,
    profile_mixture,
    profile_mixture_diagnostics,
)

# Enable 64 bit floating point precision
jax.config.update("jax_enable_x64", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
jax.config.update("jax_platform_name", "cpu")
