__all__ = ["profile_ll"]

import jax
from .profile_lk import profile_ll

# Enable 64 bit floating point precision
jax.config.update("jax_enable_x64", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
jax.config.update('jax_platform_name', 'cpu')


