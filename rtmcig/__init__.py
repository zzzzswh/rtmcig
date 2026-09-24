"""rtmcig: GPU reverse-time migration to angle-domain common-image gathers.

    from rtmcig import migrate, RTMCIG

``migrate`` is the one-shot function; ``RTMCIG`` is the operator, which also
does forward modelling with the same geometry (``op.model``). Wave
propagation is Deepwave's (scalar or staggered-grid acoustic, C-PML); the
imaging conditions and the angle decomposition are PyTorch.
"""
from ._version import __version__
from ._operator import RTMCIG, fold_angles, migrate
from ._wavelet import ricker
from ._imaging import half_derivative, laplacian, slant_stack
from ._geometry import Survey


def cuda_available() -> bool:
    """True when a CUDA device is visible to PyTorch."""
    import torch
    return torch.cuda.is_available()


__all__ = [
    "__version__",
    "RTMCIG",
    "migrate",
    "fold_angles",
    "ricker",
    "slant_stack",
    "laplacian",
    "half_derivative",
    "Survey",
    "cuda_available",
]
