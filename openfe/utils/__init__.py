# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Utilities and tools for OpenFE.

Of particular note is the :class:`.GufeTokenizable` class, which enforces
immutability and provides serialization tools for most OpenFE data types.
"""

from . import custom_typing
from .optional_imports import requires_package
from .remove_oechem import without_oechem_backend
from .system_probe import log_system_probe
from gufe.tokenization import GufeTokenizable

__all__ = [
    "GufeTokenizable",
    "custom_typing",
    "requires_package",
    "without_oechem_backend",
    "log_system_probe",
]
