# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from importlib.metadata import version

from . import commands
from .plugins import OFECommandPlugin

__version__ = version("openfe")
