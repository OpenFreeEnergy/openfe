# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from .plugins import OFECommandPlugin
from . import commands

from importlib.metadata import version
__version__ = version("openfe")
