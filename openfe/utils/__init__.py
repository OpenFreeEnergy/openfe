# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from . import custom_typing as typing
from . import molhashing
from . import errors

# without this, `from openfe.utils.typing` raises a ModuleNotFound error --
# the import above only gives us the ability to do `from openfe.utils import
# typing`.
import sys
sys.modules['openfe.utils.typing'] = typing
del sys
