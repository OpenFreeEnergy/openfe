# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from plugcli.params import NOT_PARSED
from openfecli.utils import import_thing


def import_parameter(import_str):
    try:
        result = import_thing(import_str)
    except (ImportError, AttributeError):
        result = NOT_PARSED
    return result
