# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from plugcli.params import NOT_PARSED
from openfecli.utils import import_thing


def import_parameter(import_str: str):
    """Return object from a qualname, or NOT_PARSED if not valid.

    This is used specifically for parameter instantiation strategies based
    on importing an object given by the user on the command line. If the
    user input cannot interpreted as a qualname, then NOT_PARSED is
    returned.

    Parameters
    ----------
    import_str : str
        the qualname

    Returns
    -------
    Any :
        the desired object or NOT_PARSED if an error was encountered.
    """
    try:
        result = import_thing(import_str)
    except (ImportError, AttributeError):
        result = NOT_PARSED
    return result