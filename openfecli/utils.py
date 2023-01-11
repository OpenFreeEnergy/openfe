# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import importlib
import click


def import_thing(import_string: str):
    """Obtain an object from a valid qualname (or fully qualified name)

    Parameters
    ----------
    import_string : str
        the qualname

    Returns
    -------
    Any :
        the object from that namespace
    """
    splitted = import_string.split('.')
    if len(splitted) > 1:
        # if the string has a dot, import the module and getattr the object
        obj = splitted[-1]
        mod = ".".join(splitted[:-1])
        module = importlib.import_module(mod)
        result = getattr(module, obj)
    else:
        # if there is no dot, import and return the module
        mod = splitted[0]
        result = importlib.import_module(mod)
    return result


def write(string: str):
    """

    This is abstracted so that we can change output mechanism here and it
    will automatically update in all commands.
    """
    click.echo(string)

