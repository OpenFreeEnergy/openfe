# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
import importlib
import functools
from typing import Callable
from datetime import datetime


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


def print_duration(function: Callable) -> Callable:
    """
    Helper function to denote that a function should print a duration information.
    A function decorated with this decorator will print out the execution time of
    the decorated function.

    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()

        result = function(*args, **kwargs)

        end_time = datetime.now()
        duration = end_time - start_time
        write("\tDuration: " + str(duration) + "\n")
        return result

    return wrapper


def handle_radial_generator(radial_hub, molecules):
    """Create the network generator for radial networks

    Returns
    -------
    planner, molecules
      the planner function and list of molecules

    Raises
    ------
    click.BadParameter
      if ambiguous or unknown radial_hub is used
    """
    from functools import partial
    from openfe.setup.ligand_network_planning import generate_radial_network

    if radial_hub is not None:
        # check that hub input is unambiguous
        candidates = [m for m in molecules if m.name == radial_hub]
        if len(candidates) > 1:
            raise click.BadParameter(f"Ambiguous ligand name ({radial_hub}) given as radial_hub")
        elif not candidates:
            raise click.BadParameter(f"ligand name {radial_hub} not found")
        central_lig = candidates[0]
    else:
        central_lig = molecules[0]
    # avoid hub to hub edges
    small_molecules = [m for m in molecules if m is not central_lig]
    write(f"\tUsing {central_lig} as radial hub")
    ligand_network_planner = partial(generate_radial_network,
                                     central_ligand=central_lig)

    return ligand_network_planner, molecules
