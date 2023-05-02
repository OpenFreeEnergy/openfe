# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import os
from collections import defaultdict
import json
import pathlib
from openfecli.utils import write

from gufe import AlchemicalNetwork


def deduce_label(t) -> str:
    """Take a transformation and classify as either 'complex', 'solvent' or 'vacuum' leg"""
    s = t.stateA

    if 'protein' in s.components:
        return 'complex'
    elif 'solvent' in s.components:
        return 'solvent'
    else:
        return 'vacuum'


def plan_alchemical_network_output(alchemical_network: AlchemicalNetwork, folder_path: pathlib.Path):
    """Write the contents of an alchemical network into the structure
    """
    import gufe
    from gufe import tokenization

    an_dict = alchemical_network.to_dict()

    base_name = folder_path.name
    folder_path.mkdir(parents=True, exist_ok=True)

    an_json = folder_path / f"{base_name}.json"
    json.dump(an_dict, an_json.open(mode="w"), cls=tokenization.JSON_HANDLER.encoder)
    write("\t\t- " + base_name + ".json")

    for transformation in alchemical_network.edges:
        transformation_name = transformation.name or transformation.key
        filename = f"{transformation_name}.json"
        transformation.dump(folder_path / filename)
        write("\t\t\t\t- " + filename)


