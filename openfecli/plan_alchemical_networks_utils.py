# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from __future__ import annotations
import os
from collections import defaultdict
import json
import pathlib
from openfecli.utils import write
import typing

def plan_alchemical_network_output(
    alchemical_network: AlchemicalNetwork,
    ligand_network: LigandNetwork,
    folder_path: pathlib.Path
):
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

    ln_fname = "ligand_network.graphml"
    with open(folder_path / ln_fname, mode='w') as f:
        f.write(ligand_network.to_graphml())
    write(f"\t\t- {ln_fname}")

    transformations_dir = folder_path / "transformations"
    transformations_dir.mkdir(parents=True, exist_ok=True)

    for transformation in alchemical_network.edges:
        transformation_name = transformation.name or transformation.key
        filename = f"{transformation_name}.json"
        transformation.dump(transformations_dir / filename)
        write("\t\t\t\t- " + filename)


