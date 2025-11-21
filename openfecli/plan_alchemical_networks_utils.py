# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from __future__ import annotations

import json
import pathlib

from openfe import AlchemicalNetwork, LigandNetwork
from openfecli.utils import write


def plan_alchemical_network_output(
    alchemical_network: AlchemicalNetwork,
    ligand_network: LigandNetwork,
    folder_path: pathlib.Path,
):
    """Write the contents of an alchemical network into the structure"""

    base_name = folder_path.name
    folder_path.mkdir(parents=True, exist_ok=True)

    an_json = folder_path / f"{base_name}.json"
    alchemical_network.to_json(an_json)
    write("\t\t- " + base_name + ".json")

    ln_fname = "ligand_network.graphml"
    with open(folder_path / ln_fname, mode="w") as f:
        f.write(ligand_network.to_graphml())
    write(f"\t\t- {ln_fname}")

    transformations_dir = folder_path / "transformations"
    transformations_dir.mkdir(parents=True, exist_ok=True)

    for transformation in alchemical_network.edges:
        transformation_name = transformation.name or transformation.key
        filename = f"{transformation_name}.json"
        transformation.to_json(transformations_dir / filename)
        write("\t\t\t\t- " + filename)
