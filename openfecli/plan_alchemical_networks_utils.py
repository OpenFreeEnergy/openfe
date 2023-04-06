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
    """Write the contents of an alchemical network into directory structure

    The following structure is created:
    folder_path/
    - {base_name}.json
      - {ligand_pair_name}
        - solvent
          - transformation.json
        - vacuum
          - transformation.json
        - complex
          - transformation.json
    """
    import gufe
    from gufe import tokenization

    an_dict = alchemical_network.to_dict()

    base_name = folder_path.name
    if not folder_path.exists():
        folder_path.mkdir()

    an_json = folder_path / f"{base_name}.json"
    json.dump(an_dict, an_json.open(mode="w"), cls=tokenization.JSON_HANDLER.encoder)
    write("\t\t- " + base_name + ".json")

    transformation_dir = folder_path / "transformations"
    if not transformation_dir.exists():
        transformation_dir.mkdir()
    write("\t\t- " + str(transformation_dir) + '/')

    # group legs of a given edge together
    legs = defaultdict(list)
    for t in alchemical_network.edges:
        # "key" for each transformation
        # we're dealing with
        k = tuple(sorted([t.stateA['ligand'], t.stateB['ligand']]))

        legs[k].append(t)

    # write out ligand pair legs
    for lig_pair, v in legs.items():
        lig_path = transformation_dir / f"{lig_pair[0].name}_{lig_pair[1].name}"
        lig_path.mkdir()

        write("\t\t\t- " + f"{lig_pair[0].name}_{lig_pair[1].name}/")

        leg: gufe.Transformation
        for leg in v:
            label = deduce_label(leg)

            out_dir = lig_path / label
            out_dir.mkdir()
            out_path = out_dir / f"{base_name}_{leg.name}.json"

            leg.dump(out_path)
            write("\t\t\t\t- " + f"{label}/{base_name}_{leg.name}.json")

