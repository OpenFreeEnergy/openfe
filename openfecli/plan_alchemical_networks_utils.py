import os
import json
import pathlib
from openfecli.utils import write

from gufe import AlchemicalNetwork

def plan_alchemical_network_output(alchemical_network: AlchemicalNetwork, folder_path: pathlib.Path):
    # Todo: this is an uggly peace of output code.
    #  it does not recognize overwriting
    an_dict = alchemical_network.to_dict()

    base_name = folder_path.name
    if not folder_path.exists():
        folder_path.mkdir()

    an_json = folder_path / f"{base_name}.json"
    json.dump(an_dict, an_json.open(mode="w"))
    write("\t\t- " + base_name + ".json")

    transformation_dir = folder_path / "transformations"
    if not transformation_dir.exists():
        transformation_dir.mkdir()
    write("\t\t- " + str(transformation_dir))

    for transformation in alchemical_network.edges:
        out_path = transformation_dir / f"{base_name}_{transformation.name}.json"
        transformation.dump(out_path)
        write("\t\t\t- " + base_name + "_" + transformation.name + ".json")
