"""
Write out the file that is used to test the quickrun command.

This will need to be run if the serialized transformation changes such that
the old file can't be read.

USAGE:

    python write_transformation_json.py ../data/

(Assuming you run from within this directory.)
"""

import gufe
import openfe
import json
import argparse
import pathlib

from gufe.tests.conftest import benzene_modifications
from gufe.tests.test_protocol import DummyProtocol, BrokenProtocol
from gufe.tokenization import JSON_HANDLER

parser = argparse.ArgumentParser()
parser.add_argument("directory")
opts = parser.parse_args()
directory = pathlib.Path(opts.directory)
if not directory.exists() and directory.is_dir():
    raise ValueError(f"Bad parameter for directory: {directory}")

benzene_modifications = benzene_modifications.__pytest_wrapped__.obj()

benzene = gufe.SmallMoleculeComponent.from_rdkit(benzene_modifications["benzene"])
toluene = gufe.SmallMoleculeComponent.from_rdkit(benzene_modifications["toluene"])
solvent = gufe.SolventComponent(positive_ion="K", negative_ion="Cl")

benz_dict = {"ligand": benzene}
tol_dict = {"ligand": toluene}
solv_dict = {"solvent": solvent}

solv_benz = gufe.ChemicalSystem(dict(**benz_dict, **solv_dict))
solv_tol = gufe.ChemicalSystem(dict(**tol_dict, **solv_dict))

mapper = openfe.setup.LomapAtomMapper()
mapping = list(mapper.suggest_mappings(benzene, toluene))[0]

protocol = DummyProtocol(settings=DummyProtocol.default_settings())

transformation = gufe.Transformation(solv_benz, solv_tol, protocol, mapping)
bad_protocol = BrokenProtocol(settings=BrokenProtocol.default_settings())
bad_transformation = gufe.Transformation(solv_benz, solv_tol, bad_protocol, mapping)

transformation.dump(directory / "transformation.json")
bad_transformation.dump(directory / "bad_transformation.json")
