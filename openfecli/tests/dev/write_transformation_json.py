"""
Write out the file that is used to test the quickrun command.

This will need to be run if the serialized transformation changes such that
the old file can't be read.

USAGE:

    python write_transformation_json.py > ../data/transformation.json

(Assuming you run from within this directory.)
"""

import gufe
import json

from gufe.tests.conftest import benzene_modifications
from gufe.tests.test_protocol import DummyProtocol

benzene_modifications = benzene_modifications.__pytest_wrapped__.obj()

benzene = gufe.SmallMoleculeComponent.from_rdkit(benzene_modifications['benzene'])
toluene = gufe.SmallMoleculeComponent.from_rdkit(benzene_modifications['toluene'])
solvent = gufe.SolventComponent(positive_ion="K", negative_ion="Cl")

benz_dict = {'ligand': benzene}
tol_dict = {'ligand': toluene}
solv_dict = {'solvent': solvent}

solv_benz = gufe.ChemicalSystem(dict(**benz_dict, **solv_dict))
solv_tol = gufe.ChemicalSystem(dict(**tol_dict, **solv_dict))

mapping = None
# waiting on AtomMapping being gufe tokenizable
# mapper = openfe.setup.LomapAtomMapper()
# mapping = list(mapper.suggest_mappings(benzene, toluene))[0]

protocol = DummyProtocol()

transformation = gufe.Transformation(solv_benz, solv_tol, protocol, mapping)

# TODO: update to handle our encoder
print(json.dumps(transformation.to_dict()))
