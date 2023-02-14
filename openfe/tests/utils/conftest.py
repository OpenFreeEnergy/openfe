# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from rdkit import Chem
from importlib import resources

from openfe import SmallMoleculeComponent

from openfe.tests.setup.test_ligand_network import (
    mols, std_edges, simple_network
)


@pytest.fixture(scope='session')
def benzene_transforms():
    # a dict of Molecules for benzene transformations
    mols = {}
    with resources.path('openfe.tests.data',
                        'benzene_modifications.sdf') as fn:
        supplier = Chem.SDMolSupplier(str(fn), removeHs=False)
        for mol in supplier:
            mols[mol.GetProp('_Name')] = SmallMoleculeComponent(mol)
    return mols
