# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import importlib
import pytest
from importlib import resources
from rdkit import Chem

import gufe
import openfe
from gufe import SmallMoleculeComponent


@pytest.fixture(scope='session')
def benzene_modifications():
    files = {}
    with importlib.resources.path('openfe.tests.data',
                                  'benzene_modifications.sdf') as fn:
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)
        for rdmol in supp:
            files[rdmol.GetProp('_Name')] = SmallMoleculeComponent(rdmol)
    return files


@pytest.fixture(scope='session')
def T4_protein_component():
    with resources.path('openfe.tests.data', '181l_only.pdb') as fn:
        comp = gufe.ProteinComponent.from_pdb_file(str(fn), name="T4_protein")

    return comp
