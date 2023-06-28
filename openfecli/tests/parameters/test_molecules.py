import os
import importlib
from importlib import resources

import pytest
import click

import openfe
from openfecli.parameters.molecules import load_molecules
from openfe import SmallMoleculeComponent


def test_get_dir_molecules_sdf():
    with importlib.resources.path(
        "openfe.tests.data.serialization", "__init__.py"
    ) as file_path:
        # Note: the template doesn't include a valid version, but it loads
        # anyway. In the future, we may need to create a temporary file with
        # template substitutions done, but that seemed like overkill now.
        dir_path = os.path.dirname(file_path)
        mols = load_molecules(dir_path)

        assert len(mols) == 1
        assert mols[0].smiles == "CC"
        assert mols[0].name == "ethane"

def test_load_molecules_sdf_file():
    files = importlib.resources.files('openfe.tests.data')
    ref = files / "benzene_modifications.sdf"
    with importlib.resources.as_file(ref) as path:
        mols = load_molecules(path)

    assert len(mols) == 7

def test_get_dir_molecules_mol2():
    with importlib.resources.path(
        "openfe.tests.data.lomap_basic", "__init__.py"
    ) as file_path:
        # Note: the template doesn't include a valid version, but it loads
        # anyway. In the future, we may need to create a temporary file with
        # template substitutions done, but that seemed like overkill now.
        dir_path = os.path.dirname(file_path)
        mols = load_molecules(dir_path)

        assert len(mols) == 8
        all_smiles = {mol.smiles for mol in mols}
        all_names = {mol.name for mol in mols}
        assert "Cc1cc(C)c2cc(C)ccc2c1" in all_smiles
        assert "*****" in all_names


def test_get_molecule_error():
    with pytest.raises(ValueError, match="Unable to find"):
        load_molecules("foobar")
