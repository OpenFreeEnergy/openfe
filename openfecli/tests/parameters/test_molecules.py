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
        assert mols[0].smiles == "Cc1cc(C)c2cc(C)ccc2c1"
        assert mols[0].name == "*****"


def test_get_molecule_error():
    with pytest.raises(click.BadParameter):
        load_molecules("foobar")
