import importlib
from importlib import resources

import pytest
import click

import openfe
from openfecli.parameters.mol import get_molecule
from openfe.setup import Molecule


def test_get_molecule_smiles():
    mol = get_molecule("CC")
    assert isinstance(mol, Molecule)
    assert mol.name == ""
    assert mol.smiles == "CC"


def test_get_molecule_sdf():
    with importlib.resources.path("openfe.tests.data.serialization",
                                  "ethane_template.sdf") as filename:
        # Note: the template doesn't include a valid version, but it loads
        # anyway. In the future, we may need to create a temporary file with
        # template substitutions done, but that seemed like overkill now.
        mol = get_molecule(filename)
        assert mol.smiles == "CC"
        assert mol.name == "ethane"


def test_get_molecule_mol2():
    with importlib.resources.path("openfe.tests.data.lomap_basic",
                                  "toluene.mol2") as f:
        mol = get_molecule(str(f))

        assert mol.smiles == 'Cc1ccccc1'


def test_get_molecule_error():
    with pytest.raises(click.BadParameter):
        get_molecule("foobar")
