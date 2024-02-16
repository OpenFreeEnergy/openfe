from importlib import resources

import click
import pytest

import openfe
from openfe import SmallMoleculeComponent
from openfecli.parameters.mol import get_molecule


def test_get_molecule_smiles():
    mol = get_molecule("CC")
    assert isinstance(mol, SmallMoleculeComponent)
    assert mol.name == ""
    assert mol.smiles == "CC"


def test_get_molecule_sdf():
    with resources.files("openfe.tests.data.serialization") as d:
        filename = d / "ethane_template.sdf"
        # Note: the template doesn't include a valid version, but it loads
        # anyway. In the future, we may need to create a temporary file with
        # template substitutions done, but that seemed like overkill now.
        mol = get_molecule(filename)
        assert mol.smiles == "CC"
        assert mol.name == "ethane"


def test_get_molecule_mol2():
    with resources.files("openfe.tests.data.lomap_basic") as d:
        f = d / "toluene.mol2"
        mol = get_molecule(str(f))

        assert mol.smiles == "Cc1ccccc1"


def test_get_molecule_error():
    with pytest.raises(click.BadParameter):
        get_molecule("foobar")
