import importlib

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
    contents = importlib.resources.read_text(
        "openfe.tests.data.serialization", "ethane_template.sdf"
    ).format(OFE_VERSION=openfe.__version__)

    mol = Molecule.from_sdf_string(contents)
    assert mol.name == "ethane"
    assert mol.smiles == "CC"


def test_get_molecule_error():
    with pytest.raises(click.BadParameter):
        get_molecule("foobar")
