import pytest
import click

from openfecli.parameters.mol import get_molecule
from openfe.setup import Molecule


def test_get_molecule():
    mol = get_molecule("CC")
    assert isinstance(mol, Molecule)
    assert mol.name == ""
    assert mol.smiles == "CC"


def test_get_molecule_error():
    with pytest.raises(click.BadParameter):
        get_molecule("foobar")
