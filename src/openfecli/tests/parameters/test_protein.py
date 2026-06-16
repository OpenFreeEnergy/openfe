from importlib import resources

import click
import pytest
from rdkit import Chem

from openfe import ProteinComponent, ProteinMembraneComponent
from openfe.tests.conftest import a2a_protein_membrane_pdb
from openfecli.parameters.protein import _get_protein, _get_protein_membrane
from openfecli.tests.commands.test_plan_rbfe_network import protein_membrane_args


def test_load_protein_pdb():
    with resources.as_file(resources.files("gufe.tests.data")) as d:
        filename = str(d / "181l.pdb")
    protein_comp = _get_protein(filename, None)

    assert isinstance(protein_comp, ProteinComponent)
    assert isinstance(protein_comp.to_rdkit(), Chem.Mol)


def test_load_protein_pdbx():
    with resources.as_file(resources.files("gufe.tests.data")) as d:
        filename = str(d / "181l.cif")
    protein_comp = _get_protein(filename, None)

    assert isinstance(protein_comp, ProteinComponent)
    assert isinstance(protein_comp.to_rdkit(), Chem.Mol)


def test_load_protein_invalid_file_error():
    with pytest.raises(click.exceptions.BadParameter, match="File extension must contain"):
        _get_protein("not_a_file.txt", None)


def test_load_protein_with_membrane(a2a_protein_membrane_pdb):
    """A protein-membrane component passed to --protein won't fail here, but will be caught in validation downstream."""

    protein_comp = _get_protein(a2a_protein_membrane_pdb, None)
    assert isinstance(protein_comp, ProteinComponent)
    assert isinstance(protein_comp.to_rdkit(), Chem.Mol)


def test_load_membrane_pdb(a2a_protein_membrane_pdb):
    protein_membrane_comp = _get_protein_membrane(a2a_protein_membrane_pdb, None)
    assert isinstance(protein_membrane_comp, ProteinMembraneComponent)
    assert isinstance(protein_membrane_comp.to_rdkit(), Chem.Mol)


def test_load_membrane_pdbx():
    with resources.as_file(resources.files("gufe.tests.data")) as d:
        filename = str(d / "a2a.cif.gz")
    protein_membrane_comp = _get_protein_membrane(filename, None)
    assert isinstance(protein_membrane_comp, ProteinMembraneComponent)
    assert isinstance(protein_membrane_comp.to_rdkit(), Chem.Mol)


def test_load_membrane_with_protein_only():
    """Should error if a protein-only system (like 181l.pdb) is loaded as a ProteinMembraneComponent."""
    with resources.as_file(resources.files("gufe.tests.data")) as d:
        filename = str(d / "181l.pdb")

    with pytest.raises(click.exceptions.BadParameter, match="Could not determine box_vectors."):
        _ = _get_protein_membrane(filename, None)
