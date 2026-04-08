from importlib import resources

import pytest
from gufe import ProteinComponent, ProteinMembraneComponent
from rdkit import Chem

from openfecli.parameters.protein import _load_protein_membrane_file, get_protein


def test_get_protein_pdb():
    with resources.as_file(resources.files("gufe.tests.data")) as d:
        filename = str(d / "181l.pdb")
        protein_comp = get_protein(filename)

        assert isinstance(protein_comp, ProteinComponent)
        assert isinstance(protein_comp.to_rdkit(), Chem.Mol)


def test_get_protein_pdbx():
    with resources.as_file(resources.files("gufe.tests.data")) as d:
        filename = str(d / "181l.cif")
        protein_comp = get_protein(filename)

        assert isinstance(protein_comp, ProteinComponent)
        assert isinstance(protein_comp.to_rdkit(), Chem.Mol)


def test_load_protein_membrane_error():
    with resources.as_file(resources.files("gufe.tests.data")) as d:
        filename = str(d / "181l.cif")

        with pytest.raises(ValueError, "as ProteinMembraneComponent"):
            _ = _load_protein_membrane_file(filename)


def test_load_protein_membrane_a2a(a2a_protein_membrane_pdb):
    with resources.as_file(resources.files("gufe.tests.data")) as d:
        filename = str(d / "a2a.pdb.gz")
    # TODO: refactor context out of this?
    protein_membrane_comp = _load_protein_membrane_file(filename, None)
    assert isinstance(protein_membrane_comp, ProteinMembraneComponent)
