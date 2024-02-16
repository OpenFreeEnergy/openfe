from importlib import resources

from gufe import ProteinComponent
from rdkit import Chem

from openfecli.parameters.protein import get_molecule


def test_get_protein_pdb():
    with resources.files("gufe.tests.data") as d:
        filename = str(d / "181l.pdb")
        protein_comp = get_molecule(filename)

        assert isinstance(protein_comp, ProteinComponent)
        assert isinstance(protein_comp.to_rdkit(), Chem.Mol)


def test_get_protein_pdbx():
    with resources.files("gufe.tests.data") as d:
        filename = str(d / "181l.cif")
        protein_comp = get_molecule(filename)

        assert isinstance(protein_comp, ProteinComponent)
        assert isinstance(protein_comp.to_rdkit(), Chem.Mol)
