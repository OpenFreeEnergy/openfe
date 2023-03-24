import importlib
from importlib import resources

from rdkit import Chem
from gufe import ProteinComponent
from openfecli.parameters.protein import get_molecule

def test_get_protein_pdb():
    with importlib.resources.path("openfe.tests.data",
                                  "181l_only.pdb") as filename:
        protein_comp = get_molecule(str(filename))
                                            
        assert isinstance(protein_comp, ProteinComponent)
        assert isinstance(protein_comp.to_rdkit(), Chem.Mol)
