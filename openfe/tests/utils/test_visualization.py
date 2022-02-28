import pytest

from rdkit import Chem

from openfe.utils.visualization import _match_elements

@pytest.mark.parametrize("at1, idx1, at2, idx2, response", [
    ["N", 0, "C", 0, False,],
    ["C", 0, "C", 0, True,],
    ["COC", 1, "NOC", 1, True],
    ["COON", 2, "COC", 2, False],]
)
def test_match_elements(at1, idx1, at2, idx2, response):
    mol1 = Chem.MolFromSmiles(at1)
    mol2 = Chem.MolFromSmiles(at2)
    retval = _match_elements(mol1, idx2, mol2, idx2)

    assert retval == response

