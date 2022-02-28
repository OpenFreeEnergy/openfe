import pytest

from rdkit import Chem

from openfe.utils.visualization import (_match_elements,
                                        _get_unique_bonds_and_atoms,)

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


def test_benzene_to_phenol_uniques(benzene_transforms):
    mol1 = benzene_transforms['benzene']
    mol2 = benzene_transforms['phenol']

    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
               7: 7, 8: 8, 9: 9, 10: 12, 11: 11}

    uniques = _get_unique_bonds_and_atoms(mapping,
                                          mol1.to_rdkit(), mol2.to_rdkit())

    # no unique atoms in benzene
    assert uniques['atoms'] == set()
    # H->O
    assert uniques['elements'] == {10,}
    # One bond involved
    assert uniques['bonds'] == {10,}
