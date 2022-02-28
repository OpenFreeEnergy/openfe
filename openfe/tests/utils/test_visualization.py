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


@pytest.fixture(scope='module')
def maps():
    MAPS = {
        'phenol': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
                   7: 7, 8: 8, 9: 9, 10: 12, 11: 11},
        'anisole': {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10,
                    6: 11, 7: 12, 8: 13, 9: 14, 10: 2, 11: 15}}
    return MAPS


@pytest.mark.parametrize('molname, atoms, elems, bonds', [
    ['phenol', {10,}, {12,}, {10, 12}],
    ['anisole', {0, 1, 3, 4}, {2,}, {0, 1, 2, 3, 13}]
])
def test_benzene_to_phenol_uniques(molname, atoms, elems, bonds,
                                   benzene_transforms, maps):
    mol1 = benzene_transforms['benzene']
    mol2 = benzene_transforms[molname]

    mapping = maps[molname]

    uniques = _get_unique_bonds_and_atoms(mapping,
                                          mol1.to_rdkit(), mol2.to_rdkit())

    # The benzene perturbations don't change
    # no unique atoms in benzene
    assert uniques['atoms'] == set()
    # H->O
    assert uniques['elements'] == {10,}
    # One bond involved
    assert uniques['bonds'] == {10,}

    # invert and check the mol2 uniques
    inv_map = {v:k for k,v in mapping.items()}

    uniques = _get_unique_bonds_and_atoms(inv_map,
                                          mol2.to_rdkit(), mol1.to_rdkit())

    assert uniques['atoms'] == atoms
    assert uniques['elements'] == elems
    assert uniques['bonds'] == bonds
