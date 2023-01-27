import pytest
from openfe.setup import LigandAtomMapping
from openfe.utils.visualization_3D import draw_mapping_on_3Dstructure

@pytest.fixture(scope='module')
def maps():
    MAPS = {
        'phenol': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
                   7: 7, 8: 8, 9: 9, 10: 12, 11: 11},
        'anisole': {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10,
                    6: 11, 7: 12, 8: 13, 9: 14, 10: 2, 11: 15}}
    return MAPS

@pytest.fixture(scope='module')
def benzene_phenol_mapping(benzene_transforms, maps):
    mol1 = benzene_transforms['benzene']
    mol2 = benzene_transforms['phenol']
    mapping = maps['phenol']
    return LigandAtomMapping(mol1, mol2,mapping)

def test_draw_mapping_on_3Dstructure(benzene_phenol_mapping):
    """
    smoke test just checking if nothing goes horribly wrong
    """
    draw_mapping_on_3Dstructure(edge=benzene_phenol_mapping)
