import pytest
from unittest import mock
import inspect

from rdkit import Chem

from openfe.utils.visualization import (
    _match_elements, _get_unique_bonds_and_atoms, draw_mapping,
    draw_one_molecule_mapping, draw_unhighlighted_molecule
)

# default colors currently used
_HIGHLIGHT_COLOR = (220/255, 50/255, 32/255, 1)
_CHANGED_ELEMENTS_COLOR = (0, 90/255, 181/255, 1)


def bound_args(func, args, kwargs):
    """Return a dictionary mapping parameter name to value.

    Parameters
    ----------
    func : Callable
        this must be inspectable; mocks will require a spec
    args : List
        args list
    kwargs : Dict
        kwargs Dict

    Returns
    -------
    Dict[str, Any] :
        mapping of string name of function parameter to the value it would
        be bound to
    """
    sig = inspect.Signature.from_callable(func)
    bound = sig.bind(*args, **kwargs)
    return bound.arguments


@pytest.mark.parametrize("at1, idx1, at2, idx2, response", [
    ["N", 0, "C", 0, False],
    ["C", 0, "C", 0, True],
    ["COC", 1, "NOC", 1, True],
    ["COON", 2, "COC", 2, False]]
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


@pytest.fixture(scope='module')
def benzene_phenol_mapping(benzene_transforms, maps):
    mol1 = benzene_transforms['benzene'].to_rdkit()
    mol2 = benzene_transforms['phenol'].to_rdkit()
    mapping = maps['phenol']
    return mapping, mol1, mol2


@pytest.mark.parametrize('molname, atoms, elems, bonds', [
    ['phenol', {10, }, {12, }, {10, 12}],
    ['anisole', {0, 1, 3, 4}, {2, }, {0, 1, 2, 3, 13}]
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
    assert uniques['elements'] == {10, }
    # One bond involved
    assert uniques['bonds'] == {10, }

    # invert and check the mol2 uniques
    inv_map = {v: k for k, v in mapping.items()}

    uniques = _get_unique_bonds_and_atoms(inv_map,
                                          mol2.to_rdkit(), mol1.to_rdkit())

    assert uniques['atoms'] == atoms
    assert uniques['elements'] == elems
    assert uniques['bonds'] == bonds


@mock.patch("openfe.utils.visualization._draw_molecules", autospec=True)
def test_draw_mapping(mock_func, benzene_phenol_mapping):
    # ensure that draw_mapping passes the desired parameters to our internal
    # _draw_molecules method
    mapping, mol1, mol2 = benzene_phenol_mapping
    draw_mapping(mapping, mol1, mol2)

    mock_func.assert_called_once()
    args = bound_args(mock_func, mock_func.call_args.args,
                      mock_func.call_args.kwargs)
    assert args['mols'] == [mol1, mol2]
    assert args['atoms_list'] == [{10}, {10, 12}]
    assert args['bonds_list'] == [{10}, {10, 12}]
    assert args['atom_colors'] == [{10: _CHANGED_ELEMENTS_COLOR},
                                   {12: _CHANGED_ELEMENTS_COLOR}]
    assert args['highlight_color'] == _HIGHLIGHT_COLOR


@pytest.mark.parametrize('inverted', [True, False])
@mock.patch("openfe.utils.visualization._draw_molecules", autospec=True)
def test_draw_one_molecule_mapping(mock_func, benzene_phenol_mapping,
                                   inverted):
    # ensure that draw_one_molecule_mapping passes the desired parameters to
    # our internal _draw_molecules method
    mapping, mol1, mol2 = benzene_phenol_mapping
    if inverted:
        mapping = {v: k for k, v in mapping.items()}
        mol1, mol2 = mol2, mol1
        atoms_list = [{10, 12}]
        bonds_list = [{10, 12}]
        atom_colors = [{12: _CHANGED_ELEMENTS_COLOR}]
    else:
        atoms_list = [{10}]
        bonds_list = [{10}]
        atom_colors = [{10: _CHANGED_ELEMENTS_COLOR}]

    draw_one_molecule_mapping(mapping, mol1, mol2)

    mock_func.assert_called_once()
    args = bound_args(mock_func, mock_func.call_args.args,
                      mock_func.call_args.kwargs)

    assert args['mols'] == [mol1]
    assert args['atoms_list'] == atoms_list
    assert args['bonds_list'] == bonds_list
    assert args['atom_colors'] == atom_colors
    assert args['highlight_color'] == _HIGHLIGHT_COLOR


@mock.patch("openfe.utils.visualization._draw_molecules", autospec=True)
def test_draw_unhighlighted_molecule(mock_func, benzene_transforms):
    # ensure that draw_unhighlighted_molecule passes the desired parameters
    # to our internal _draw_molecules method
    mol = benzene_transforms['benzene'].to_rdkit()
    draw_unhighlighted_molecule(mol)

    mock_func.assert_called_once()
    args = bound_args(mock_func, mock_func.call_args.args,
                      mock_func.call_args.kwargs)
    assert args['mols'] == [mol]
    assert args['atoms_list'] == [[]]
    assert args['bonds_list'] == [[]]
    assert args['atom_colors'] == [{}]
    # technically, we don't care what the highlight color is, so no
    # assertion on that


def test_draw_mapping_integration_smoke(benzene_phenol_mapping):
    # integration test/smoke test to catch errors if the upstream drawing
    # code changes
    draw_mapping(*benzene_phenol_mapping)


def test_draw_one_molecule_integration_smoke(benzene_phenol_mapping):
    # integration test/smoke test to catch errors if the upstream drawing
    # code changes
    draw_one_molecule_mapping(*benzene_phenol_mapping)


def test_draw_unhighlighted_molecule_integration_smoke(benzene_transforms):
    # integration test/smoke test to catch errors if the upstream drawing
    # code changes
    draw_unhighlighted_molecule(benzene_transforms['benzene'].to_rdkit())
