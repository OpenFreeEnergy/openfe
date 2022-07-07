# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest
from numpy.testing import assert_allclose, assert_

import numpy as np

from .conftest import gufe_atom_mapping_matrix

from openfe.setup import perses_scorers

pytest.importorskip('perses')
pytest.importorskip('oechem')
try:
    from perses.rjmc.atom_mapping import AtomMapper, AtomMapping
except ImportError:
    pass


def test_perses_normalization_not_using_positions(gufe_atom_mapping_matrix):
    # now run the openfe equivalent with the same ligand atom _mappings
    scorer = perses_scorers.default_perses_scorer
    molecule_row = np.max(list(gufe_atom_mapping_matrix.keys()))+1
    norm_scores = np.zeros([molecule_row, molecule_row])

    for (i, j), ligand_atom_mapping in gufe_atom_mapping_matrix.items():
        norm_score = scorer(
            ligand_atom_mapping,
            use_positions=False,
            normalize=True)
        norm_scores[i, j] = norm_scores[j, i] = norm_score
    assert norm_scores.shape == (8, 8)

    assert_(np.all((norm_scores <= 1) & (norm_scores >= 0.0)),
            msg="OpenFE norm value larger than 1 or smaller than 0")


def test_perses_not_implemented_position_using(gufe_atom_mapping_matrix):
    scorer = perses_scorers.default_perses_scorer

    first_key = list(gufe_atom_mapping_matrix.keys())[0]
    match_re = "normalizing using positions is not currently implemented"
    with pytest.raises(NotImplementedError, match=match_re):
        norm_score = scorer(
            gufe_atom_mapping_matrix[first_key],
            use_positions=True,
            normalize=True)


def test_perses_regression(gufe_atom_mapping_matrix):
    # This is the way how perses does scoring
    molecule_row = np.max(list(gufe_atom_mapping_matrix.keys()))+1
    matrix = np.zeros([molecule_row, molecule_row])
    for x in gufe_atom_mapping_matrix.items():
        (i, j), ligand_atom_mapping = x
        # Build Perses Mapping:
        perses_atom_mapping = AtomMapping(
            old_mol=ligand_atom_mapping.molA.to_openff(),
            new_mol=ligand_atom_mapping.molB.to_openff(),
            old_to_new_atom_map=ligand_atom_mapping.molA_to_molB
        )
        # score Perses Mapping - Perses Style
        matrix[i, j] = matrix[j, i] = AtomMapper(
        ).score_mapping(perses_atom_mapping)

    assert matrix.shape == (8, 8)

    # now run the openfe equivalent with the same ligand atom _mappings
    scorer = perses_scorers.default_perses_scorer
    scores = np.zeros_like(matrix)
    for (i, j), ligand_atom_mapping in gufe_atom_mapping_matrix.items():
        score = scorer(
            ligand_atom_mapping,
            use_positions=True,
            normalize=False)

        scores[i, j] = scores[j, i] = score

    assert_allclose(
        actual=matrix,
        desired=scores,
        err_msg="openFE was not close to perses")
