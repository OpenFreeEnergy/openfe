# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import itertools
import numpy as np
from numpy.testing import assert_allclose, assert_

from lomap import mcs as lomap_mcs
from perses.rjmc.atom_mapping import AtomMapper, AtomMapping


import openfe
from openfe.setup.atom_mapping import perses_scorers, LigandAtomMapping


# full back to back test again lomap
def test_perses_regression(lomap_basic_test_files_dir,  # in a dir for lomap
                           lomap_basic_test_files):

    # generate test_data - Ligand_atom_mappings with lomap
    def mapper(a, b):
        return lomap_mcs.MCS(a, b, time=20, threed=False, max3d=1000)

    ligand_atom_mappings = {}
    for i, molA in enumerate(list(lomap_basic_test_files.keys())):
        for j, molB in enumerate(list(lomap_basic_test_files.keys())):
            # Build Ligand Atom Mapping for testing
            try:
                # print(molA, lomap_basic_test_files[molA])
                mcs = mapper(
                    a=lomap_basic_test_files[molA].to_rdkit(),
                    b=lomap_basic_test_files[molB].to_rdkit())
                mapping_string = mcs.all_atom_match_list()
                atom_mapping = dict((map(int, v.split(':'))
                                    for v in mapping_string.split(',')))
            except ValueError:
                atom_mapping = {}

            ligand_mapping = LigandAtomMapping(
                molA=lomap_basic_test_files[molA],
                molB=lomap_basic_test_files[molB],
                molA_to_molB=atom_mapping
            )
            ligand_atom_mappings[(i, j)] = ligand_mapping

    # This is the way how perses does scoring
    matrix = np.zeros([len(lomap_basic_test_files),
                      len(lomap_basic_test_files)])
    for (i, j) in ligand_atom_mappings:
        ligand_atom_mapping = ligand_atom_mappings[(i, j)]
        perses_atom_mapping = AtomMapping(
            old_mol=ligand_atom_mapping.molA.to_openff(),
            new_mol=ligand_atom_mapping.molB.to_openff(),
            old_to_new_atom_map=ligand_atom_mapping.molA_to_molB
        )
        matrix[i, j] = matrix[j, i] = AtomMapper(
        ).score_mapping(perses_atom_mapping)

    assert matrix.shape == (8, 8)

    # now run the openfe equivalent with the same ligand atom _mappings
    scorer = perses_scorers.default_perses_scorer
    scores = np.zeros_like(matrix)
    norm_scores = np.zeros_like(matrix)
    for (i, j) in ligand_atom_mappings:
        ligand_atom_mapping = ligand_atom_mappings[(i, j)]
        score = scorer(
            ligand_atom_mapping,
            use_positions=True,
            normalize=False)
        norm_score = scorer(
            ligand_atom_mapping,
            use_positions=False,
            normalize=True)
        scores[i, j] = scores[j, i] = score
        norm_scores[i, j] = norm_scores[j, i] = norm_score

    # print("matrix: ")
    # print(matrix)
    # print("scores: ")
    # print(scores)
    # print("Norm scores between 0 and 1: ")
    # print(np.all((norm_score <= 1) & (norm_score >= 0.0)))

    assert_allclose(
        actual=matrix,
        desired=scores,
        err_msg="openFE was not close to perses")
    assert_(np.all((norm_score <= 1) & (norm_score >= 0.0)),
            msg="OpenFE norm value larger than 1 or smaller than 0")
