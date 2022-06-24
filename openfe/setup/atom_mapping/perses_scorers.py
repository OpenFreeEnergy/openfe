# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from perses.rjmc.atom_mapping import AtomMapper, AtomMapping

from .ligandatommapping import LigandAtomMapping


def default_perses_scorer(mapping: LigandAtomMapping):
    """

    Parameters
    ----------
    mapping

    Returns
    -------

    """
    score = AtomMapper().score_mapping(AtomMapping(old_mol=mapping.molA.to_openff(), new_mol=mapping.molB.to_openff(),
                                                   old_to_new_atom_map=mapping.molA_to_molB))
    return score
