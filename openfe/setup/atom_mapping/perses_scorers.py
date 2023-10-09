# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Functions from Perses for scoring atom mappings.
"""

from typing import Callable

from openfe.utils import requires_package

from ...utils.silence_root_logging import silence_root_logging
try:
    with silence_root_logging():
        from perses.rjmc.atom_mapping import AtomMapper, AtomMapping
except ImportError:
    pass    # Don't throw  error, will happen later

from . import LigandAtomMapping


# Helpfer Function / reducing code amount
def _get_all_mapped_atoms_with(oeyMolA,
                               oeyMolB,
                               numMaxPossibleMappingAtoms: int,
                               criterium: Callable) -> int:
    molA_allAtomsWith = len(
        list(filter(criterium, oeyMolA.GetAtoms())))
    molB_allAtomsWith = len(
        list(filter(criterium, oeyMolB.GetAtoms())))

    if (molA_allAtomsWith > molB_allAtomsWith and
            molA_allAtomsWith <= numMaxPossibleMappingAtoms):
        numMaxPossibleMappings = molA_allAtomsWith
    else:
        numMaxPossibleMappings = molB_allAtomsWith

    return numMaxPossibleMappings


@requires_package("perses")
def default_perses_scorer(mapping: LigandAtomMapping,
                          use_positions: bool = False,
                          normalize: bool = True) -> float:
    """
    Score an atom mapping with the default Perses score function.

    Parameters
    ----------
    mapping: LigandAtomMapping
        is an OpenFE Ligand Mapping, that should be mapped
    use_positions: bool, optional
        if the positions are used, perses takes the inverse eucledian distance
        of mapped atoms into account.
        else the number of mapped atoms is used for the score.
        default True
    normalize: bool, optional
        if true, the scores get normalized, such that different molecule pairs
        can be compared for one scorer metric, default = True
        *Warning* does not work for use_positions right now!

    Raises
    ------
    NotImplementedError
        Normalization of the score using positions is not implemented right
        now.

    Returns
    -------
        float
    """
    score = AtomMapper(use_positions=use_positions).score_mapping(
        AtomMapping(old_mol=mapping.componentA.to_openff(),
                    new_mol=mapping.componentB.to_openff(),
                    old_to_new_atom_map=mapping.componentA_to_componentB))

    # normalize
    if (normalize):
        oeyMolA = mapping.componentA.to_openff().to_openeye()
        oeyMolB = mapping.componentB.to_openff().to_openeye()
        if (use_positions):
            raise NotImplementedError("normalizing using positions is "
                                      "not currently implemented")
        else:

            smallerMolecule = oeyMolA if (
                    oeyMolA.NumAtoms() < oeyMolB.NumAtoms()) else oeyMolB
            numMaxPossibleMappingAtoms = smallerMolecule.NumAtoms()
            # Max possible Aromatic mappings
            numMaxPossibleAromaticMappings = _get_all_mapped_atoms_with(
                oeyMolA=oeyMolA, oeyMolB=oeyMolB,
                numMaxPossibleMappingAtoms=numMaxPossibleMappingAtoms,
                criterium=lambda x: x.IsAromatic())

            # Max possible heavy mappings
            numMaxPossibleHeavyAtomMappings = _get_all_mapped_atoms_with(
                oeyMolA=oeyMolA, oeyMolB=oeyMolB,
                numMaxPossibleMappingAtoms=numMaxPossibleMappingAtoms,
                criterium=lambda x: x.GetAtomicNum() > 1)

            # Max possible ring mappings
            numMaxPossibleRingMappings = _get_all_mapped_atoms_with(
                oeyMolA=oeyMolA, oeyMolB=oeyMolB,
                numMaxPossibleMappingAtoms=numMaxPossibleMappingAtoms,
                criterium=lambda x: x.IsInRing())

            # These weights are totally arbitrary
            normalize_score = (1.0 * numMaxPossibleMappingAtoms +
                               0.8 * numMaxPossibleAromaticMappings +
                               0.5 * numMaxPossibleHeavyAtomMappings +
                               0.4 * numMaxPossibleRingMappings)

        score /= normalize_score  # final normalize score

    return score
