# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing.Callable import callable
from perses.rjmc.atom_mapping import AtomMapper, AtomMapping

from .ligandatommapping import LigandAtomMapping


def default_perses_scorer(mapping: LigandAtomMapping, use_positions: bool = False, normalize: bool = True) -> float:
    """
        This function is accessing the default perser scorer function and returns, the score as float.

    Parameters
    ----------
    mapping: LigandAtomMapping
        is an openFE Ligand Mapping, that should be mapped
    use_positions: bool, optional
        if the positions are used, perses takes the inverse eucledian distance of mapped atoms into account.
        else the number of mapped atoms is used for the score.
        default True
    normalize: bool, optional
        if true, the scores get normalized, such that different molecule pairs can be compared for one scorer metric, default = True
        *Warning* does not work for use_positions right now!

    Returns
    -------
        float
    """
    score = AtomMapper(use_positions=use_positions).score_mapping(
        AtomMapping(old_mol=mapping.molA.to_openff(), new_mol=mapping.molB.to_openff(),
                    old_to_new_atom_map=mapping.molA_to_molB))

    # normalize
    if (normalize):
        oeyMolA = mapping.molA.to_openeye()
        oeyMolB = mapping.molB.to_openeye()
        if (use_positions):
            raise Exception("Not Implemented yet")

        else:
            # Helpfer Function / reducing code amount
            def _getAllMappableAtomsWith(oeyMolA, oeyMolB, numMaxPossibleMappingAtoms: int, criterium: callable) -> int:
                molA_allAtomsWith = len(list(filter(criterium, oeyMolA.GetAtoms())))
                molB_allAtomsWith = len(list(filter(criterium, oeyMolB.GetAtoms())))

                if (molA_allAtomsWith > molB_allAtomsWith and molA_allAtomsWith <= numMaxPossibleMappingAtoms):
                    numMaxPossibleMappings = molA_allAtomsWith
                else:
                    numMaxPossibleMappings = molB_allAtomsWith

                return numMaxPossibleMappings

            smallerMolecule = oeyMolA if (oeyMolA.NumAtoms() < oeyMolB.NumAtoms()) else oeyMolB
            numMaxPossibleMappingAtoms = smallerMolecule.NumAtoms()
            # Max possible Aromatic mappings
            numMaxPossibleAromaticMappings = _getAllMappableAtomsWith(oeyMolA=oeyMolA, oeyMolB=oeyMolB,
                                                                      numMaxPossibleMappingAtoms=numMaxPossibleMappingAtoms,
                                                                      criterium=lambda x: x.IsAromatic())

            # Max possible heavy mappings
            numMaxPossibleHeavyAtomMappings = _getAllMappableAtomsWith(oeyMolA=oeyMolA, oeyMolB=oeyMolB,
                                                                       numMaxPossibleMappingAtoms=numMaxPossibleMappingAtoms,
                                                                       criterium=lambda x: x.GetAtomicNum() > 1)

            # Max possible ring mappings
            numMaxPossibleRingMappings = _getAllMappableAtomsWith(oeyMolA=oeyMolA, oeyMolB=oeyMolB,
                                                                  numMaxPossibleMappingAtoms=numMaxPossibleMappingAtoms,
                                                                  criterium=lambda x: x.IsInRing())

            # These weights are totally arbitrary
            normalize_score = 1.0 * numMaxPossibleMappingAtoms \
                              + 0.8 * numMaxPossibleAromaticMappings \
                              + 0.5 * numMaxPossibleHeavyAtomMappings \
                              + 0.4 * numMaxPossibleRingMappings

        score /= normalize_score  # final normalize score

    return score
