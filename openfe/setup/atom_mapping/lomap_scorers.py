# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from collections import defaultdict
from lomap import dbmol as _dbmol
from lomap import mcs as lomap_mcs
import math
from rdkit import Chem

from . import LigandAtomMapping

DEFAULT_ANS_DIFFICULTY = {
    # H to element - not sure this has any effect currently
    1: {9: 0.5, 17: 0.25, 35: 0, 53: -0.5},
    # O to element - methoxy to Cl/Br is easier than expected
    8: {17: 0.85, 35: 0.85},
    # F to element
    9: {17: 0.5, 35: 0.25, 53: 0},
    # Cl to element
    17: {35: 0.85, 53: 0.65},
    # Br to element
    35: {53: 0.85},
}


def ecr_score(mapping: LigandAtomMapping) -> float:
    molA = mapping.componentA.to_rdkit()
    molB = mapping.componentB.to_rdkit()

    return _dbmol.ecr(molA, molB)


def mcsr_score(mapping: LigandAtomMapping, beta: float = 0.1) -> float:
    """Maximum command substructure rule

    This rule is defined as::

        mcsr = exp( - beta * (n1 + n2 - 2 * n_common))

    Where n1 and n2 are the number of atoms in each molecule, and n_common
    the number of atoms in the MCS.

    Giving a value in the range [0, 1.0], with 1.0 being complete agreement
    """
    molA = mapping.componentA.to_rdkit()
    molB = mapping.componentB.to_rdkit()
    molA_to_molB = mapping.componentA_to_componentB

    n1 = molA.GetNumHeavyAtoms()
    n2 = molB.GetNumHeavyAtoms()
    # get heavy atom mcs count
    n_common = 0
    for i, j in molA_to_molB.items():
        if (molA.GetAtomWithIdx(i).GetAtomicNum() != 1
                and molB.GetAtomWithIdx(j).GetAtomicNum() != 1):
            n_common += 1

    mcsr = math.exp(-beta * (n1 + n2 - 2 * n_common))

    return mcsr


def mncar_score(mapping: LigandAtomMapping, ths: int = 4) -> float:
    """Minimum number of common atoms rule

    Parameters
    ----------
    ths : int
      the minimum number of atoms to share
    """
    molA = mapping.componentA.to_rdkit()
    molB = mapping.componentB.to_rdkit()
    molA_to_molB = mapping.componentA_to_componentB

    n1 = molA.GetNumHeavyAtoms()
    n2 = molB.GetNumHeavyAtoms()
    n_common = 0
    for i, j in molA_to_molB.items():
        if (molA.GetAtomWithIdx(i).GetAtomicNum() != 1
                and molB.GetAtomWithIdx(j).GetAtomicNum() != 1):
            n_common += 1

    ok = (n_common > ths) or (n1 < ths + 3) or (n2 < ths + 3)

    return 1.0 if ok else 0.0


def tmcsr_score(self, mapping: LigandAtomMapping):
    raise NotImplementedError


def atomic_number_score(mapping: LigandAtomMapping, beta=0.1,
                        difficulty=None) -> float:
    """A score on the elemental changes happening in the mapping

    For each transmuted atom, a mismatch score is summed, according to the
    difficulty scores (see difficult parameter).  The final score is then
    given as:

    score = exp(-beta * mismatch)

    Parameters
    ----------
    mapping : LigandAtomMapping
    beta : float, optional
      scaling factor for this rule, default 0.1
    difficulty : dict, optional
      a dict of dicts, mapping atomic number of one species, to another,
      to a mismatch in the identity of these elements.  1.0 indicates two
      elements are considered interchangeable, 0.0 indicates two elements
      are incompatible, a default of 0.5 is used.
      The scores in openfe.setup.lomap_mapper.DEFAULT_ANS_DIFFICULT are
      used by default

    Returns
    -------
    score : float
    """
    molA = mapping.componentA.to_rdkit()
    molB = mapping.componentB.to_rdkit()
    molA_to_molB = mapping.componentA_to_componentB

    if difficulty is None:
        difficulty = DEFAULT_ANS_DIFFICULTY

    nmismatch = 0
    for i, j in molA_to_molB.items():
        atom_i = molA.GetAtomWithIdx(i)
        atom_j = molB.GetAtomWithIdx(j)

        n_i = atom_i.GetAtomicNum()
        n_j = atom_j.GetAtomicNum()

        if n_i == n_j:
            continue
        elif n_i == 1 or n_j == 1:  # ignore hydrogen switches?
            continue

        try:
            ij = difficulty[n_i][n_j]
        except KeyError:
            ij = -1
        try:
            ji = difficulty[n_j][n_i]
        except KeyError:
            ji = -1
        diff = max(ij, ji)
        if diff == -1:
            diff = 0.5

        nmismatch += 1 - diff

    return math.exp(-beta * nmismatch)


def hybridization_score(mapping: LigandAtomMapping, beta=0.15) -> float:
    """

    Score calculated as:

    math.exp(-beta * nmismatch)

    Parameters
    ----------
    mapping : LigandAtomMapping
    beta : float, optional
      default 0.15

    Returns
    -------
    score : float
    """
    mol1 = mapping.componentA.to_rdkit()
    mol2 = mapping.componentB.to_rdkit()
    molA_to_molB = mapping.componentA_to_componentB

    nmismatch = 0
    for i, j in molA_to_molB.items():
        atom_i = mol1.GetAtomWithIdx(i)
        atom_j = mol2.GetAtomWithIdx(j)

        if atom_i.GetAtomicNum() == 1 or atom_j.GetAtomicNum() == 1:
            # skip hydrogen changes
            continue

        hyb_i = lomap_mcs.atom_hybridization(atom_i)
        hyb_j = lomap_mcs.atom_hybridization(atom_j)

        mismatch = hyb_i != hyb_j
        # Allow Nsp3 to match Nsp2, otherwise guanidines etc become painful
        if (atom_i.GetAtomicNum() == 7 and atom_j.GetAtomicNum() == 7 and
                hyb_i in [2, 3] and hyb_j in [2, 3]):
            mismatch = False

        if mismatch:
            nmismatch += 1

    return math.exp(- beta * nmismatch)


def sulfonamides_score(mapping: LigandAtomMapping, beta=0.4) -> float:
    """Checks if a sulfonamide appears and disallow this.

    Returns math.exp(- beta) if this happens, else 1.0
    """
    molA = mapping.componentA.to_rdkit()
    molB = mapping.componentB.to_rdkit()
    molA_to_molB = mapping.componentA_to_componentB

    def has_sulfonamide(mol):
        return mol.HasSubstructMatch(Chem.MolFromSmarts('S(=O)(=O)N'))

    # create "remainders" of both molA and molB
    remA = Chem.EditableMol(molA)
    # this incremental deletion only works when we go from high to low,
    # as atoms are reindexed as we delete
    for i, j in sorted(molA_to_molB.items(), reverse=True):
        if (molA.GetAtomWithIdx(i).GetAtomicNum() !=
                molB.GetAtomWithIdx(j).GetAtomicNum()):
            continue
        remA.RemoveAtom(i)
    # loop molB separately, sorted by A indices doesn't necessarily sort
    # the B indices too, so these loops are in different orders
    remB = Chem.EditableMol(molB)
    for i, j in sorted(molA_to_molB.items(), key=lambda x: x[1],
                       reverse=True):
        if (molA.GetAtomWithIdx(i).GetAtomicNum() !=
                molB.GetAtomWithIdx(j).GetAtomicNum()):
            continue
        remB.RemoveAtom(j)

    if has_sulfonamide(remA.GetMol()) or has_sulfonamide(remB.GetMol()):
        return math.exp(-beta)
    else:
        return 1.0


def heterocycles_score(mapping: LigandAtomMapping, beta=0.4) -> float:
    """Checks if a heterocycle is formed from a -H

    Pyrrole, furan and thiophene *are* pemitted however

    Returns math.exp(-beta) if this happens, else 1.0
    """
    molA = mapping.componentA.to_rdkit()
    molB = mapping.componentB.to_rdkit()
    molA_to_molB = mapping.componentA_to_componentB

    def creates_heterocyle(mol):
        # these patterns are lifted from lomap2 repo
        return (mol.HasSubstructMatch(
            Chem.MolFromSmarts('[n]1[c,n][c,n][c,n][c,n][c,n]1'))
            or
            mol.HasSubstructMatch(
            Chem.MolFromSmarts('[o,n,s]1[n][c,n][c,n][c,n]1'))
            or
            mol.HasSubstructMatch(
            Chem.MolFromSmarts('[o,n,s]1[c,n][n][c,n][c,n]1')))

    # create "remainders" of both molA and molB
    # create "remainders" of both molA and molB
    remA = Chem.EditableMol(molA)
    # this incremental deletion only works when we go from high to low,
    # as atoms are reindexed as we delete
    for i, j in sorted(molA_to_molB.items(), reverse=True):
        if (molA.GetAtomWithIdx(i).GetAtomicNum() !=
                molB.GetAtomWithIdx(j).GetAtomicNum()):
            continue
        remA.RemoveAtom(i)
    # loop molB separately, sorted by A indices doesn't necessarily sort
    # the B indices too, so these loops are in different orders
    remB = Chem.EditableMol(molB)
    for i, j in sorted(molA_to_molB.items(), key=lambda x: x[1],
                       reverse=True):
        if (molA.GetAtomWithIdx(i).GetAtomicNum() !=
                molB.GetAtomWithIdx(j).GetAtomicNum()):
            continue
        remB.RemoveAtom(j)

    if (creates_heterocyle(remA.GetMol()) or
            creates_heterocyle(remB.GetMol())):
        return math.exp(- beta)
    else:
        return 1.0


def transmuting_methyl_into_ring_score(mapping: LigandAtomMapping,
                                       beta=0.1, penalty=6.0) -> float:
    """Penalises having a non-mapped ring atoms become a non-ring

    This score would for example penalise R-CH3 to R-Ph where R is the same
    mapped atom and both CH3 and Ph are unmapped. Does not penalise R-H to R-Ph.
    If any atoms trigger the rule returns a score of::

      exp(-1 * beta * penalty)

    Parameters
    ----------
    mapping : LigandAtomMapping
    beta : float
    penalty : float

    Returns
    -------
    score : float
    """
    molA = mapping.componentA.to_rdkit()
    molB = mapping.componentB.to_rdkit()
    # filter out hydrogens from mapping
    # this matches the current implementation in Lomap
    molA_to_molB = {}
    for i, j in mapping.componentA_to_componentB.items():
        if molA.GetAtomWithIdx(i).GetAtomicNum() == 1:
            continue
        if molB.GetAtomWithIdx(j).GetAtomicNum() == 1:
            continue
        molA_to_molB[i] = j

    ringbreak = False
    for i, j in molA_to_molB.items():
        atomA = molA.GetAtomWithIdx(i)

        for bA in atomA.GetBonds():
            otherA = bA.GetOtherAtom(atomA)
            if otherA.GetAtomicNum() == 1:
                continue
            if otherA.GetIdx() in molA_to_molB:
                # if other end of bond in core, ignore
                continue

            # try and find the corresponding atom in molecule B
            atomB = molB.GetAtomWithIdx(j)
            for bB in atomB.GetBonds():
                otherB = bB.GetOtherAtom(atomB)
                if otherB.GetAtomicNum() == 1:
                    continue
                if otherB.GetIdx() in molA_to_molB.values():
                    continue

                if otherA.IsInRing() ^ otherB.IsInRing():
                    ringbreak = True

    if not ringbreak:
        return 1.0
    else:
        return math.exp(- beta * penalty)


def transmuting_ring_sizes_score(mapping: LigandAtomMapping) -> float:
    """Checks if mapping alters a ring size"""
    molA = mapping.componentA.to_rdkit()
    molB = mapping.componentB.to_rdkit()
    molA_to_molB = mapping.componentA_to_componentB

    def gen_ringdict(mol):
        # maps atom idx to ring sizes
        ringinfo = mol.GetRingInfo()

        idx_to_ringsizes = defaultdict(list)
        for r in ringinfo.AtomRings():
            for idx in r:
                idx_to_ringsizes[idx].append(len(r))
        return idx_to_ringsizes

    # generate ring size dicts
    ringdictA = gen_ringdict(molA)
    ringdictB = gen_ringdict(molB)

    is_bad = False
    # check first degree neighbours of core atoms to see if their ring
    # sizes are the same
    for i, j in molA_to_molB.items():
        atomA = molA.GetAtomWithIdx(i)

        for bA in atomA.GetBonds():
            otherA = bA.GetOtherAtom(atomA)
            if otherA.GetIdx() in molA_to_molB:
                # if other end of bond in core, ignore
                continue
            # otherA is an atom not in the mapping, but bonded to an
            # atom in the mapping
            if not otherA.IsInRing():
                continue

            # try and find the corresponding atom in molecule B
            atomB = molB.GetAtomWithIdx(j)
            for bB in atomB.GetBonds():
                otherB = bB.GetOtherAtom(atomB)
                if otherB.GetIdx() in molA_to_molB.values():
                    continue
                if not otherB.IsInRing():
                    continue

                # ringdict[idx] will give the list of ringsizes for an atom
                if set(ringdictA[otherA.GetIdx()]) != set(
                        ringdictB[otherB.GetIdx()]):
                    is_bad = True

    return 0.1 if is_bad else 1.0


def default_lomap_score(mapping: LigandAtomMapping) -> float:
    """The default score function from Lomap2

    Parameters
    ----------
    mapping : LigandAtomMapping
      the mapping function to score

    Returns
    -------
    score : float
       A rating of how good this mapping is, from 0.0 (terrible) to 1.0 (great).
       This score is a combination of many rules combined and considers factors such as the
       number of heavy atoms in common, if ring sizes are changed or rings are broken,
       or if other alchemically unwise transformations are attempted.
    """
    score = math.prod((
        ecr_score(mapping),
        mncar_score(mapping),
        mcsr_score(mapping),
        atomic_number_score(mapping),
        hybridization_score(mapping),
        sulfonamides_score(mapping),
        heterocycles_score(mapping),
        transmuting_methyl_into_ring_score(mapping),
        transmuting_ring_sizes_score(mapping)
    ))

    return score
