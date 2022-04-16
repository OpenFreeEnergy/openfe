# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""
The MCS class from Lomap shamelessly wrapped and used here to match our API.

"""
from lomap import mcs as lomap_mcs
import math
import sys
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
     from typing_extensions import TypeAlias

from . import LigandAtomMapper, LigandAtomMapping

Lomap_MCS: TypeAlias = lomap_mcs.MCS

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


class LomapAtomMapper(LigandAtomMapper):
    time: int
    threed: bool
    max3d: float

    def __init__(self, time: int = 20, threed: bool = False,
                 max3d: float = 1000.0):
        """Wraps the MCS atom mapper from Lomap.

        Kwargs are passed directly to the MCS class from Lomap for each mapping
        created

        Parameters
        ----------
        time : int, optional
          timeout of MCS algorithm, passed to RDKit
          default 20
        threed : bool, optional
          if true, positional info is used to choose between symmetry
          equivalent mappings, default False
        max3d : float, optional
          maximum discrepancy in Angstroms between atoms before mapping is not
          allowed, default 1000.0, which effectively trims no atoms
        """
        self.time = time
        self.threed = threed
        self.max3d = max3d
        self._mcs_cache = {}

    def _mappings_generator(self, molA, molB):
        try:
            mcs = lomap_mcs.MCS(molA, molB, time=self.time,
                                threed=self.threed, max3d=self.max3d)
        except ValueError:
            # if no match found, Lomap throws ValueError, so we just yield
            # generator with no contents
            return
        # TODO: Once Lomap scorers exist, we'll want to keep a cache of
        #       these mcs objects ({mapping: mcs}), so we can later query the
        #       mcs that made a particular mapping to retrieve scores.

        mapping_string = mcs.all_atom_match_list()
        # lomap spits out "1:1,2:2,...,x:y", so split around commas,
        # then colons and coerce to ints
        mapping_dict = dict((map(int, v.split(':'))
                             for v in mapping_string.split(',')))

        yield mapping_dict
        return

    def _get_mcs(self, mapping):
        # get mcs from cache, else create and place into cache
        try:
            mcs = self._mcs_cache[mapping]
        except KeyError:
            mcs = lomap_mcs.MCS(mapping.molA.to_rdkit(), mapping.molB.to_rdkit(),
                                self.time, threed=self.threed,
                                max3d=self.max3d)
            self._mcs_cache[mapping] = mcs
        return mcs

    @staticmethod
    def mcsr_score(mapping: LigandAtomMapping, beta: float = 0.1):
        """Maximum command substructure rule

        This rule was originally defined as::

        mcsr = exp( - beta * (n1 + n2 - 2 * n_common))

        Where n1 and n2 are the number of atoms in each molecule, and n_common
        the number of atoms in the MCS.

        Giving a value in the range [0, 1.0], with 1.0 being complete agreement

        This is turned into a score by simply returning (1-mcsr)
        """
        n1 = mapping.molA.to_rdkit().GetNumAtoms()
        n2 = mapping.molB.to_rdkit().GetNumAtoms()
        n_common = len(mapping.molA_to_molB)

        mcsr = math.exp(-beta * (n1 + n2 - 2 * n_common))

        return 1 - mcsr

    @staticmethod
    def mcnar_score(mapping: LigandAtomMapping, ths: int = 4):
        """Minimum number of common atoms rule


        """
        n1 = mapping.molA.to_rdkit().GetNumHeavyAtoms()
        n2 = mapping.molB.to_rdkit().GetNumHeavyAtoms()
        n_common = len(mapping.molA_to_molB)

        ok = (n_common > ths) or (n1 < ths + 3) or (n2 < ths + 3)

        return 0.0 if ok else float('inf')

    def tmcsr_score(self, mapping: LigandAtomMapping):
        raise NotImplementedError

    @staticmethod
    def atomic_number_score(mapping: LigandAtomMapping, beta=0.1,
                            difficulty=None):
        """A score on the elemental changes happening in the mapping

        For each transmuted atom, a mismatch score is summed, according to the
        difficulty scores (see difficult parameter).  The final score is then
        given as:

        score = 1 - exp(-beta * mismatch)

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
        if difficulty is None:
            difficulty = DEFAULT_ANS_DIFFICULTY

        mol1 = mapping.molA.to_rdkit()
        mol2 = mapping.molB.to_rdkit()

        nmismatch = 0
        for i, j in mapping.molA_to_molB.items():
            atom_i = mol1.GetAtomWithIdx(i)
            atom_j = mol2.GetAtomWithIdx(j)

            n_i = atom_i.GetAtomicNum()
            n_j = atom_j.GetAtomicNum()

            if n_i == n_j:
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

        atomic_number_rule = math.exp(-beta * nmismatch)

        return 1 - atomic_number_rule

    @staticmethod
    def hybridization_score(mapping: LigandAtomMapping, beta=0.1, penalty=1.5):
        """

        Score calculated as:

        1 - math.exp(-beta * nmismatch * penalty)

        Parameters
        ----------
        mapping : LigandAtomMapping
        beta : float, optional
          default 0.1
        penalty : float, optional
          default 1.5

        Returns
        -------
        score : float
        """
        nmismatch = 0
        mol1 = mapping.molA.to_rdkit()
        mol2 = mapping.molB.to_rdkit()

        for i, j in mapping.molA_to_molB.items():
            atom_i = mol1.GetAtomWithIdx(i)
            atom_j = mol2.GetAtomWithIdx(j)

            hyb_i = lomap_mcs.atom_hybridization(atom_i)
            hyb_j = lomap_mcs.atom_hybridization(atom_j)

            mismatch = hyb_i != hyb_j
            # Allow Nsp3 to match Nsp2, otherwise guanidines etc become painful
            if (atom_i.GetAtomicNum() == 7 and atom_j.GetAtomicNum() == 7 and
                hyb_i in [2, 3] and hyb_j in [2, 3]):
                mismatch = False

            if mismatch:
                nmismatch += 1

        hybridization_rule = math.exp(- beta * nmismatch * penalty)

        return 1 - hybridization_rule

    def sulfonamides_score(self, mapping: LigandAtomMapping, penalty=4):
        mcs = self._get_mcs(mapping)
        return 1 - mcs.sulfonamides_rule(penalty)

    def heterocycles_score(self, mapping: LigandAtomMapping, penalty=4):
        mcs = self._get_mcs(mapping)
        return 1 - mcs.heterocycles_rule(penalty)

    def transmuting_methyl_into_ring_score(self, mapping: LigandAtomMapping,
                                           penalty=6):
        mcs = self._get_mcs(mapping)
        return 1 - mcs.transmuting_methyl_into_ring_rule(penalty)

    def transmuting_ring_sizes_score(self, mapping: LigandAtomMapping):
        mcs = self._get_mcs(mapping)
        return 1 - mcs.transmuting_ring_sizes_rule()
