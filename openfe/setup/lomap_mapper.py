# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""
The MCS class from Lomap shamelessly wrapped and used here to match our API.

"""
from typing import Dict, TypeVar
from lomap import mcs as lomap_mcs
import math


from . import LigandAtomMapper


class LomapAtomMapper(LigandAtomMapper):
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
            mcs = lomap_mcs.MCS(mapping.mol1.rdkit, mapping.mol2.rdkit,
                                self.time, threed=self.threed,
                                max3d=self.max3d)
            self._mcs_cache[mapping] = mcs
        return mcs

    @staticmethod
    def mcsr_score(mapping: AtomMapping, beta: float = 0.1):
        """Maximum command substructure rule

        This rule was originally defined as::

        mcsr = exp( - beta * (n1 + n2 - 2 * n_common))

        Where n1 and n2 are the number of atoms in each molecule, and n_common
        the number of atoms in the MCS.

        Giving a value in the range [0, 1.0], with 1.0 being complete agreement

        This is turned into a score by simply returning (1-mcsr)
        """
        n1 = mapping.mol1.rdkit.GetNumAtoms()
        n2 = mapping.mol2.rdkit.GetNumAtoms()
        n_common = len(mapping.mol1_to_mol2)

        mcsr = math.exp(-beta * (n1 + n2 - 2 * n_common))

        return 1 - mcsr

    @staticmethod
    def mcnar_score(mapping: AtomMapping, ths: int = 4):
        """Minimum number of common atoms rule


        """
        n1 = mapping.mol1.rdkit.GetNumHeavyAtoms()
        n2 = mapping.mol2.rdkit.GetNumHeavyAtoms()
        n_common = len(mapping.mol1_to_mol2)

        ok = (n_common > ths) or (n1 < ths + 3) or (n2 < ths + 3)

        return 0.0 if ok else math.inf

    def tmcsr_score(self, mapping: AtomMapping):
        raise NotImplementedError

    def atomic_number_score(self, mapping: AtomMapping):
        mcs = self._get_mcs(mapping)
        return 1 - mcs.atomic_number_rule()

    def hybridization_score(self, mapping: AtomMapping, penalty=1.5):
        mcs = self._get_mcs(mapping)
        return 1 - mcs.hybridization_score(penalty_score=penalty)

    def sulfonamides_score(self, mapping: AtomMapping, penalty=4):
        mcs = self._get_mcs(mapping)
        return 1 - mcs.sulfonamides_rule(penalty)

    def heterocycles_score(self, mapping: AtomMapping, penalty=4):
        mcs = self._get_mcs(mapping)
        return 1 - mcs.heterocycles_rule(penalty)

    def transmuting_methyl_into_ring_score(self, mapping: AtomMapping,
                                           penalty=6):
        mcs = self._get_mcs(mapping)
        return 1 - mcs.transmuting_methyl_into_ring_rule(penalty)

    def transmuting_ring_sizes_score(self, mapping: AtomMapping):
        mcs = self._get_mcs(mapping)
        return 1 - mcs.transmuting_ring_sizes_rule()
