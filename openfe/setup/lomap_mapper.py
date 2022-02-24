# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""
The MCS class from Lomap shamelessly wrapped and used here to match our API.

"""
from lomap import mcs as lomap_mcs


from . import AtomMapper


class LomapAtomMapper(AtomMapper):
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

    def _mappings_generator(self, mol1, mol2):
        try:
            mcs = lomap_mcs.MCS(mol1, mol2, time=self.time,
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
