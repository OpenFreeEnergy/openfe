# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""
The MCS class from Lomap shamelessly wrapped and used here to match our API.

"""
from lomap import mcs as lomap_mcs

from .ligandatommapper import LigandAtomMapper


class LomapAtomMapper(LigandAtomMapper):
    time: int
    threed: bool
    max3d: float

    def __init__(self, time: int = 20, threed: bool = True,
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
          if true, positional info is used to choose between symmetrically
          equivalent mappings and prune the mapping, default True
        max3d : float, optional
          maximum discrepancy in Angstroms between atoms before mapping is not
          allowed, default 1000.0, which effectively trims no atoms
        """
        self.time = time
        self.threed = threed
        self.max3d = max3d

    def _mappings_generator(self, molA, molB):
        try:
            mcs = lomap_mcs.MCS(molA.to_rdkit(), molB.to_rdkit(), time=self.time,
                                threed=self.threed, max3d=self.max3d)
        except ValueError:
            # if no match found, Lomap throws ValueError, so we just yield
            # generator with no contents
            return

        mapping_string = mcs.all_atom_match_list()
        # lomap spits out "1:1,2:2,...,x:y", so split around commas,
        # then colons and coerce to ints
        mapping_dict = dict((map(int, v.split(':'))
                             for v in mapping_string.split(',')))

        yield mapping_dict
        return
