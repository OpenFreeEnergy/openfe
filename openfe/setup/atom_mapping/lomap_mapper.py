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
    element_change: bool
    seed: str
    shift: bool

    def __init__(self, *, time: int = 20, threed: bool = True,
                 max3d: float = 1000.0, element_change: bool = True,
                 seed: str = '', shift: bool = True):
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
        element_change: bool, optional
          whether to allow element changes in the mappings, default True
        seed: str, optional
          SMARTS string to use as seed for MCS searches.  When used across an entire set
          of ligands, this can create
        shift: bool, optional
          when determining 3D overlap, if to translate the two molecules MCS to minimise
          RMSD to boost potential alignment.
        """
        self.time = time
        self.threed = threed
        self.max3d = max3d
        self.element_change = element_change
        self.seed = seed
        self.shift = shift

    def _mappings_generator(self, componentA, componentB):
        try:
            mcs = lomap_mcs.MCS(componentA.to_rdkit(), componentB.to_rdkit(),
                                time=self.time,
                                threed=self.threed, max3d=self.max3d,
                                element_change=self.element_change, seed=self.seed,
                                shift=self.shift)
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
