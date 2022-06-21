# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""
The MCS class from Lomap shamelessly wrapped and used here to match our API.

"""

from enum import Enum
from perses.rjmc.atom_mapping import AtomMapper

from . import LigandAtomMapper

class PersesMappingType(Enum):
    best = 1
    sampled = 2
    proposed = 3
    all = 4

class PersesAtomMapper(LigandAtomMapper):
    unmap_partially_mapped_cycles: bool
    preserve_chirality: bool
    mapping_type:PersesMappingType

    __atom_mapper: AtomMapper


    def __init__(self, full_cycles_only:bool=True, preserve_chirality:bool=True,
                 mapping_type:PersesMappingType=PersesMappingType.best):
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
        self.unmap_partially_mapped_cycles = unmap_partially_mapped_cycles
        self.preserve_chirality = preserve_chirality
        self.mapping_type = mapping_type



    def _mappings_generator(self, molA, molB):
        self.__atom_mapper = AtomMapper()

        if(self.mapping_type == PersesMappingType.best):
            self.atom_mappings = self.__atom_mapper.get_best_mapping(old_mol=molA, new_mol=molB)
        elif(self.mapping_type == PersesMappingType.sampled):
            self.__atom_mapper.get_sampled_mapping(old_mol=molA, new_mol=molB)
        elif(self.mapping_type == PersesMappingType.proposed): #Not Implemented right now
            self.__atom_mapper.propose_mapping(old_mol=molA, new_mol=molB)
        elif(self.mapping_type == PersesMappingType.all):
            self.__atom_mapper.get_all_mappings(old_mol=molA, new_mol=molB)
        else:
            raise ValueError("Mapping type value error! Please chose one of the provided Enum options. Given: "str(self.mapping_type))

        mapping_dict = dict((map(int, v.split(':'))
                             for v in mapping_string.split(',')))

        yield mapping_dict
        return
from perses.rjmc.atom_mapping import AtomMapper
