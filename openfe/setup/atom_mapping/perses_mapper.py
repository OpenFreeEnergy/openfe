# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""
The MCS class from Lomap shamelessly wrapped and used here to match our API.

"""

from enum import Enum

from openmm import unit

from perses.rjmc.atom_mapping import AtomMapper, AtomMapping

from .ligandatommapper import LigandAtomMapper

class PersesMappingType(Enum):
    best = 1
    sampled = 2
    proposed = 3
    all = 4


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






class PersesAtomMapper(LigandAtomMapper):
    unmap_partially_mapped_cycles: bool
    preserve_chirality: bool
    mapping_type:PersesMappingType

    _atom_mappings: AtomMapping

    __atom_mapper: AtomMapper


    def __init__(self, full_cycles_only:bool=False, preserve_chirality:bool=True, use_positions:bool=True,
                 mapping_type:PersesMappingType=PersesMappingType.all, coordinate_tolerance:float=0.25*unit.angstrom):
        """
        This class uses the perses code to facilitate the mapping of the atoms of two molecules to each other.

        Parameters
        ----------
        full_cycles_only: bool, optional
            this option checks if on only full cycles of the molecules shall be mapped, default: False
        preserve_chirality: bool, optional
             , default: True
        use_positions: bool, optional
            this option defines, if the
        mapping_type: PersesMappingType, optional
            how to calculate the mapping and amount of mappings, default: PersesMappingType.best
        coordinate_tolerance: float, optional
            tolerance on how close coordinates need to be, such they can be mapped, default: 0.25*unit.angstrom
        """

        self.unmap_partially_mapped_cycles = full_cycles_only
        self.preserve_chirality = preserve_chirality
        self.mapping_type = mapping_type
        #self.map_stength = map_strength
        self.use_positions = use_positions
        self.coordinate_tolerance = coordinate_tolerance

        self._atom_mappings = None


    def _mappings_generator(self, molA, molB):
        self.__atom_mapper = AtomMapper(use_positions=self.use_positions, coordinate_tolerance=self.coordinate_tolerance)

        #Type of mapping
        if(self.mapping_type == PersesMappingType.best):
            self._atom_mappings = self.__atom_mapper.get_best_mapping(old_mol=molA.to_openff(), new_mol=molB.to_openff())
        elif(self.mapping_type == PersesMappingType.sampled):
            self._atom_mappings = self.__atom_mapper.get_sampled_mapping(old_mol=molA.to_openff(), new_mol=molB.to_openff())
        elif(self.mapping_type == PersesMappingType.proposed): #Not Implemented right now
            self._atom_mappings = self.__atom_mapper.propose_mapping(old_mol=molA.to_openff(), new_mol=molB.to_openff())
        elif(self.mapping_type == PersesMappingType.all):
            self._atom_mappings = self.__atom_mapper.get_all_mappings(old_mol=molA.to_openff(), new_mol=molB.to_openff())
        else:
            raise ValueError("Mapping type value error! Please chose one of the provided Enum options. Given: "+str(self.mapping_type))

        #Post processing
        if(self.unmap_partially_mapped_cycles):
            self._atom_mappings.unmap_partially_mapped_cycles()
        if(self.preserve_chirality):
            self._atom_mappings.preserve_chirality()

        mapping_dict = self._atom_mappings.old_to_new_atom_map
        yield mapping_dict
        return
