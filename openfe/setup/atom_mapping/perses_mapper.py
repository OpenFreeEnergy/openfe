# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""
The MCS class from Perses shamelessly wrapped and used here to match our API.

"""

from openmm import unit
from openfe.utils import requires_package

try:
    from perses.rjmc.atom_mapping import AtomMapper, InvalidMappingException
except ImportError:
    pass    # Don't throw  error, will happen later

from .ligandatommapper import LigandAtomMapper


class PersesAtomMapper(LigandAtomMapper):
    allow_ring_breaking: bool
    preserve_chirality: bool
    use_positions: bool

    @requires_package("perses")
    def __init__(self, allow_ring_breaking: bool = True,
                 preserve_chirality: bool = True,
                 use_positions: bool = True,
                 coordinate_tolerance: float = 0.25 * unit.angstrom):
        """
        This class uses the perses code to facilitate the mapping of the
        atoms of two molecules to each other.

        Parameters
        ----------
        allow_ring_breaking: bool, optional
            this option checks if on only full cycles of the molecules shall
            be mapped, default: False
        preserve_chirality: bool, optional
            if mappings must strictly preserve chirality, default: True
        use_positions: bool, optional
            this option defines, if the
        coordinate_tolerance: float, optional
            tolerance on how close coordinates need to be, such they
            can be mapped, default: 0.25*unit.angstrom

        """
        self.allow_ring_breaking = allow_ring_breaking
        self.preserve_chirality = preserve_chirality
        self.use_positions = use_positions
        self.coordinate_tolerance = coordinate_tolerance

    def _mappings_generator(self, molA, molB):
        # Construct Perses Mapper
        _atom_mapper = AtomMapper(
            use_positions=self.use_positions,
            coordinate_tolerance=self.coordinate_tolerance,
            allow_ring_breaking=self.allow_ring_breaking)

        # Try generating a mapping
        try:
            _atom_mappings = _atom_mapper.get_all_mappings(
                    old_mol=molA.to_openff(), new_mol=molB.to_openff())
        except InvalidMappingException:
            _atom_mappings = []

        # Catch empty mappings here
        if (_atom_mappings is None):
            _atom_mappings = []

        # Post processing
        if self.preserve_chirality:
            for x in _atom_mappings:
                x.preserve_chirality()

        # Translate mapping objects
        if(len(_atom_mappings) > 0):
            mapping_dict = map(lambda x: x.old_to_new_atom_map, _atom_mappings)
        else:
            mapping_dict = []

        yield from mapping_dict
