from typing import Iterable, Dict

from gufe import SmallMoleculeComponent
from kartograf.atom_mapping.geom_mapper import geometric_atom_mapper

from .ligandatommapper import LigandAtomMapper


class GeomAtomMapper(LigandAtomMapper, geometric_atom_mapper):
    """This is a Interface for wally to seperate the imports/dependencies.
    """
    def _mappings_generator(
        self,
        molA: SmallMoleculeComponent,
        molB: SmallMoleculeComponent,
    ) -> Iterable[Dict[int, int]]:
        mapping = self.get_mapping(molA._rdkit, molB._rdkit)
        yield mapping