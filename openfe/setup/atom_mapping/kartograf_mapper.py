from typing import Iterable, Dict

from kartograf.atom_mapping.geom_mapping import geom_mapping

from .ligandatommapper import LigandAtomMapper
from ...setup import SmallMoleculeComponent


class GeomAtomMapper(LigandAtomMapper, geom_mapping):
    """This is a Interface for wally to seperate the imports/dependencies.
    """
    def _mappings_generator(
        self,
        molA: SmallMoleculeComponent,
        molB: SmallMoleculeComponent,
    ) -> Iterable[Dict[int, int]]:
        mapping = self.get_mapping(molA._rdkit, molB._rdkit)
        yield mapping