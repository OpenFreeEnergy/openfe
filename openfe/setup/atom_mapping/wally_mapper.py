from typing import Iterable, Dict

from wally import wally_atom_mapper
from wally.wally_mapper import wally_mapping_algorithm

from .ligandatommapper import LigandAtomMapper
from ...setup import SmallMoleculeComponent


class WallyAtomMapper(LigandAtomMapper, wally_atom_mapper):
    """This is a Interface for wally to seperate the imports/dependencies.
    """
    def _mappings_generator(
        self,
        molA: SmallMoleculeComponent,
        molB: SmallMoleculeComponent,
    ) -> Iterable[Dict[int, int]]:
        mapping = self.get_mapping(molA._rdkit, molB._rdkit)
        yield mapping