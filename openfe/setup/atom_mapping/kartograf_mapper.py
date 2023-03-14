from typing import Iterable, Dict

from gufe import SmallMoleculeComponent
from kartograf import kartograf_atom_mapper

from .ligandatommapper import LigandAtomMapper


class KartografAtomMapper(LigandAtomMapper, kartograf_atom_mapper):
    """This is a Interface for wally to seperate the imports/dependencies.
    """
    def _mappings_generator(
        self,
        molA: SmallMoleculeComponent,
        molB: SmallMoleculeComponent,
    ) -> Iterable[Dict[int, int]]:
        mapping = self.get_mapping(molA._rdkit, molB._rdkit)
        yield mapping