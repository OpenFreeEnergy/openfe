# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Iterable, Dict

from . import LigandAtomMapping, LigandMolecule
from ..utils.errors import ABSTRACT_ERROR_STRING


class LigandAtomMapper:
    """Suggests AtomMappings for a pair of :class:`LigandMolecule`s.

    Subclasses will typically implement the ``_mappings_generator`` method,
    which returns an iterable of :class:`.LigandAtomMapping` suggestions.
    """
    def _mappings_generator(self, mol1, mol2) -> Iterable[Dict[int, int]]:
        """
        Suggest mapping options for the input molecules.

        Parameters
        ----------
        mol1, mol2 : rdkit.Mol
            the two molecules to create a mapping for

        Returns
        -------
        Iterable[Dict[int, int]] :
            an iterable over proposed mappings from mol1 to mol2
        """
        raise NotImplementedError(ABSTRACT_ERROR_STRING.format(
            cls=self.__class__.__name__,
            func='_mappings_generator'
        ))

    def suggest_mappings(
        self, mol1: LigandMolecule, mol2: LigandMolecule
    ) -> Iterable[LigandAtomMapping]:
        """
        Suggest :class:`.LigandAtomMapping` options for the input molecules.

        Parameters
        ---------
        mol1, mol2 : :class:`.LigandMolecule`
            the two molecules to create a mapping for

        Returns
        -------
        Iterable[LigandAtomMapping] :
            an iterable over proposed mappings
        """
        # For this base class, implementation is redundant with
        # _mappings_generator. However, we keep it separate so that abstract
        # subclasses of this can customize suggest_mappings while always
        # maintaining the consistency that concrete implementations must
        # implement _mappings_generator.
        for map_dct in self._mappings_generator(mol1.to_rdkit(),
                                                mol2.to_rdkit()):
            yield LigandAtomMapping(mol1, mol2, map_dct)
