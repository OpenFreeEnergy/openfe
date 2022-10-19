# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Iterable, Dict

from openfe.setup import SmallMoleculeComponent
from .ligandatommapping import LigandAtomMapping
import gufe

from openfe.utils.errors import ABSTRACT_ERROR_STRING


class LigandAtomMapper(gufe.AtomMapper):
    """Suggests AtomMappings for a pair of :class:`SmallMoleculeComponent`s.

    Subclasses will typically implement the ``_mappings_generator`` method,
    which returns an iterable of :class:`.LigandAtomMapping` suggestions.
    """

    def _mappings_generator(self,
                            molA: SmallMoleculeComponent,
                            molB: SmallMoleculeComponent
                            ) -> Iterable[Dict[int, int]]:
        """
        Suggest mapping options for the input molecules.

        Parameters
        ----------
        molA, molB : rdkit.Mol
            the two molecules to create a mapping for

        Returns
        -------
        Iterable[Dict[int, int]] :
            an iterable over proposed mappings from molA to molB
        """
        raise NotImplementedError(ABSTRACT_ERROR_STRING.format(
            cls=self.__class__.__name__,
            func='_mappings_generator'
        ))

    def suggest_mappings(
            self, molA: SmallMoleculeComponent, molB: SmallMoleculeComponent
    ) -> Iterable[LigandAtomMapping]:
        """
        Suggest :class:`.LigandAtomMapping` options for the input molecules.

        Parameters
        ---------
        molA, molB : :class:`.SmallMoleculeComponent`
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
        for map_dct in self._mappings_generator(molA, molB):
            yield LigandAtomMapping(molA, molB, map_dct)
