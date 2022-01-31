# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import TypeVar, Iterable
from openfe.setup import AtomMapping
from openfe.setup.errors import ABSTRACT_ERROR_STRING

from openfe.setup.scorer import ScoreAnnotation

RDKitMol = TypeVar("RDKitMol")


class AtomMapper:
    """AtomMapper suggests AtomMappings for a pair of molecules.

    Subclasses will typically implement the ``_mappings_generator`` method,
    which returns an iterable of :class:`.AtomMapping` suggestions.
    """
    def _mappings_generator(
        self, mol1: RDKitMol, mol2: RDKitMol
    ) -> Iterable[AtomMapping]:
        """
        Suggest :class:`.AtomMapping` options for the input molecules.

        Parameters
        ----------
        mol1, mol2 : rdkit.Mol
            the two molecules to create a mapping for

        Returns
        -------
        Iterable[AtomMapping] :
            an iterable over proposed mappings
        """
        raise NotImplementedError(ABSTRACT_ERROR_STRING.format(
            cls=self.__class__.__name__,
            func='_mappings_generator'
        ))

    def suggest_mappings(
        self, mol1: RDKitMol, mol2: RDKitMol
    ) -> Iterable[AtomMapping]:
        """
        Suggest :class:`.AtomMapping` options for the input molecules.

        Parameters
        ----------
        mol1, mol2 : rdkit.Mol
            the two molecules to create a mapping for

        Returns
        -------
        Iterable[AtomMapping] :
            an iterable over proposed mappings
        """
        # For this base class, implementation is redundant with
        # _mappings_generator. However, we keep it separate so that abstract
        # subclasses of this can customize suggest_mappings while always
        # maintaining the consistency that concrete implementations must
        # implement _mappings_generator.
        for mapping in self._mappings_generator(mol1, mol2):
            yield mapping
