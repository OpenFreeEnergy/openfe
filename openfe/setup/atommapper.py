# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import TypeVar, Iterable, Dict


from . import AtomMapping, Molecule
from ..utils.errors import ABSTRACT_ERROR_STRING

RDKitMol = TypeVar("RDKitMol")


class AtomMapper:
    """AtomMapper suggests AtomMappings for a pair of molecules.

    Subclasses will typically implement the ``_mappings_generator`` method,
    which returns an iterable of :class:`.AtomMapping` suggestions.
    """
    def _mappings_generator(self, mol1, mol2) -> Iterable[Dict[int, int]]:
        """
        Suggest mapping options for the input molecules.

        Parameters
        ----------
        mol1, mol2 : Molecule
            the two molecules to create a mapping for

        Returns
        -------
        Iterable[Dict[int, int]] :
            an iterable over proposed mappings from mol1 to mol2


        Notes
        -----
        To convert the openfe hashable Molecule object to something more
        useful, use the ``as_rdkit`` decorator to convert Molecule arguments
        to rdkit (or oechem, offtk etc).
        """
        raise NotImplementedError(ABSTRACT_ERROR_STRING.format(
            cls=self.__class__.__name__,
            func='_mappings_generator'
        ))

    def suggest_mappings(
        self, mol1: Molecule, mol2: Molecule
    ) -> Iterable[AtomMapping]:
        """
        Suggest :class:`.AtomMapping` options for the input molecules.

        Parameters
        ---------
        mol1, mol2 : :class:`.Molecule`
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
        for map_dct in self._mappings_generator(mol1.rdkit, mol2.rdkit):
            yield AtomMapping(mol1, mol2, map_dct)
