# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import abc
from typing import Iterable

from gufe import SmallMoleculeComponent
from . import LigandAtomMapping
import gufe


class LigandAtomMapper(gufe.AtomMapper):
    """Suggests AtomMappings for a pair of :class:`SmallMoleculeComponent`s.

    Subclasses will typically implement the ``_mappings_generator`` method,
    which returns an iterable of :class:`.LigandAtomMapping` suggestions.
    """
    _no_element_changes: bool = False #TODO: to be removed
    
    @abc.abstractmethod
    def _mappings_generator(self,
                            componentA: SmallMoleculeComponent,
                            componentB: SmallMoleculeComponent
                            ) -> Iterable[dict[int, int]]:
        """
        Suggest mapping options for the input molecules.

        Parameters
        ----------
        componentA, componentB : rdkit.Mol
            the two molecules to create a mapping for

        Returns
        -------
        Iterable[dict[int, int]] :
            an iterable over proposed mappings from componentA to componentB
        """
        ...

    def suggest_mappings(self, componentA: SmallMoleculeComponent,
                         componentB: SmallMoleculeComponent
    ) -> Iterable[LigandAtomMapping]:
        """
        Suggest :class:`.LigandAtomMapping` options for the input molecules.

        Parameters
        ---------
        componentA, componentB : :class:`.SmallMoleculeComponent`
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

        for map_dct in self._mappings_generator(componentA, componentB):

            #TODO: this is a temporary Code snippet - avoids element changes - START
            if self._no_element_changes:
                filtered_map_dct = {}
                for i, j in map_dct.items():
                    atomA = componentA.to_rdkit().GetAtomWithIdx(i)
                    atomB = componentB.to_rdkit().GetAtomWithIdx(j)
                    if atomA.GetAtomicNum() == atomB.GetAtomicNum():
                        filtered_map_dct[i]= j

                if len(filtered_map_dct) == 0:
                    raise ValueError("Could not map ligands - Element Changes are not allowed currently.")
                
                map_dct = filtered_map_dct
            #TODO: this is a temporary Code snippet - avoids element changes - END

            yield LigandAtomMapping(componentA, componentB, map_dct)
