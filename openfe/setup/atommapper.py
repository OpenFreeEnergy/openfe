# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from collections.abc import Iterable
from typing import TypeVar
from openfe.setup import AtomMapping
from openfe.setup.errors import ABSTRACT_ERROR_STRING

RDKitMol = TypeVar("RDKitMol")

from openfe.setup.scorer import ScoreAnnotation


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


class ScoredAtomMapper(AtomMapper):
    """Abstract AtomMapper that calculates/caches an associated score.

    Concrete subclasses of this must implement the ``_calculate_score``
    method. Then ``ScoredAtomMapper.score`` works as a scoring function.
    Additionally, as with all AtomMappers, subclasses must implement
    ``._mappings_generator``.
    """
    def __init__(self):
        super().__init__()
        self._score_cache = {}

    def _calculate_score(self, atommapping: AtomMapping) -> float:
        """
        Calculate the score for a given AtomMapping.

        This method allows the score to be calculated and cached during
        creation of the AtomMapping.

        Parameters
        ----------
        atommapping : :class:`.AtomMapping`
            the AtomMapping to calculate the score for

        Returns
        -------
        float :
            score for the input AtomMapping
        """
        raise NotImplementedError(ABSTRACT_ERROR_STRING.format(
            cls=self.__class__.__name__,
            func='_calculate_score'
        ))

    def suggest_mappings(
        self, mol1: RDKitMol, mol2: RDKitMol
    ) -> Iterable[AtomMapping]:
        for mapping in self._mappings_generator(mol1, mol2):
            self._score_cache[mapping] = self._calculate_score(mapping)
            yield mapping

    def score(self, atommapping: AtomMapping) -> ScoreAnnotation:
        """Return the score for a given AtomMapping.

        Parameters
        ----------
        atommapping : :class:`.AtomMapping`
            the AtomMapping to calculate the score for

        Returns
        -------
        :class:`.ScoreAnnotation`
            the score and empty annotation for the input AtomMapping
        """
        try:
            score = self._score_cache[atommapping]
        except KeyError:
            score = self._calculate_score(atommapping)
        return ScoreAnnotation(score=score, annotation={})
