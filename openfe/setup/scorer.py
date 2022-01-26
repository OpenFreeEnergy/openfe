# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""
Scorers
=======

A scorer is used to determine a score (and optionally, annotations) for a
given AtomMapping.
"""

from typing import NamedTuple, Dict, Any, Union


class ScoreAnnotation(NamedTuple):
    """Container for a score from a mapping and any associated annotations.

    Parameters
    ----------
    score : float
        The score, or ``None`` if there is no score associated
    annotation : Dict[str, Any]
        Mapping of annotation label to annotation value for any annotations
    """
    score: Union[float, None]
    annotation: Dict[str, Any]


class Scorer:
    """Abstract base class for Scorers.

    To implement a subclass, you must implement the ``score`` method. If
    your ``Scorer`` only returns annotations, then return None from the
    ``score`` method. You may optionally implement the ``annotation``
    method. The default ``annotation`` returns an empty dictionary.

    Use a ``Scorer`` by calling it as a function.
    """
    def score(self, atommapping) -> Union[float, None]:
        """Calculate the score for an AtomMapping.

        Parameters
        ----------
        atommapping : AtomMapping
            AtomMapping to score

        Returns
        -------
        Union[float, None] :
            The score, or ``None`` if no score is calculated
        """
        raise NotImplementedError()

    def annotation(self, atommapping) -> Dict[str, Any]:
        """Create annotation dict for an AtomMapping.

        Parameters
        ----------
        atommapping : AtomMapping
            Atommapping to annotate

        Returns
        -------
        Dict[str, Any] :
            Mapping of annotation labels to annotation values
        """
        return {}

    def __call__(self, atommapping) -> ScoreAnnotation:
        return ScoreAnnotation(score=self.score(atommapping),
                               annotation=self.annotation(atommapping))
