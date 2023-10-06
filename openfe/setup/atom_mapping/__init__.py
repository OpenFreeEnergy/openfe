"""
Mappings from atoms in one molecule to those in another.

:class:`LigandAtomMapper` defines the interface of atom mappers, and is
implemented by :class:`LomapAtomMapper` and :class:`PersesAtomMapper`.
:class:`LigandAtomMapping` is a simple container for an atom maping. It
 describes which atoms in one molecule should be transformed into which atoms
 in the other, and which atoms should be destroyed or created over the
 transformation.

The :mod:`.perses_scorers` and :mod:`.lomap_scorers` modules provide scoring
functions used for both atom mappings and transformations. A scorer takes
a :class:`..mapping.LigandAtomMapping` and possibly some other parameters and
returns a score between 0 and 1. Higher scores represent better mappings. These
scores are used by :class:`LigandNetwork<openfe.new_api.setup.LigandNetwork>`
planners to select the best mapping for a given transformation and also to
compare different transformations while optimising network topologies.

"""

from gufe import LigandAtomMapping
from .ligandatommapper import LigandAtomMapper

from .lomap_mapper import LomapAtomMapper
from .perses_mapper import PersesAtomMapper

from . import perses_scorers
from . import lomap_scorers
