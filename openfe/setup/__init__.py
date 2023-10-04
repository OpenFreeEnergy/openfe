# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Set up a free energy campaign by preparing an :class:`.AlchemicalNetwork`.

Setup consists of preparing an :class:`.AlchemicalNetwork`, which contains a
collection of :class:`.Transformation` objects. Each ``Transformation``
describes an alchemic mutation from one chemical system to another, including
full descriptions of both chemical systems as well as the simulation protocol
used to compute the change. Descriptions of chemical systems and their
components are found in the :mod:`.system` module.

An ``AlchemicalNetwork`` is usually prepared from a :class:`.LigandNetwork`,
which is an undirected graph of small molecules. Each edge represents a
yet-to-be-calculated mutation including an atom mapping, but without specifying
a simulation protocol or explicit chemical system. A ``LigandNetwork`` is
created by a planner function from the :mod:`.ligand_network_planning` module.
A planner may generate the network layout automatically, load it from another
tool, or require the user to specify it by hand. Most planners generate atom
mappings automatically by optimising over a scorer function
from :mod:`.lomap_scorers` or :mod:`.perses_scorers`; these and other atom
mapping utilities are found in the :mod:`.atom_mapping` module.

An ``AlchemicalNetwork`` additionally needs a :class:`Protocol<openfe.protocols.Protocol>`
for each transformation, though it usually uses
the same protocol for all transformations. A protocol describes and implements
the process of computing the free energy from the two chemical systems
described by a transformation.

"""


from .atom_mapping import (
    LigandAtomMapping,
    LigandAtomMapper,
    LomapAtomMapper,
    lomap_scorers,
    PersesAtomMapper,
    perses_scorers,
)

from gufe import (
    LigandNetwork,
    Transformation,
    AlchemicalNetwork,
)
from . import ligand_network_planning, atom_mapping, system

from .alchemical_network_planner import (
    RHFEAlchemicalNetworkPlanner,
    RBFEAlchemicalNetworkPlanner,
)

__all__ = [
    "atom_mapping",
    "system",
    "ligand_network_planning",
    "alchemical_network_planner",
    "LigandNetwork",
    "Transformation",
    "AlchemicalNetwork",
]
