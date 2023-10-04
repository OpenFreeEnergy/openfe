"""
Chemical systems and their components.

A :class:`ChemicalSystem` describes a simulation box and its contents. They are
the nodes of an ``AlchemicalNetwork``; two are included in each
``Transformation`` as end states. The abstract base class :class:`Component`
defines the interface for a components of a chemical system. A ``Component``
need not refer to a single molecule; it may represent a protein of multiple
chains and their cofactors, or the entire solvent including ions.

.. admonition:: Custom ``Components`` require support from the ``Protocol``

    ``Component`` types are handled individually by each ``Protocol``; it is
    not presently possible to define a custom ``Component`` and have it work
    in an existing ``Protocol``.

"""

from gufe import (
    ChemicalSystem,
    Component,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
)

__all__ = [
    "ChemicalSystem",
    "Component",
    "ProteinComponent",
    "SmallMoleculeComponent",
    "SolventComponent",
]
