.. _alchemical-network-model:

Alchemical Networks: Representation of a Simulation
===================================================

The goal of the setup stage is to create an :class:`.AlchemicalNetwork`,
which contains all the information needed for a campaign of simulations.
This section will describe the composition of the achemical network,
including describing the OpenFE objects that describe chemistry, as well as
alchemical transformations.

Like any network, the :class:`.AlchemicalNetwork` can be described in terms
of nodes and edges between nodes. The nodes are :class:`.ChemicalSystem`\ s,
which describe the specific molecules involved. The edges are
:class:`.Transformation` objects, which carry all the information about how
the simulation is to be performed.

In practice, nodes must be associated with a transformation in order to be
relevant in an alchemical network; that is, there are no disconnected nodes.
This means that the alchemical network can be fully described by just the
edges (which contain information on the nodes they connect). Note that this
does not mean that the entire network must be fully connected -- just that
there are no solitary nodes.

Each :class:`.Transformation` represents everything that is needed to
calculate the free energy differences between the two
:class:`.ChemicalSystem`\ s that are the nodes for that edge. In addition to
containing the information for each :class:`.ChemicalSystem`, the
:class:`.Transformation` also contains a :class:`.Protocol` and, when
relevant, atom mapping information for alchemical transformations.

A :class:`.ChemicalSystem` is made up of one or more ``ChemicalComponent``\
s. Each component represents a conceptual part of the total molecular
system. A ligand would be represented by a :class:`.SmallMoleculeComponent`.
A protein would be a :class:`.ProteinComponent`. The solvent to be added is
represented as a :class:`.SolventComponent`. This allows us to easily
identify what is changing between two nodes -- for example, a relative
binding free energy (RBFE) edge for ligand binding would have the same
solvent and protein components, but different ligand components.

The :class:`.Protocol` object describes how the simulation should be run.
This includes choice of algorithm, as well as specific settings for the
edge. Each protocol has its own :class:`.Settings` subclass, which contains
all the settings relevant for that protocol.

.. TODO where to find details on settings
