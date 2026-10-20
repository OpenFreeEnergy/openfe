.. _userguide_protocols:

Protocols
=========

**openfe** provides several protocols for calculating free energy
differences, each suited to different scientific problems. This page
provides an overview of the available protocols, guidance on choosing
between them, and their main limitations.


Available binding free energy protocols
---------------------------------------

:ref:`Relative Hybrid Topology Protocol <userguide_relative_hybrid_topology_protocol>`
   Computes the relative binding free energy (:math:`\Delta\Delta G`) between two ligands by
   alchemically transforming one into the other using a hybrid
   topology defined by an atom mapping. The same protocol can also
   calculate relative hydration free energies.

:ref:`Separated Topologies Protocol <userguide_septop_protocol>`
   Computes the relative binding free energy (:math:`\Delta\Delta G`) between two ligands by simultaneously inserting
   one ligand and removing the other, using orientational restraints to keep both in the
   binding site. No atom mapping is required, so it can handle chemically diverse pairs such
   as scaffold hops.

:ref:`Absolute Binding Protocol <userguide_abfe_protocol>`
   Computes the absolute binding free energy (:math:`\Delta G`) of a single ligand by decoupling it from
   the binding site and from solvent, using orientational restraints in the complex.

Other protocols
~~~~~~~~~~~~~~~

* :doc:`Absolute Solvation Protocol <absolutesolvation>`: absolute hydration free energy of a
  small molecule.
* :doc:`Plain MD Protocol <plainmd>`: conventional (non-alchemical) molecular dynamics, for
  example to check the stability of a binding pose before running free energy calculations.


Choosing a protocol
-------------------

Protocol selection depends on the quantity you wish to calculate, the
relationship between the ligands, and the characteristics of the molecular
system. The table below provides starting points rather than definitive
recommendations: more than one protocol may be applicable to a given problem.

.. list-table:: Guidance for selecting a binding free energy protocol
   :header-rows: 1
   :widths: 34 26 40

   * - Scientific problem
     - Protocol to consider
     - Important considerations
   * - Compare binding affinities of structurally related ligands.
     - :ref:`Relative Hybrid Topology
       <userguide_relative_hybrid_topology_protocol>`
     - Requires a suitable atom mapping. Mappings that break or form bonds,
       including ring-opening transformations, are not supported.
   * - Compare binding affinities of chemically diverse ligands, including
       scaffold hops or ligand pairs without a suitable common core.
     - :ref:`Separated Topologies <userguide_septop_protocol>`
     - Does not require an atom mapping. Requires a stable, well-defined
       binding pose for both ligands, as restraint atoms are selected
       automatically after a short equilibration.
   * - Compare binding affinities of ligands that differ in net charge.
     - :ref:`Relative Hybrid Topology
       <userguide_relative_hybrid_topology_protocol>`
     - The only protocol that currently supports net-charge-changing
       transformations.
   * - Calculate the absolute binding free energy of an individual ligand.
     - :ref:`Absolute Binding <userguide_abfe_protocol>`
     - Requires a stable, well-defined binding pose, as restraint atoms are
       selected automatically after a short equilibration.


Domain of applicability
-----------------------

OpenFE's binding free energy protocols are primarily intended for
noncovalent binding of small molecules to proteins.
Applicability depends on both the molecular
system and the alchemical method being used.

General scope and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Molecular system and parameterization
"""""""""""""""""""""""""""""""""""""

**openfe** supports molecular systems that can be parameterized using
the available force fields. Systems containing non-standard
residues, post-translational modifications, or metal coordination
sites may not be supported if suitable force field parameters
are unavailable.

The binding free energy protocols do not support alchemical
transformations of protein residues, such as protein mutations, or covalently bound ligands.

Chemical states
"""""""""""""""

The protocols use fixed protonation and tautomeric states.
Changes in the populations of these states upon binding are not
accounted for automatically.

Sampling challenges
"""""""""""""""""""

Some systems may require additional sampling beyond the default
protocol settings. Examples include large receptor conformational
changes, transitions between alternative ligand binding modes, and slow exchange of
buried water molecules.

System types
""""""""""""

The binding free energy protocols support soluble proteins and
protein-membrane systems. Membrane systems require a fully prepared,
solvated, and pre-equilibrated structure with valid periodic box
vectors.

Applications involving peptide ligands or nucleic acid targets
may be possible, but their applicability depends on suitable
system preparation, parameterization, and sampling. These
applications have not been validated by the
OpenFE team.

Protocol-specific limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the general limitations described above, each
binding free energy protocol has specific restrictions arising
from its alchemical method and implementation.

.. list-table:: Protocol-specific limitations
   :header-rows: 1
   :widths: 25 75

   * - Protocol
     - Important limitations
   * - :ref:`Relative Hybrid Topology
       <userguide_relative_hybrid_topology_protocol>`
     - * Requires a suitable atom mapping. Mappings that break or
         form bonds, including ring-opening transformations, are
         not supported.
       * Net-charge-changing transformations use an alchemical water
         molecule to keep the system neutral, and typically need
         longer simulations to converge.
   * - :ref:`Separated Topologies
       <userguide_septop_protocol>`
     - * Requires a stable, well-defined binding pose for both ligands,
         as restraint atoms are selected automatically. Flexible ligands
         may sample poorly once decoupled.
       * Net-charge-changing transformations are not currently
         supported.
   * - :ref:`Absolute Binding
       <userguide_abfe_protocol>`
     - * Requires a stable, well-defined binding pose, as restraint atoms
         are selected automatically. Flexible ligands may sample poorly
         once decoupled.
       * Charged ligands are not currently supported.
       * Large protein conformational changes upon ligand binding require
         additional sampling. Unlike in relative calculations, where
         protein reorganisation partly cancels between ligands, it
         contributes directly to the absolute binding free energy.


See also
--------

* :doc:`Choose and configure a protocol </cookbook/choose_protocol>`
* :doc:`Troubleshooting simulations </guide/troubleshooting>`

.. toctree::
   :hidden:

   relativehybridtopology
   septop
   absolutebinding
   absolutesolvation
   plainmd