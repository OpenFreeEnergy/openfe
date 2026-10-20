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
   Computes the relative binding free energy (:math:`\Delta\Delta G`) between two ligands
   by simultaneously inserting one ligand and removing the other,
   using orientational restraints to maintain their positions
   in the binding site. Unlike the hybrid topology protocol,
   it does not require an atom mapping.

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

Different binding free energy protocols are suited to different
applications. The table below provides guidance on choosing a
protocol, although more than one may be suitable for a given problem.


.. list-table:: Guidance for selecting a binding free energy protocol
   :header-rows: 1
   :widths: 34 26 40

   * - Scientific problem
     - Protocol to consider
     - Important considerations
   * - Compare binding affinities of structurally related ligands.
     - :ref:`Relative Hybrid Topology
       <userguide_relative_hybrid_topology_protocol>`
     - Requires a suitable atom mapping. Mappings that break or
       form bonds are not supported.
   * - Compare binding affinities of chemically diverse ligands,
       including scaffold hops or ligand pairs without a suitable
       common core.
     - :ref:`Separated Topologies <userguide_septop_protocol>`
     - Does not require an atom mapping. Requires well-defined
       binding poses for selecting suitable restraints.
   * - Compare binding affinities of ligands that differ in net charge.
     - :ref:`Relative Hybrid Topology
       <userguide_relative_hybrid_topology_protocol>`
     - Supports charge-changing transformations using an explicit
       charge correction.
   * - Calculate the absolute binding free energy of an individual ligand.
     - :ref:`Absolute Binding <userguide_abfe_protocol>`
     - Requires a well-defined binding pose for selecting
       suitable restraints.



Domain of applicability
-----------------------

OpenFE's binding free energy protocols are primarily designed
to calculate the noncovalent binding of small molecules to proteins.
Not all molecular systems or alchemical transformations are supported.

General scope and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Molecular system and parameterization**

**openfe** supports molecular systems that can be parameterized using
the available force fields. Systems containing non-standard
residues, post-translational modifications, or metal coordination
sites may not be supported if suitable force field parameters
are unavailable.

The binding free energy protocols do not support alchemical
transformations of protein residues, such as protein mutations, or covalently bound ligands.

**Chemical states**

The protocols use fixed protonation and tautomeric states.
Changes in the populations of these states upon binding are not
accounted for automatically.

**Sampling limitations**

Slow processes, such as large protein conformational changes,
transitions between ligand binding modes, or exchange of buried
water molecules, may not be adequately sampled within standard
simulation timescales. Such systems may require specialized
sampling approaches beyond the standard OpenFE protocols.

**System types**

OpenFE supports soluble and membrane proteins. Membrane systems
must be provided as prepared, solvated, and pre-equilibrated
structures with valid periodic box vectors.

Calculations involving peptide ligands or nucleic acid targets
may be possible, but these applications have not been validated
by the OpenFE team.

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
       * Net-charge-changing transformations require explicit
         charge correction using an alchemical water molecule.
         Charge changes greater than one are not supported.
   * - :ref:`Separated Topologies
       <userguide_septop_protocol>`
     - * Restraint selection and sampling can be challenging for
         highly flexible ligands.
       * Net-charge-changing transformations are not currently
         supported.
   * - :ref:`Absolute Binding
       <userguide_abfe_protocol>`
     - * Restraint selection and sampling can be challenging for
         highly flexible ligands.
       * Charged ligands are not currently supported.


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