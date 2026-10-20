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
   :widths: 28 20 34 18

   * - Scientific problem
     - Protocol to consider
     - Important considerations
     - Relative cost
   * - Compare binding affinities of structurally related ligands.
     - :ref:`Relative Hybrid Topology
       <userguide_relative_hybrid_topology_protocol>`
     - Requires a suitable atom mapping. Mappings that break or
       form bonds are not supported.
     - Lowest per edge
   * - Compare binding affinities of chemically diverse ligands, including
       scaffold hops or ligand pairs without a suitable common core.
     - :ref:`Separated Topologies <userguide_septop_protocol>`
     - Does not require an atom mapping. Requires well-defined
       binding poses for selecting suitable restraints.
     - Higher per edge (more λ windows, longer simulation time)
   * - Compare binding affinities of ligands that differ in net charge.
     - :ref:`Relative Hybrid Topology
       <userguide_relative_hybrid_topology_protocol>`
     - The only protocol that currently supports net-charge-changing
       transformations.
     - Higher per edge (more λ windows, longer simulation time)
   * - Calculate the absolute binding free energy of an individual ligand.
     - :ref:`Absolute Binding <userguide_abfe_protocol>`
     - Requires a well-defined binding pose for selecting
       suitable restraints.
     - Higher per ligand (more λ windows, longer simulation time)

Costs are relative, using default settings. To see the exact settings
for your installed version, inspect the protocol's ``default_settings()``.


Domain of applicability
-----------------------

OpenFE's binding free energy protocols are primarily designed
to calculate the noncovalent binding of small molecules to proteins.
Not all molecular systems or alchemical transformations are supported.

System types and transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The table below summarises which molecular systems and alchemical
transformations can be studied using OpenFE's binding free energy
protocols.

Support depends on the selected protocol, force fields, and
system preparation. **openfe** uses `openmmforcefields
<https://github.com/openmm/openmmforcefields>`_ for system
parameterization. The force fields can be configured through
``forcefield_settings``
(:class:`~openfe.protocols.openmm_utils.omm_settings.OpenMMSystemGeneratorFFSettings`).

Evidence is classified as **Benchmarked** (published quantitative
comparison with experimental data), **Internally tested**
(unpublished testing by the OpenFE team), or **No published
evidence identified**.

.. list-table:: System types and transformations
   :header-rows: 1
   :widths: 12 20 20 18 30

   * - Category
     - System or transformation
     - Supported by
     - Evidence
     - Notes

   * - Target
     - Soluble proteins
     - All binding protocols
     - **Benchmarked:** `Baumann, Horton et al. (2026) <https://doi.org/10.1021/acs.jcim.6c00089>`_
       (Relative Hybrid Topology); **Internally tested** (SepTop and ABFE)
     - Standard application for noncovalent protein-ligand
       binding.

   * - Target
     - Membrane proteins
     - All binding protocols
     - **Internally tested**
     - Requires a prepared, solvated and pre-equilibrated
       system with valid periodic box vectors.

   * - Complex
     - Ternary complexes (molecular glues)
     - Relative Hybrid Topology (benchmarked)
     - **Benchmarked:** `Lukauskis et al. (2025) <https://doi.org/10.26434/chemrxiv-2025-tb29n>`_
     - Applicability of other protocols has not been tested yet.

   * - Target
     - Nucleic acid targets
     - Depends on force field compatibility
     - No published evidence identified
     - Requires suitable nucleic acid force fields.

   * - Target
     - Metalloproteins
     - Depends on the force field model
     - No published OpenFE validation identified
     - Standard nonbonded models may not accurately describe
       metal coordination, particularly when the ligand
       coordinates directly to the metal. See
       `Melse et al. (2023) <https://doi.org/10.1002/jcc.27052>`_.

   * - Target
     - Proteins with modified residues
     - Depends on force field support
     - **Benchmarked:** `Baumann, Horton et al. (2026) <https://doi.org/10.1021/acs.jcim.6c00089>`_
     - Parameters for phosphorylated residues are included in the
       default force fields. Input structures must use matching
       residue names (e.g. ``SEP``, ``TPO``) and the required
       ``CONECT`` records. Other modified residues may require
       additional force field parameters.

   * - Ligand
     - Peptide ligands
     - Depends on parameterization and system setup
     - No published evidence identified
     -

   * - Ligand
     - Macrocycles
     - Protocol-dependent
     - **Internally tested**
     - Relative Hybrid Topology requires bond-preserving atom mappings.
       Ring-opening, ring-closing, and some ring-size-changing
       transformations are therefore not supported. SepTop may be an
       alternative for comparing such ligands.

   * - Ligand
     - Covalently bound ligands
     - Not supported
     -
     - The standard binding free energy protocols do not
       support ligands covalently bonded to the receptor.

   * - Transformation
     - Net-charge-changing ligand transformations
     - Relative Hybrid Topology
     - **Benchmarked:** `Baumann, Horton et al. (2026) <https://doi.org/10.1021/acs.jcim.6c00089>`_
       (Relative Hybrid Topology)
     - Requires explicit charge correction. Charge changes greater
       than one elementary charge are not supported.

   * - Transformation
     - Protein mutations
     - Not supported
     -
     - Alchemical transformations of protein residues
       are not implemented by the **openfe** protocols, but available through the community developed protocols at `feflow
       <https://github.com/OpenFreeEnergy/feflow>`_.

General methodological limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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