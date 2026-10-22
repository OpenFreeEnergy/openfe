.. _userguide_protocols:

Protocols
=========

**openfe** provides several protocols for calculating free energy
differences, each suited to different scientific problems. This page
provides an overview of the available protocols, guidance on choosing
between them, and their main limitations.


Available protocols
-------------------

Relative binding free energies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Relative protocols compute the difference in binding free energy (:math:`\Delta\Delta G`) between two ligands.

:ref:`Relative Hybrid Topology Protocol <userguide_relative_hybrid_topology_protocol>` [1]_
   Transforms one ligand into the other using a hybrid
   topology defined by an atom mapping. Can also
   calculate relative hydration free energies.

:ref:`Separated Topologies Protocol <userguide_septop_protocol>` [2]_, [3]_
   Inserts one ligand while removing the other,
   using orientational restraints to maintain their positions
   in the binding site. No atom mapping required.

Absolute binding free energies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Absolute protocols compute the binding free energy (:math:`\Delta G`)
of a single ligand.

:ref:`Absolute Binding Protocol <userguide_abfe_protocol>`
   Decouples the ligand from
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
applications. The table below summarises what each protocol is best
suited for and its main limitations. More than one protocol may be
suitable for a given problem.

.. list-table:: Guidance for selecting a binding free energy protocol
   :header-rows: 1
   :widths: 16 30 38 16

   * - Protocol
     - Best suited for
     - Limitations
     - Relative cost
   * - :ref:`Relative Hybrid Topology <userguide_relative_hybrid_topology_protocol>`
     - * Structurally related ligands sharing a common core.
       * Ligands that differ in net charge (the only protocol that currently supports this).
     - * Requires a suitable atom mapping. Mappings that break or form
         bonds, such as ring opening, are not supported.
       * Net-charge-changing transformations require explicit charge
         correction; charge changes greater than one are not supported.
     - Lowest per edge; higher for net charge changes
   * - :ref:`Separated Topologies <userguide_septop_protocol>`
     - * Chemically diverse ligands, including scaffold hops and pairs
         without a suitable common core.
     - * Requires well-defined binding poses for restraint selection.
       * Highly flexible ligands may sample poorly.
       * Net-charge-changing transformations are not supported.
     - Higher per edge
   * - :ref:`Absolute Binding <userguide_abfe_protocol>`
     - * Absolute binding free energies of individual ligands.
     - * Requires a well-defined binding pose for restraint selection.
       * Highly flexible ligands may sample poorly.
       * Charged ligands are not supported.
     - Higher per ligand

Costs are relative, using default settings. To see the exact settings
for your installed version, inspect the protocol's ``default_settings()``.


Domain of applicability
-----------------------

OpenFE's binding free energy protocols are primarily designed
to calculate the noncovalent binding of small molecules to proteins.
Not all molecular systems or alchemical transformations are supported.

System types and transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following tables group molecular systems and transformations
according to the available testing and validation evidence.
These categories indicate the current level of experience with
OpenFE, rather than a guarantee of accuracy for a given system.

Support also depends on the selected force fields and on system
preparation.

.. TODO: Link to a dedicated page on force fields and system parameterization.

Well tested
"""""""""""

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - System or transformation
     - Notes
   * - Soluble proteins
     - * Standard application for noncovalent protein-ligand binding.
       * Tested by OpenFE [1]_ (Relative Hybrid Topology).
   * - Membrane proteins
     - * Tested internally by the OpenFE team.
       * Lipids are limited to those parameterised by the Amber Lipid17
         and Lipid21 force fields.
       * Requires a prepared, solvated and pre-equilibrated system with
         valid periodic box vectors.
   * - Net-charge-changing ligand transformations
     - * Relative Hybrid Topology only. Tested by OpenFE [1]_.
       * Requires explicit charge correction, and longer simulations
         than neutral transformations to converge.
       * Charge changes greater than one elementary charge are not
         supported.


Limited testing
"""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - System or transformation
     - Notes
   * - Ternary complexes (molecular glues)
     - * Single published study using Relative Hybrid Topology
         (`Lukauskis et al. (2025) <https://doi.org/10.26434/chemrxiv-2025-tb29n>`_).
       * Other protocols have not been tested.
   * - Macrocycles
     - * Limited internal testing.
       * Relative Hybrid Topology requires bond-preserving atom mappings,
         so ring-opening, ring-closing and some ring-size-changing
         transformations are not supported.
       * Separated Topologies may be an alternative for such ligands.
   * - Peptide ligands
     - * Limited internal testing.
       * Requires NAGL partial charges and potentially the OpenFF
         Rosemary force field.
   * - Phosphorylated residues
     - * Only phosphoserine (``SEP``) and phosphothreonine (``TPO``) are
         supported.
       * Input structures must use these residue names and include the
         required ``CONECT`` records.

Untested, may be possible
"""""""""""""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - System or transformation
     - Notes
   * - Nucleic acid targets
     - May be possible by adding a suitable Amber nucleic acid force field
       to ``forcefield_settings.forcefields``, e.g.
       ``amber14/DNA.OL15.xml`` for DNA or ``amber14/RNA.OL3.xml`` for
       RNA.
   * - Metalloproteins
     - * Only non-bonded ion models of metal sites are supported.
       * Iron-containing proteins are not supported.
       * See `Güven et al. (2024) <https://doi.org/10.1021/acs.jpcb.4c06379>`_ for a comparison
         of metal models in free energy calculations.

Not supported
"""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - System or transformation
     - Notes
   * - Post translational modifications
     - Modified residues other than ``SEP`` and ``TPO`` (e.g.
       glycosylation, lipidation, methylation, acetylation) cannot
       currently be read from PDB files, regardless of the force field
       used.
   * - Covalently bound ligands
     - Ligands covalently bonded to the receptor are not supported.
   * - Protein mutations
     - * Alchemical transformations of protein residues are not
         implemented by the **openfe** protocols.
       * Experimental support is available in the community-developed
         protocol `feflow <https://github.com/OpenFreeEnergy/feflow>`_.


Known limitations
"""""""""""""""""

These limitations apply to all binding free energy protocols.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Limitation
     - Notes
   * - Fixed protonation and tautomeric states
     - Protonation and tautomeric states are fixed for the whole
       simulation. Changes in these states upon binding are not
       accounted for.
   * - Slow conformational transitions
     - * Large protein conformational changes and transitions between
         ligand binding modes may not be sampled within standard
         simulation timescales.
       * No enhanced sampling methods other than Hamiltonian replica
         exchange are currently available.
   * - Water sampling
     - * Exchange of buried or occluded water molecules may not be
         sampled within standard simulation timescales.
       * No enhanced water sampling methods are currently available.


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

References
----------

.. [1] Large-Scale Collaborative Assessment of Binding Free Energy Calculations for Drug Discovery Using OpenFE, H. Baumann, J Horton et al., J. Chem. Inf. Model., 2026, 66, 11, 6429–6452. doi: 10.1021/acs.jcim.6c00089
.. [2] Separated topologies--a method for relative binding free energy calculations using orientational restraints, G. Rocklin, D. Mobley, K. Dill; J. Chem. Phys., 2013; 138(8):085104. doi: 10.1063/1.4792251
.. [3] Broadening the Scope of Binding Free Energy Calculations Using a Separated Topologies Approach, H. Baumann, E. Dybeck, C. McClendon, F. Pickard IV, V. Gapsys, L. Pérez-Benito, D. Hahn, G. Tresadern, A. Mathiowetz, D. Mobley, J. Chem. Theory Comput., 2023, 19, 15, 5058–5076. doi: 10.1021/acs.jctc.3c00282
