Absolute Solvation Protocol
===========================

The absolute solvation protocol calculates the free energy change 
associate with transferring a molecule from vacuum into a solvent.

.. note::
   Currently, water is the only supported solvent, however, more solvents might be possible in the future.

The absolute hydration free energy is calculated through a thermodynamic cycle. 
In this cycle, the interactions of the molecule are decoupled, meaning turned off, both in the solvent and in the vacuum phases.
The absolute hydration free energy is then obtained via summation of free energy differences along the thermodynamic cycle.

.. figure:: img/ahfe_thermocycle.png
   :scale: 100%

   Thermodynamic cycle for the absolute hydration free energy protocol.

the coulombic interactions of the molecule are fully turned off (annihilated), while the Lennard-Jones interactions are decoupled, meaning the intermolecular interactions turned off, keeping the intramolecular Lennard-Jones interactions.
In the :class:`.AbsoluteSolvationProtocol` the coulombic interactions of the molecule are fully turned off (annihilated),
while the Lennard-Jones interactions are decoupled, meaning the intermolecular interactions turned off, keeping the intramolecular Lennard-Jones interactions.
Molecular interactions are turned off during an alchemical path with a discrete set of lambda windows. The electrostatic interactions are turned off first, followed by the decoupling of the van-der-Waals interactions. A soft-core potential is applied to the Lennard-Jones potential to avoid instablilites in intermediate lambda windows. All MD simulations are run in the NPT ensemble.
The free energy differences are obtained from simulation data using the MBAR estimator (multistate Bennett acceptance ratio estimator).

See Also
~~~~~~~~

Setting up RFE calculations
+++++++++++++++++++++++++++

:ref:`define_rbfe`
:ref:`define_rhfe`

Tutorials
+++++++++

<Insert relevant tutorial page, note: see issue 781>

Cookbooks
+++++++++

Maybe a list of relevant cookbooks, otherwise just a link to the cookbook page.

API Documentation
+++++++++++++++++

:ref:`afe solvation protocol api`
<Insert Protocol API docs link>
<Insert Protocol Settings API docs link> 
