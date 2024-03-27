Absolute Solvation Protocol
===========================

Overview
--------

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

Scientific Details
-----------------

Partial annhilation scheme
~~~~~~~~~~~~~~~~~~~~~~~~~~

In the :class:`.AbsoluteSolvationProtocol` the coulombic interactions of the molecule are fully turned off (annihilated),
while the Lennard-Jones interactions are decoupled, meaning the intermolecular interactions turned off, keeping the intramolecular Lennard-Jones interactions.

The lambda schedule
~~~~~~~~~~~~~~~

Molecular interactions are turned off during an alchemical path with a discrete set of lambda windows. The electrostatic interactions are turned off first, followed by the decoupling of the van-der-Waals interactions. A soft-core potential is applied to the Lennard-Jones potential to avoid instablilites in intermediate lambda windows. 

Simulation details
~~~~~~~~~~~~~~~~~~
The protocol applies a LangevinMiddleIntegrator which uses Langevin dynamics, with the LFMiddle discretization (J. Phys. Chem. A 2019, 123, 28, 6056-6079).
Before running the production MD simulation in the NPT ensemble, the protocol performs a minimization of the system, followed by an equilibration in the NPT ensemble. A MonteCarloBarostat is used in the NPT ensemble to maintain constant pressure.

Getting the free energy estimate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The free energy differences are obtained from simulation data using the MBAR estimator (multistate Bennett acceptance ratio estimator).
TODO: ADD what you can get out of the results object's methods

Simulation overview
-------------------

This section should essentially give an overview of the DAG and what each unit is doing.

For example we would want to say that each unit is doing a non-alchemical equilibration followed by an alchemical production.

We would also mention how the DAG constructs and runs both the vacuum and solvent legs concurrently.

See Also
-------

Setting up RFE calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :ref:`Setting up RBFE calculations <define_rbfe>`
* :ref:`Setting up RHFE calculations <define_rhfe>`

Tutorials
~~~~~~~~~

<Insert relevant tutorial page, note: see issue 781>

Cookbooks
~~~~~~~~~

Maybe a list of relevant cookbooks, otherwise just a link to the cookbook page.

API Documentation
~~~~~~~~~~~~~~~~~

* :ref:`OpenMM Absolute Solvation Free Energy <afe solvation protocol api>`
* :ref:`OpenMM Protocol Settings <openmm protocol settings api>`

References
----------
Some relevant references that folks can look at, maybe links to pymbar/yank/perses/openmmtools/openmm/etc...
