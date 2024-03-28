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
------------------

Partial annhilation scheme
~~~~~~~~~~~~~~~~~~~~~~~~~~

In the :class:`.AbsoluteSolvationProtocol` the coulombic interactions of the molecule are fully turned off (annihilated),
while the Lennard-Jones interactions are decoupled, meaning the intermolecular interactions turned off, keeping the intramolecular Lennard-Jones interactions.

The lambda schedule
~~~~~~~~~~~~~~~~~~~

Molecular interactions are turned off during an alchemical path using a discrete set of lambda windows. The electrostatic interactions are turned off first, followed by the decoupling of the van-der-Waals interactions. A soft-core potential is applied to the Lennard-Jones potential to avoid instablilites in intermediate lambda windows. 

Simulation details
~~~~~~~~~~~~~~~~~~

The protocol applies a LangevinMiddleIntegrator which uses Langevin dynamics, with the LFMiddle discretization [1]_.
Before running the production MD simulation in the NPT ensemble, the protocol performs a minimization of the system, followed by an equilibration in the NPT ensemble. A MonteCarloBarostat is used in the NPT ensemble to maintain constant pressure.

Getting the free energy estimate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The free energy differences are obtained from simulation data using the MBAR estimator (multistate Bennett acceptance ratio estimator).
TODO: Link to results page once done
In addition to the MBAR estimates of the two legs of the thermodynamic cycle and the overall absolute solvation free energy difference, the protocol also returns some metrics to help assess convergence of the results. The forward and reverse analysis looks at the time convergence of the free energy estimates. The MABR overlap matrix checks how well lambda states overlap. Since the accuracy of the MBAR estimator depends on sufficient overlap between lambda states, this is a very important metric. 
To assess the mixing of lambda states in the Hamiltonian replica exchange method, the results object returns the both replica exchange transition matrix, which can be plotted as the replica exchange overlap matrix, as well as a time series of all replica states.  

Simulation overview
-------------------

The ``ProtocolDAG`` of the :class:`.AbsoluteSolvationProtocol` contains both the units from the vacuum and from the solvent transformations. Therefore, both legs of the thermodynamic cycle are constructured and run concurrently in the same ``ProtocolDAG``.
If multiple repeats of the protocol are run, the ``ProtocolDAG`` contains multiple units of both vacuum and solvent transformations. 
For each ``ProtocolUnit`` in the ``ProtocolDAG`` first, a non-alchemical equilibration is carried out (minimization, NVT and NPT equilibration), followed by an alchemical production.

See Also
--------

Setting up RFE calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :ref:`Setting up RBFE calculations <define-rbfe>`
* :ref:`Setting up RHFE calculations <define-rsfe>`

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
`pymbar <https://pymbar.readthedocs.io/en/stable/>`_
`yank <http://getyank.org/latest/>`_
`perses <https://perses.readthedocs.io/en/latest/>`_
`OpenMMTools <https://openmmtools.readthedocs.io/en/stable/>`_
`OpenMM <https://openmm.org/>`_

.. [1] Unified Efficient Thermostat Scheme for the Canonical Ensemble with Holonomic or Isokinetic Constraints via Molecular Dynamics, Zhijun Zhang, Xinzijian Liu, Kangyu Yan, Mark E. Tuckerman, and Jian Liu, J. Phys. Chem. A 2019, 123, 28, 6056-6079
