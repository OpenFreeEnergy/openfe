Relative Hybrid Topology Protocol
=================================

Overview
--------

The relative free energy calculation approach calculates the difference in 
free energy between two similar ligands. Depending on the ChemicalSystem 
provided, the protocol either calculates the relative binding free energy 
(RBFE), or the relative hydration free energy (RHFE). In a thermodynamic 
cycle, one ligand is converted into the other ligand by alchemically 
transforming the atoms that vary between the two ligands. The 
transformation is carried out in both environments, meaning both in the 
solvent (ΔG\ :sub:`solv`\) and in the binding site (ΔG\ :sub:`site`\) for RBFE calculations 
and in the solvent (ΔG\ :sub:`solv`\) and vacuum (ΔG\ :sub:`vacuum`\) for RHFE calculations.

.. _label: Thermodynamic cycle for the relative binding free energy protocol
.. figure:: img/rbfe_thermocycle.png
   :scale: 50%

   Thermodynamic cycle for the relative binding free energy protocol.
   
Scientific Details
------------------

The Hybrid Topology approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.RelativeHybridTopologyProtocol` uses a hybrid topology approach to represent the two
ligands, meaning that a single set of coordinates is used to represent the
common core of the two ligands while the atoms that differ between the two
ligands are represented separately. An atom map defines which atoms belong
to the core (mapped atoms) and which atoms are unmapped and represented
separately. During the alchemical transformation, mapped atoms are switched
from the type in one ligand to the type in the other ligands, while unmapped
atoms are switched on or off, depending on which ligand they belong to.

The lambda schedule
~~~~~~~~~~~~~~~~~~~

The protocol interpolated molecular interactions between the initial and final state of the perturbation using a discrete set of lambda windows. A function describes how the different lambda components (bonded and nonbonded terms) are interpolated.
Only parameters that differ between state A (``lambda=0``) and state B (``lambda=1``) are interpolated. 
In the default lambda function in the :class:`.RelativeHybridTopologyProtocol`, first the electrostatic interactions of state A are turned off while simulataneously turning on the van-der-Waals interactions of state B. Then, the van-der-Waals interactions of state A are turned off while simulatenously turning on the electrostatic interactions of state B. Bonded interactions are interpolated linearly between lambda=0 and lambda=1. 

Simulation details
~~~~~~~~~~~~~~~~~~

The protocol applies a LangevinMiddleIntegrator which uses Langevin dynamics, with the LFMiddle discretization [1]_.
Before running the production MD simulation in the NPT ensemble, the protocol performs a minimization of the system, followed by an equilibration in the NPT ensemble. A MonteCarloBarostat is used in the NPT ensemble to maintain constant pressure.

Getting the free energy estimate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The free energy differences are obtained from simulation data using the MBAR estimator (multistate Bennett acceptance ratio estimator).
In addition to the MBAR estimates of the two legs of the thermodynamic cycle and the overall realtive binding free energy difference, 
the protocol also returns some metrics to help assess convergence of the results. 
The forward and reverse analysis looks at the time convergence of the free energy estimates. 
The MABR overlap matrix checks how well lambda states overlap. Since the accuracy of the MBAR estimator depends on sufficient overlap between lambda states, this is a very important metric.
To assess the mixing of lambda states in the Hamiltonian replica exchange method, the results object returns the replica exchange transition matrix, which can be plotted as the replica exchange overlap matrix, as well as a time series of all replica states. (Todo: link to the results page in case examples of these plots are deposited there)

Simulation overview
-------------------

The ``ProtocolDAG`` of the :class:`.RelativeHybridTopologyProtocol` contains the ``ProtocolUnits`` from one leg of the thermodynamic
cycle. 
This means that each ``ProtocolDAG`` only runs a single leg of a thermodynamic cycle and therefore two Protocol instances need to be run to get the overall relative free energy difference, DDG. 
If multiple repeats of the protocol are run, the ``ProtocolDAG`` contains multiple units of the transformation.

See Also
--------

Setting up RFE calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :ref:`Setting up RBFE calculations <define-rbfe>`
* :ref:`Setting up RHFE calculations <define-rsfe>`

Tutorials
~~~~~~~~~

* :ref:`RBFE CLI tutorial <rfe cli tutorial>`
* :ref:`RBFE Python tutorial <rfe python tutorial>`

Cookbooks
~~~~~~~~~

:ref:`Cookbooks <cookbooks>`

API Documentation
~~~~~~~~~~~~~~~~~

* :ref:`OpenMM Relative Hybrid Topology Protocol <rfe protocol api>`
* :ref:`OpenMM Protocol Settings <openmm protocol settings api>`

References
----------
* `pymbar <https://pymbar.readthedocs.io/en/stable/>`_
* `perses <https://perses.readthedocs.io/en/latest/>`_
* `OpenMMTools <https://openmmtools.readthedocs.io/en/stable/>`_
* `OpenMM <https://openmm.org/>`_

.. [1] Unified Efficient Thermostat Scheme for the Canonical Ensemble with Holonomic or Isokinetic Constraints via Molecular Dynamics, Zhijun Zhang, Xinzijian Liu, Kangyu Yan, Mark E. Tuckerman, and Jian Liu, J. Phys. Chem. A 2019, 123, 28, 6056-6079
