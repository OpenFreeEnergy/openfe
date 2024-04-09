Absolute Solvation Protocol
===========================

Overview
--------

The :class:`absolute solvation protocol <.AbsoluteSolvationProtocol>` calculates the free energy change 
associate with transferring a molecule from vacuum into a solvent.

.. note::
   Currently, water is the only supported solvent, however, more solvents might be possible in the future.

The absolute solvation free energy is calculated through a thermodynamic cycle. 
In this cycle, the interactions of the molecule are decoupled, meaning turned off, using a partial annhilation scheme (see below) both in the solvent and in the vacuum phases.
The absolute solvation free energy is then obtained via summation of free energy differences along the thermodynamic cycle.

.. figure:: img/ahfe_thermocycle.png
   :scale: 100%

   Thermodynamic cycle for the absolute solvation free energy protocol.

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

The protocol applies a 
`LangevinMiddleIntegrator <https://openmmtools.readthedocs.io/en/latest/api/generated/openmmtools.mcmc.LangevinDynamicsMove.html>`_ which 
uses Langevin dynamics, with the LFMiddle discretization [1]_.
Before running the production MD simulation in the NPT ensemble, the protocol performs a minimization of the system, followed by an equilibration in the NPT ensemble. A MonteCarloBarostat is used in the NPT ensemble to maintain constant pressure.

Getting the free energy estimate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The free energy differences are obtained from simulation data using the MBAR estimator (multistate Bennett acceptance ratio estimator).
Both the MABR estimates of the two legs of the thermodynamic cycle, and the overall absolute solvation free energy (of the entire cycle) are obtained,
which is different compared to the results in the :class:`.RelativeHybridTopologyProtocol` where results from two legs of the thermodynamic cycle are obtained separately.
TODO: Link to results page once done

In addition to the estimates of the free energy changes and their uncertainty, the protocol also returns some metrics to help assess convergence of the results. The forward and reverse analysis looks at the time convergence of the free energy estimates. The MABR overlap matrix checks how well lambda states overlap. Since the accuracy of the MBAR estimator depends on sufficient overlap between lambda states, this is a very important metric. 
To assess the mixing of lambda states in the Hamiltonian replica exchange method, the results object returns the replica exchange transition matrix, which can be plotted as the replica exchange overlap matrix, as well as a time series of all replica states. (Todo: link to the results page in case examples of these plots are deposited there) 

See Also
--------

**Setting up AFE calculations**

* :ref:`Defining the Protocol <defining-protocols>`

To be added: Setting up AHFE calculations

**Tutorials**

* :any:`Absolute Hydration Free Energies tutorial <../../tutorials/ahfe_tutorial>`

**Cookbooks**

:ref:`Cookbooks <cookbooks>`

**API Documentation**

* :ref:`OpenMM Absolute Solvation Free Energy <afe solvation protocol api>`
* :ref:`OpenMM Protocol Settings <openmm protocol settings api>`

References
----------

* `pymbar <https://pymbar.readthedocs.io/en/stable/>`_
* `yank <http://getyank.org/latest/>`_
* `OpenMMTools <https://openmmtools.readthedocs.io/en/stable/>`_
* `OpenMM <https://openmm.org/>`_

.. [1] Avoiding singularities and numerical instabilities in free energy calculations based on molecular simulations, T.C. Beutler, A.E. Mark, R.C. van Schaik, P.R. Greber, and W.F. van Gunsteren, Chem. Phys. Lett., 222 529–539 (1994)
.. [2] New Soft-Core Potential Function for Molecular Dynamics Based Alchemical Free Energy Calculations, V. Gapsys, D. Seeliger, and B.L. de Groot, J. Chem. Theor. Comput., 8 2373-2382 (2012)
.. [3] Unified Efficient Thermostat Scheme for the Canonical Ensemble with Holonomic or Isokinetic Constraints via Molecular Dynamics, Zhijun Zhang, Xinzijian Liu, Kangyu Yan, Mark E. Tuckerman, and Jian Liu, J. Phys. Chem. A 2019, 123, 28, 6056-6079
