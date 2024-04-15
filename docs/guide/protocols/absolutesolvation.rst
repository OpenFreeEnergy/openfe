Absolute Solvation Protocol
===========================

Overview
--------

The :class:`AbsoluteSolvationProtocol <.AbsoluteSolvationProtocol>` calculates the free energy change 
associate with transferring a molecule from vacuum into a solvent.

.. note::
   Currently, water is the only supported solvent, however, more solvents might be possible in the future.

The absolute solvation free energy is calculated through a thermodynamic cycle. 
In this cycle, the interactions of the molecule are decoupled, meaning turned off, using a partial annhilation scheme (see below) both in the solvent and in the vacuum phases.
The absolute solvation free energy is then obtained via summation of free energy differences along the thermodynamic cycle.

.. figure:: img/ahfe_thermocycle.png
   :scale: 80%

   Thermodynamic cycle for the absolute solvation free energy protocol.

Scientific Details
------------------

Partial annhilation scheme
~~~~~~~~~~~~~~~~~~~~~~~~~~

In the :class:`.AbsoluteSolvationProtocol` the coulombic interactions of the molecule are fully turned off (annihilated). 
The Lennard-Jones interactions are instead decoupled, meaning the intermolecular interactions turned off, keeping the intramolecular Lennard-Jones interactions.

The lambda schedule
~~~~~~~~~~~~~~~~~~~

Molecular interactions are turned off during an alchemical path using a discrete set of lambda windows. The electrostatic interactions are turned off first, followed by the decoupling of the Lennard-Jones interactions. 
A soft-core potential is applied to the Lennard-Jones potential to avoid instablilites in intermediate lambda windows. 
Both the soft-core potential functions from Beutler et al. [1]_ and from Gapsys et al. [2]_ are available and can be specified in the ``alchemical_settings.softcore_LJ`` settings
(default: ``gapsys``).
The lambda schedule is defined in the ``lambda_settings`` objects ``lambda_elec`` and ``lambda_vdw``. Note that the ``lambda_restraints`` setting is ignored for the :class:`.AbsoluteSolvationProtocol`.

Simulation overview
~~~~~~~~~~~~~~~~~~~

The :class:`.ProtocolDAG` of the :class:`.AbsoluteSolvationProtocol` contains :class:`.ProtocolUnit`\ s from both the vacuum and solvent transformations.
This means that both legs of the thermodynamic cycle are constructured and run concurrently in the same :class:`.ProtocolDAG`. This is different from the :class:`.RelativeHybridTopologyProtocol` where the :class:`.ProtocolDAG` only runs a single leg of a thermodynamic cycle.
If multiple ``protocol_repeats`` are run (default: ``protocol_repeats=3``), the :class:`.ProtocolDAG` contains multiple :class:`.ProtocolUnit`\ s of both vacuum and solvent transformations.

Simulation steps
""""""""""""""""

Each :class:`.ProtocolUnit` (whether vacuum or solvent) carries out the following steps:

1. Parameterize the system using `OpenMMForceFields <https://github.com/openmm/openmmforcefields>`_ and `Open Force Field <https://github.com/openforcefield/openff-forcefields>`_.
2. Equilibrate the fully interacting system using a short MD simulation using the same approach as the :class:`.PlainMDProtocol` (in the solvent leg this will include rounds of NVT and NPT equilibration).
3. Create an alchemical system.
4. Minimize the alchemical sysem.
5. Equilibrate and production simulate the alchemical system using the chosen multistate sampling method (under NPT conditions if solvent is present).
6. Analyze results for the transformation.

Note: three different types of multistate sampling (i.e. replica swapping between lambda states) methods can be chosen; HREX, SAMS, and independent (no lambda swaps attempted). By default the HREX approach is selected, this can be altered using ``solvent_simulation_settings.sampler_method`` or ``vacuum_simulation_settings.sampler_method`` (default: ``repex``).

Simulation details
""""""""""""""""""

Here are some details of how the simulation is carried out which are not detailed in the :class:`.AbsoluteSolvationSettings`:

* The protocol applies a `LangevinMiddleIntegrator <https://openmmtools.readthedocs.io/en/latest/api/generated/openmmtools.mcmc.LangevinDynamicsMove.html>`_ which uses Langevin dynamics, with the LFMiddle discretization [3]_.
* A MonteCarloBarostat is used in the NPT ensemble to maintain constant pressure.

Getting the free energy estimate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The free energy differences are obtained from simulation data using the `MBAR estimator <https://www.alchemistry.org/wiki/Multistate_Bennett_Acceptance_Ratio>`_ (multistate Bennett acceptance ratio estimator) as implemented in the `PyMBAR package <https://pymbar.readthedocs.io/en/master/mbar.html>`_.
Both the MBAR estimates of the two legs of the thermodynamic cycle, and the overall absolute solvation free energy (of the entire cycle) are obtained,
which is different compared to the results in the :class:`.RelativeHybridTopologyProtocol` where results from two legs of the thermodynamic cycle are obtained separately.

In addition to the estimates of the free energy changes and their uncertainty, the protocol also returns some metrics to help assess convergence of the results, these are detailed in the :ref:`Results <userguide_results>` section.

See Also
--------

**Setting up AFE calculations**

* :ref:`Defining the Protocol <defining-protocols>`

..
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
