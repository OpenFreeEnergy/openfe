Plain MD Protocol
=================

Overview
--------

The :class:`.PlainMDProtocol` enables the user to run an MD simulation of a ``ChemicalSystem``, which can contain e.g. a solvated protein-ligand complex, or a ligand and water.

TODO: Later add ref to ChemicalSystem section

Scientific Details
------------------

The :class:`.PlainMDProtocol` runs MD simulations of a system either in solvent or vacuum, depending on the input provided by the user in the `ChemicalSystem`.
The protocol applies a 
`LangevinMiddleIntegrator <http://docs.openmm.org/development/api-python/generated/openmm.openmm.LangevinMiddleIntegrator.html>`_ 
which uses Langevin dynamics, with the LFMiddle discretization [1]_.  

Simulation Steps
~~~~~~~~~~~~~~~~

If there is a ``SolventComponent`` in the ``ChemicalSystem``, the each :class:`ProtocolUnit` carries out the following steps:

1. Parameterize the system using `OpenMMForceFields <https://github.com/openmm/openmmforcefields>`_ and `Open Force Field <https://github.com/openforcefield/openff-forcefields>`_.
2. Minimize the system
3. Equilibrate in the canonical ensemble
4. Equilibrate and production simulate the system (under NPT conditions using a MonteCarloBarostat to maintain constant pressure)

Relevant settings under solvent conditions include the solvation settings that control the ``solvent_model`` and ``solvent_padding``.

If the ``ChemicalSystem`` does not contain a ``SolventComponent``, the protocol runs an MD simulation in vacuum. After a minimization, the protocol performs an NVT equilibration, followed by an NVT production run with no periodic boundary conditions and infinite cutoffs. Settings that control the barostat or the solvation are ignored for vaccum MD simulations.

Performance consideration for gas phase MD simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For gas phase MD simulations, we suggest setting ``OPENMM_CPU_THREADS=1`` to obtain good performance.

See Also
--------

**Tutorials**

* :any:`MD tutorial <../../tutorials/md_tutorial>`

**API Documentation**

* :ref:`OpenMM plain MD protocol <md protocol api>`
* :ref:`OpenMM Protocol Settings <openmm protocol settings api>`

References
----------
* `OpenMMTools <https://openmmtools.readthedocs.io/en/stable/>`_
* `OpenMM <https://openmm.org/>`_

.. [1] Unified Efficient Thermostat Scheme for the Canonical Ensemble with Holonomic or Isokinetic Constraints via Molecular Dynamics, Zhijun Zhang, Xinzijian Liu, Kangyu Yan, Mark E. Tuckerman, and Jian Liu, J. Phys. Chem. A 2019, 123, 28, 6056-6079
