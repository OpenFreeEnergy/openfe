Plain MD Protocol
=================

Overview
--------

The :class:`.PlainMDProtocol` enables the user to run a Molecular Dynamics (MD) simulation of a :class:`.ChemicalSystem`, which can contain e.g. a solvated protein-ligand complex, a molecule and water, or a molecule in vacuum.

.. todo: Later add ref to ChemicalSystem section

Scientific Details
------------------

The :class:`.PlainMDProtocol` runs MD simulations of a system either in solvent or vacuum, depending on the input provided by the user in the :class:`.ChemicalSystem`.
The protocol applies a 
`LangevinMiddleIntegrator <http://docs.openmm.org/development/api-python/generated/openmm.openmm.LangevinMiddleIntegrator.html>`_ 
which uses Langevin dynamics, with the LFMiddle discretization [1]_.  

Simulation Steps and Outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If there is a ``SolventComponent`` in the :class:`.ChemicalSystem`, the each :class:`.ProtocolUnit` carries out the following steps:

.. list-table:: 
  :widths: 50 50
  :header-rows: 1

  * - Step
    - Outputs (with default names)
  * - 1. Parameterize the system using `OpenMMForceFields <https://github.com/openmm/openmmforcefields>`_ and `Open Force Field <https://github.com/openforcefield/openff-forcefields>`_
    - Forcefield cache (``db.json``)
  * - 2. OpenMM object creation
    - Structure of the full system (``system.pdb``)
  * - 3. Minimize the system
    - Minimized Structure (``minimized.pdb``)
  * - 4. Equilibrate in the canonical (NVT) ensemble
    - NVT equilibrated structure (``equil_nvt.pdb``)
  * - 5. Equilibrate the system under isobaric-isothermal (NPT) conditions
    - NPT equilibrated structure (``equil_npt.pdb``)
  * - 6. Production simulate the system under isobaric-isothermal (NPT) conditions
    - Simulation trajectory (``simulation.xtc``), Checkpoint file (``checkpoint.chk``), Log output (``simulation.log``)

A MonteCarloBarostat is used in the NPT ensemble to maintain constant pressure.
Relevant settings under solvent conditions include the solvation settings that control the ``solvent_model`` and ``solvent_padding``.

If the :class:`.ChemicalSystem` does not contain a ``SolventComponent``, the protocol runs an MD simulation in vacuum. After a minimization, the protocol performs an equilibration, followed by a production run with no periodic boundary conditions and infinite cutoffs. Settings that control the barostat or the solvation are ignored for vaccum MD simulations.

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
