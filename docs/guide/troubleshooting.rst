Troubleshooting Simulations
===========================

This guide covers tips and strategies for troubleshooting simulation failures.

Troubleshooting
---------------

- :ref:`Periodic box size / nonbonded cutoff errors <troubleshooting-periodic-box-size-nonbonded-cutoff-errors>`
- :ref:`SMIRNOFF force field parameter assignment errors <troubleshooting-smirnoff-force-field-parameter-assignment-errors>`
- :ref:`NaN errors during simulation <troubleshooting-nan-errors-during-simulation>`
- :ref:`Log debug information <troubleshooting-log-debug-information>`
- :ref:`JAX warnings <troubleshooting-jax-warnings>`
- :ref:`PYMBAR_DISABLE_JAX <troubleshooting-pymbar-disable-jax>`

.. _troubleshooting-periodic-box-size-nonbonded-cutoff-errors:

Periodic box size / nonbonded cutoff errors
-------------------------------------------

You may see errors like:

- ``openmm.OpenMMException: The periodic box size has decreased to less than twice the nonbonded cutoff.``
- ``openmm.OpenMMException: NonbondedForce: The cutoff distance cannot be greater than half the periodic box size.``

These errors have the same cause: the system is too small for the chosen nonbonded cutoff.

Possible causes
^^^^^^^^^^^^^^^

- Insufficient solvent padding.
- Incorrect periodic box vectors.
- A solvated system that does not leave enough space around the solute.

Suggested fixes
^^^^^^^^^^^^^^^

- If the system is solvated by the OpenFE protocol, increase the protocol's ``solvent_padding`` setting.
- If the system is explicitly solvated, e.g. when using ``SolvatedPDBComponent`` or ``ProteinMembraneComponent``:

  - Verify that the correct box vectors were supplied.
  - Ensure there is sufficient solvent around the system.

We do **not** recommend changing the nonbonded cutoff to fix this issue, since force fields are typically parameterized for these values and changing them may affect accuracy.
If you have modified the nonbonded cutoff, it may be that it is too large for the simulation box.

.. _troubleshooting-smirnoff-force-field-parameter-assignment-errors:

SMIRNOFF force field parameter assignment errors
------------------------------------------------

You may see errors like:

- ``openff.interchange.exceptions.UnassignedBondError: BondHandler was not able to find parameters for the following valence terms: - Topology indices (0, 1): names and elements (C1 C), (Si1 Si),``

This usually means the SMIRNOFF-style force field does not have parameters for part of the chemistry in the system.
In the example above, the force field was unable to assign bond parameters for bonds involving silicon.

Possible causes
^^^^^^^^^^^^^^^

- The system contains atoms or functional groups that are not covered by the selected force field.

Suggested fixes
^^^^^^^^^^^^^^^

- Check whether the chemistry in the system is expected to be supported by the chosen force field.
- Consider using a molecule specific force field with custom parameters to guarantee coverage of the chemistry in your system.

If the error lists specific bonds or valence terms, those terms are the ones that could not be assigned parameters.

.. _troubleshooting-nan-errors-during-simulation:

NaN errors during simulation
----------------------------

You may see errors like:

- ``openmm.OpenMMException: Particle coordinate is NaN.``
- ``openmmtools.multistate.utils.SimulationNaNError: Propagating replica 0 at state 10 resulted in a NaN!``

For more information on the OpenMM error, see the OpenMM FAQ entry on NaNs:
https://github.com/openmm/openmm/wiki/Frequently-Asked-Questions#nan

These errors usually mean that the simulation became numerically unstable during minimization or propagation.
In some cases the simulation can be rescued by restarting from the last stable state, by default the protocols will attempt this up to ``20`` times before giving up and so you may see multiple ``NaN`` errors in the logs.

Possible causes
^^^^^^^^^^^^^^^

- Missing capping groups or other issues in the receptor structure.
- A poor atom mapping, including mappings that break bonds, map too few heavy atoms or map atoms whose hybridization changes due to a single to double/triple bond transformation.
- An initial clash between the ligand and a crystal water, the receptor, or another part of the system that the minimizer could not relax.
- A poor input structure for the ligand that the minimizer could not fix.

Suggested fixes
^^^^^^^^^^^^^^^

- Run ``scripts/validate_transformation.py`` on the transformation JSON to try to identify the source of the problem.
- Inspect the receptor for missing residues or missing capping groups.
- Review the atom mapping for bond-breaking transformations or mappings with too few heavy atoms, try to improve the mapping by aligning the ligands before generating the mapping.

  - Consider running the transformation using the `SepTop <https://docs.openfree.energy/en/latest/guide/protocols/septop.html>`_ protocol which does not require an atom mapping and is robust to poor ligand alignment.

- Check the input ligand and starting pose for clashes or poor geometry, try to relax the ligand in the receptor before running the protocol.
- If possible, rebuild or re-prep the input structures before rerunning the protocol.

The state of the system and integrator before the error are often saved in a ``nan-error-logs`` directory, which can help with debugging.

.. _troubleshooting-log-debug-information:

Log Debug information
---------------------

.. note::

   When using a scheduler (e.g. SLURM), be sure to specify output files for standard out and standard error.
   For example, when using SLURM both ``--output=`` and ``--error=`` must be set to view errors.

One of the first troubleshooting steps is to increase the verbosity of the logging.
``openfe`` uses Python's native logging library which can be `configured <https://docs.python.org/3/howto/logging.html#configuring-logging>`_ either using a Python API or a configuration file.

.. warning::

   **We do not recommend setting the log level to debug for production runs,** as the logging may slow down the simulation and add a lot of noise to the output.

When using ``openfe quickrun``, the configuration file is more convenient.
Below is an example logging configuration file that can be used to set the log level to ``DEBUG``:

.. code-block:: ini

   [loggers]
   keys=root

   [handlers]
   keys=stdout

   [formatters]
   keys=standardFormatter,msgOnly

   [handler_stdout]
   class=StreamHandler
   level=DEBUG
   formatter=standardFormatter
   args=(sys.stdout,)

   [logger_root]
   level=DEBUG
   handlers=stdout

   [formatter_standardFormatter]
   format=%(asctime)s %(levelname)s %(name)s: %(message)s

   [formatter_msgOnly]
   format=%(message)s

Save this configuration file as ``debug_logging.conf`` and then run ``openfe quickrun`` with the ``--log`` flag, for example:

.. code-block:: bash

   $ openfe --log debug_logging.conf quickrun -d results/ -o results/result_lig_ejm_31_solvent_lig_ejm_42_solvent.json transformations/rbfe_lig_ejm_31_solvent_lig_ejm_42_solvent.json

Note that the ``--log debug_logging.conf`` argument goes between ``openfe`` and ``quickrun`` on the command line.

This will cause every package to log at the debug level, which may be quite verbose and noisy but should aid in identify what is going on right before the exception is thrown.

.. _troubleshooting-jax-warnings:

JAX warnings
------------

We use ``pymbar`` to analyze the free energy of the system.
``pymbar`` uses JAX to accelerate computation.
The JAX library can utilize a GPU to further accelerate computation.
If the necessary libraries for GPU acceleration are not installed and JAX detects a GPU, JAX will print a warning like this:

.. code-block:: bash

   WARNING:2025-06-10 09:01:40,857:jax._src.xla_bridge:966: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.

This warning does not mean that the *molecular dynamics* simulation will fall back to using the CPU.
The simulation will still use the computing platform specified in the settings.

.. _troubleshooting-pymbar-disable-jax:

PYMBAR_DISABLE_JAX
------------------

Due to a suspected memory leak in the JAX acceleration code in ``pymbar`` we disable JAX acceleration by default.
This memory leak may result in the simulation crashing, wasting compute time.
The error message may look like this:

.. code-block:: bash

   LLVM compilation error: Cannot allocate memory
   LLVM ERROR: Unable to allocate section memory!

We have decided to disable JAX acceleration by default to prevent wasted compute.
However, if you wish to use the JAX acceleration, you may set ``PYMBAR_DISABLE_JAX`` to ``TRUE`` (e.g. put ``export PYMBAR_DISABLE_JAX=FALSE`` in your submission script before running ``openfe quickrun``).
For more information, see these issues on github:

- https://github.com/choderalab/pymbar/issues/564
- https://github.com/OpenFreeEnergy/openfe/issues/1534
- https://github.com/OpenFreeEnergy/openfe/issues/1654
