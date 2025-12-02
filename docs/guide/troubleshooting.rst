
Troubleshooting Simulations 
===========================

This guide covers tips and strategies for troubleshooting simulation failures.

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
