
Troubleshooting Simulations 
===========================

There are many different failure modes for simulations.
In this guide, we will cover some tips and strategies for troubleshooting simulation failures.

Log Debug information
---------------------

.. note::

   When using a scheduler (e.g. SLURM) be sure to specify output files for standard out and standard error.
   For example when using SLURM both ``--output=`` and ``--error=`` must be set to view errors.

One of the first troubleshooting steps is to increase the verbosity of the logging.
``openfe`` uses Python's native logging library which can be `configured <https://docs.python.org/3/howto/logging.html#configuring-logging>`_ either using a Python API or a configuration file.
When using ``openfe quickrun`` using the configuration file is more convenient.
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

   $ openfe --log debug_logging.conf quickrun -d results/ -o results/result_lig_ejm_31_solvent_lig_ejm_42_solvent.json transformations/easy_rbfe_lig_ejm_31_solvent_lig_ejm_42_solvent.json

.. note::

   The ``--log debug_logging.conf`` argument goes between ``openfe`` and ``quickrun``.

This will cause every package to log at the debug level, which may be quite verbose and noisy but should aid in identify what is going on right before the exception is thrown.
We do not recommend setting the log level to debug for production runs as the logging may slow down the simulation and add a lot of noise to the output.
