.. _userguide_cli_quickrun:

Using Quickrun to execute Transformations
=========================================

The ``openfe quickrun`` command executes a single alchemical Transformation.
This is currrently the primary way to execute Transformations after they
have been created during network planning.


Basic Usage
-----------

To run a Transformation (``transformation.json``) and save results to ``results.json``:

.. code:: none

    openfe quickrun transformation.json -d workdir/ -o workdir/results.json

The ``-d`` / ``--work-dir`` flag controls where working files (checkpoints, 
trajectory data, etc...) are written. If it is ommited, the current directory
will be used.

The ``-o`` flag controls where the results file will be written. If it is omitted,
results are written to a file named ``<transformation_key>_results.json`` in the working directory, where `<transformation_key>` is a unique identifier.


Resuming a halted Job
---------------------

When ``openfe quickrun`` starts, it saves a plan of the simulation to a
cache file before execution begins:

.. code:: none

    <work-dir>/quickrun_cache/<transformation_key>-protocolDAG.json

This cache is automatically removed once the job completes successfully.

If a job is interrupted (e.g. due to a wall-time limit, node failure, or
manual cancellation), you can resume the interrupted job by passing the ``--resume`` flag:

.. code:: none

    openfe quickrun transformation.json -d workdir/ -o workdir/results.json --resume

The planned simulation cache will be used to identify where in the simulation
process it is and, if supported by the Transformation Protocol, how to resume.

.. note::

    The same ``-d`` / ``--work-dir`` used in the original run
    must be specified so that ``quickrun`` can locate the cache file.

If you pass ``--resume`` but no cache file is found (e.g. the job never
started), the following warning is printed and a fresh execution begins:

.. code:: none

    No checkpoint found at <work-dir>/quickrun_cache/<transformation_key>-protocolDAG.json!
    Starting new execution.

If the cache file is corrupted (e.g. due to an incomplete write at
the moment of interruption), ``quickrun --resume`` will raise an error with instructions to rerun the simulation:

.. code:: none

    Recovery failed, please remove <work-dir>/quickrun_cache/<transformation_key>-protocolDAG.json
    and any results from your working directory before continuing to create a new protocol, or run without `--resume`.

If you do not pass the ``--resume`` flag, the code will detect the partially
complete transformation and prevent you from accidentally starting a duplicate
run. The following error will be raised:

.. code:: none

    RuntimeError: Transformation has been started but is incomplete. Please
    remove <path>/quickrun_cache/<key>-protocolDAG.json and rerun, or resume
    execution using the ``--resume`` flag.

See Also
--------

- :ref:`cli-reference` - full CLI reference for ``openfe quickrun``
- :ref:`rbfe_cli_tutorial` - a tutorial on how to use the CLI to run hybrid topology relative binding free energy calculations.
