.. _userguide_quickrun:

Execution with ``quickrun``
=======================

The planning and preparation of a campaign of alchemical simulations using ``openfe`` is intended to be achievable on a local workstation in a matter of minutes.
The *execution* of these simulations however requires a large amount of computational power, and beyond running single calculations locally, is intended to be distributed across a HPC environment.
Doing this requires storing and sending the details of the simulation from the local workstation to a HPC environment, which can be done via the :func:`.Transformation.to_json` function which :ref:`creates a saved JSON version of the data<dumping_transformations>`.
These serialized JSON files are the currency of executing a campaign of simulations and contain all the information required to execute a single simulation.

To read the ``Transformation`` information and execute the simulation, the command line interface provides the ``openfe quickrun`` command, the full details of which are given in :ref:`the CLI reference section<cli_quickrun>`.

What quickrun does
------------------

``openfe quickrun`` executes one :class:`.Transformation`.
What a Transformation contains depends on the Protocol:
for the Relative Hybrid Topology Protocol it is one leg of an edge, for example the complex leg,
while for the Separated Topologies and absolute Protocols it is the whole thermodynamic cycle.

When quickrun starts, it creates the ``ProtocolDAG`` of the Transformation:
the setup, simulation and analysis tasks (``ProtocolUnit``\s) of every repeat (see :ref:`userguide_execution_theory`).
It saves this plan to a cache file in the work directory, then executes the tasks one at a time.
Each task writes its files to its own subdirectory of the work directory.
If a task fails, quickrun retries it up to two more times; if it still fails, quickrun stops.

When all tasks have finished, or one has failed, quickrun combines the task results into a :class:`.ProtocolResult`,
writes the free energy estimate and its uncertainty to the results file, and removes the cache file.
If quickrun is interrupted before then, the cache file remains and the run can be resumed:
tasks that had finished are skipped, and an interrupted simulation continues from its last checkpoint.

Organizing a campaign
---------------------

You decide where and when each Transformation runs, for example by submitting one job per Transformation to a cluster.
You can run multiple simulations in parallel by running several quickrun calls at the same time,
for example one per Transformation, one per repeat, or several on the same GPU.
The results files are then collected with ``openfe gather``.

If you would rather have **openfe** schedule the tasks of a whole network for you,
see :ref:`userguide_task_based_execution`.

.. admonition:: How-to guides

   * :doc:`/cookbook/run_quickrun`
   * :doc:`/cookbook/resume_quickrun`
   * :doc:`/cookbook/run_on_slurm`
   * :doc:`/cookbook/run_repeats_in_parallel`
   * :doc:`/cookbook/run_with_mps`

See Also
--------

- :ref:`userguide_results` - details on inspecting these results.
- :ref:`cli-reference` - full CLI reference for ``openfe quickrun``
- :ref:`rbfe_cli_tutorial` - a tutorial on how to use the CLI to run hybrid topology relative binding free energy calculations.
