.. _userguide_quickrun:

Execution with ``quickrun``
=======================

The planning and preparation of a campaign of alchemical simulations using ``openfe`` is intended to be achievable on a local workstation in a matter of minutes.
The *execution* of these simulations however requires a large amount of computational power, and beyond running single calculations locally, is intended to be distributed across a HPC environment.
Doing this requires storing and sending the details of the simulation from the local workstation to a HPC environment, which can be done via the :func:`.Transformation.to_json` function which :ref:`creates a saved JSON version of the data<dumping_transformations>`.
These serialized JSON files are the currency of executing a campaign of simulations and contain all the information required to execute a single simulation.

To read the ``Transformation`` information and execute the simulation, the command line interface provides the ``openfe quickrun`` command, the full details of which are given in :ref:`the CLI reference section<cli_quickrun>`.

Overview
--------

``openfe quickrun`` executes one :class:`.Transformation`.
A Transformation contains everything needed to run a free energy calculation:
the two :class:`.ChemicalSystem`\s, the :class:`.Protocol` and, if the Protocol needs one,
an :class:`.AtomMapping` (see :ref:`alchemical_network_model`).

When quickrun starts, it creates the ``ProtocolDAG`` of the Transformation:
the setup, simulation and analysis tasks (``ProtocolUnit``\s) of every repeat (see :ref:`userguide_execution_theory`).
quickrun then executes these tasks one at a time, and each task writes its files to its own subdirectory of the work directory.
If a task fails, quickrun retries it up to two more times; if it still fails, quickrun stops.

When all tasks have finished, or one has failed, quickrun combines the task results into a :class:`.ProtocolResult`
and writes the free energy estimate and its uncertainty to the results file.

quickrun keeps track of which tasks have finished, so an interrupted run can be resumed with ``--resume``.
Finished tasks are skipped, and an interrupted simulation continues from its last checkpoint
(see :doc:`/how_to_guides/resume_quickrun`).

Organizing a campaign
---------------------

You decide where and when each Transformation runs, for example by
:doc:`submitting one job per Transformation to a cluster </how_to_guides/run_on_slurm>`.
You can run multiple simulations in parallel by running several quickrun calls at the same time,
for example one per Transformation, :doc:`one per repeat </how_to_guides/run_repeats_in_parallel>`,
or :doc:`several on the same GPU </how_to_guides/run_with_mps>`.
The results files are then collected with :ref:`openfe gather <cli_gather>`.

.. TODO: link the "Gather free energy results" how-to instead of the CLI reference once it exists.

If you would rather have **openfe** schedule the tasks of a whole network for you,
see :ref:`userguide_task_based_execution`.

.. admonition:: How-to guides

   * :doc:`/how_to_guides/run_quickrun`
   * :doc:`/how_to_guides/resume_quickrun`
   * :doc:`/how_to_guides/run_on_slurm`
   * :doc:`/how_to_guides/run_repeats_in_parallel`
   * :doc:`/how_to_guides/run_with_mps`

See Also
--------

- :ref:`userguide_results` - details on inspecting these results.
- :ref:`cli-reference` - full CLI reference for ``openfe quickrun``
- :ref:`rbfe_cli_tutorial` - a tutorial on how to use the CLI to run hybrid topology relative binding free energy calculations.
