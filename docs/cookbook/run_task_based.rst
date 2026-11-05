.. _howto_run_task_based:

Run a network with task-based execution
=======================================

.. warning:: Task-based execution is an experimental feature and subject to change in future releases of openfe.

With task-based execution, you run a whole :class:`.AlchemicalNetwork` instead of one Transformation at a time.
For how it works, see :ref:`userguide_task_based_execution`.

1. Plan the network as a single AlchemicalNetwork JSON file, using ``--networks-only``:

   .. code-block:: bash

      openfe plan-rbfe-network -M ligands.sdf -p protein.pdb --networks-only -o network_setup

2. Create the Warehouse (``warehouse_<name>/``) and task database (``tasks_<name>.db``) for the campaign:

   .. code-block:: bash

      openfe setup-task-campaign --alchemical-network network_setup/network_setup.json --name my_campaign

3. Run tasks. Each call of ``openfe run-task`` claims and runs one task that is ready.
   Call it in a loop, and start several workers, for example as separate jobs, to run tasks in parallel:

   .. code-block:: bash

      openfe run-task --warehouse warehouse_my_campaign/ --task-db tasks_my_campaign.db --scratch scratch/

4. Check the progress of the campaign:

   .. code-block:: bash

      openfe status --task-db tasks_my_campaign.db --summary

5. Once all tasks are complete, convert the results to JSON files that :ref:`openfe gather <cli_gather>` can read:

   .. code-block:: bash

      openfe to-legacy-json warehouse_my_campaign/ -o result_jsons

For a full walkthrough, including a SLURM worker script, see the
:ref:`CLI <task_based_execution_cli>` and :ref:`Python API <tutorials/task_based_execution_python_api.nblink>` tutorials.