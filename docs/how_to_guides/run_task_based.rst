.. _howto_run_task_based:

Run a network with task-based execution
=======================================

.. warning:: Task-based execution is an experimental feature and subject to change in future releases of openfe.

With task-based execution, you run a whole :class:`.AlchemicalNetwork` instead of one Transformation at a time.
For how it works, see the :ref:`userguide_task_based_execution` user guide.

1. Plan the network as a single AlchemicalNetwork JSON file, using ``--networks-only``:

   .. code-block:: bash

      openfe plan-rbfe-network -M ligands.sdf -p protein.pdb --networks-only -o network_setup

2. Create the Warehouse (``warehouse_<name>/``) and task database (``tasks_<name>.db``) for the campaign:

   .. code-block:: bash

      openfe setup-task-campaign --alchemical-network network_setup/network_setup.json --name my_campaign

3. Run tasks. Each invocation of ``openfe run-task`` claims and executes one available task.

   .. code-block:: bash

      openfe run-task --warehouse warehouse_my_campaign/ --task-db tasks_my_campaign.db --scratch scratch/

   For a network with many tasks, run multiple workers in parallel.
   For example, on a SLURM cluster, you can use a job array to start 15 workers,
   using the ``run_tasks.sh`` worker script from the CLI tutorial linked below:

   .. code-block:: bash

      sbatch --array=1-15 run_tasks.sh

   Each worker repeatedly calls ``openfe run-task`` until its wall-time limit is reached.
   All workers share the same Warehouse and task database, which coordinate task execution.

   See :ref:`task_based_execution_cli` for an example worker script.

4. Check the progress of the campaign:

   .. code-block:: bash

      openfe status --task-db tasks_my_campaign.db --summary

5. Once all tasks are complete, convert the results to JSON files that :ref:`openfe gather <cli_gather>` can read:

   .. code-block:: bash

      openfe to-legacy-json warehouse_my_campaign/ -o result_jsons

For a full walkthrough see the :ref:`CLI <task_based_execution_cli>` and
:ref:`Python API <tutorials/task_based_execution_python_api.nblink>` tutorials.
