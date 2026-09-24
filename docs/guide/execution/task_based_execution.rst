.. _userguide_task_based_execution:

Task-based Execution
====================

In contrast to :ref:`quickrun execution <userguide_quickrun>`, task-based execution does not require that you explicitly define the Transformation to be executed.

.. define "task" here

This means that you can execute an entire ``AlchemicalNetwork``'s campaign just by calling the ``openfe run-task`` command iteratively until all tasks are complete!

.. include that ``task`` == ProtocolUnit, or is that confusing to non-dev users?

See below for details on how to run an openfe campaign using task-based execution using either the CLI or the Python API.

Task-based Execution with the CLI
---------------------------------

Setting up a Campaign
~~~~~~~~~~~~~~~~~~~~~

An ``AlchemicalNetwork`` will be our input for executing a campaign.
Refer to the cookbook `Create an AlchemicalNetwork <../../cookbook/create_alchemical_network.nblink>`_ for guidance on getting to this step.

If you are accustomed to using the existing ``openfe plan-rbfe-network`` or ``openfe plan-rhfe-network`` CLI commands to create, you can simply add the ``--networks-only`` argument to your existing call, and use the resulting ``alchemicalNetwork.json`` as a starting point.

.. code:: bash

    > openfe plan-rbfe-network -M ligands.sdf -p protein.pdb --networks-only -o tyk2
    ...
    > tree .
    tyk2/
    ├── ligand_network.graphml
    └── tyk2.json

Once you have an AlchemicalNetwork, use the following command to set up the task-based campaign.
By default, the ``TaskDB`` and ``Warehouse`` will be created using the input file basename (here, ``tyk2``), but you can pass in the ``--name`` parameter to define the identifier for the ``Warehouse`` and ``TaskDB`` file names.


.. code:: bash

    > openfe setup-task-campaign --alchemical-network tyk2/tyk2.json


You should see a ``Warehouse`` in the form of a directory and a ``TaskStatusDB`` file as output.

.. code:: bash

    > ls
    warehouse_tyk2/    tasks_tyk2.db


.. note::

    If you're migrating from Transformation-based execution with quickrun, know that the ``Warehouse`` directory contains all the information that a directory of transformation.json files store, just in a different structure.


Running the Campaign
~~~~~~~~~~~~~~~~~~~~

To execute the campaign all we need is the ``Warehouse`` and a ``TaskStatusDB`` which handle storage and orchestration, respectively.

At any time, you can query the execution status of the campaign using ``openfe status``.
All task status information is stored in the ``TaskStatusDB``:

.. code:: bash

    > openfe status --task-db tasks_ty2k.db
    ┌─────────────────────────────────┬───────────┬───────────────┬───────┬───────────┐
    │ taskid                          │ status    │ last_modified │ tries │ max_tries │
    ├─────────────────────────────────┼───────────┼───────────────┼───────┼───────────┤
    │ HybridTopologySetupUnit-025f3a… │ AVAILABLE │ NaT           │ 0     │ 1         │
    │ HybridTopologySetupUnit-70535a… │ AVAILABLE │ NaT           │ 0     │ 1         │
    │ HybridTopologySetupUnit-4eb000… │ AVAILABLE │ NaT           │ 0     │ 1         │
    │ HybridTopologySetupUnit-016551… │ AVAILABLE │ NaT           │ 0     │ 1         │
    │ HybridTopologyMultiStateSimula… │ BLOCKED   │ NaT           │ 0     │ 1         │
    │ HybridTopologyMultiStateSimula… │ BLOCKED   │ NaT           │ 0     │ 1         │
    ...


**Tip**: you can use the ``--summary`` flag to show a summary table of the number of tasks with each status:

.. code:: bash

    > openfe status --task-db tasks_ty2k.db
    ┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
    ┃ status           ┃ n_tasks ┃
    ┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
    │ BLOCKED          │       1 │
    │ AVAILABLE        │       0 │
    │ IN_PROGRESS      │       0 │
    │ COMPLETED        │      10 │
    │ TOO_MANY_RETRIES │       1 │
    │ ERROR            │       0 │
    └──────────────────┴─────────┘

.. TODO: explain task dag? maybe save that for developer docs?

To execute a single ``task`` (where here a ``task`` is one ``ProtocolUnit``), you can simply run:

.. code:: bash

    > openfe run-task --warehouse warehouse_tyk2/ --task-db tasks_tyk2.db

This finds next available ``task`` in the ``TaskStatusDB``, retrieves the necessary data from the ``Warehouse`` to execute the task, then executes the task.

You'll now see that one task has been completed, and a new task has been unblocked:

.. code:: bash

    > openfe status --task-db tasks_ty2k.db

    ┌───────────────────────────┬───────────┬─────────────────────┬───────┬───────────┐
    │ taskid                    │ status    │ last_modified       │ tries │ max_tries │
    ├───────────────────────────┼───────────┼─────────────────────┼───────┼───────────┤
    │ HybridTopologySetupUnit-… │ COMPLETED │ 2026-09-09 18:02:13 │ 1     │ 1         │
    │ HybridTopologySetupUnit-… │ AVAILABLE │ NaT                 │ 0     │ 1         │
    │ HybridTopologySetupUnit-… │ AVAILABLE │ NaT                 │ 0     │ 1         │
    │ HybridTopologySetupUnit-… │ AVAILABLE │ NaT                 │ 0     │ 1         │
    │ HybridTopologyMultiState… │ AVAILABLE │ 2026-09-09 18:02:13 │ 0     │ 1         │
    │ HybridTopologyMultiState… │ BLOCKED   │ NaT                 │ 0     │ 1         │
    ...

However, to run an entire campaign you would have to run ``openfe run-task`` _many_ times.

In practice, you will likely be submitting many workers simultaneously using Slurm or similar.
You can call this command in a loop, so that after a ``task`` is completed, the ``Worker`` automatically picks up a new ``task``, continuing to run tasks in serial until the walltime runs out.

.. code:: bash
   :caption: run_tasks.sh

    #!/bin/bash

    #SBATCH --job-name="openfe job"
    #SBATCH --mem-per-cpu=2G

    # activate an appropriate conda environment, or any "module load" commands required
    conda activate openfe_env

    # continue submitting run-task in serial until the wall time is hit
    # you may submit this *script* multiple times to have workers execute tasks in parallel
    while true; do
        openfe run-task --warehouse my_campaign/ --task-db my_campaign.db --scratch workdir/
    done


To run multiple workers in parallel, submit ``run_tasks.sh`` multiple times as separate jobs, for example using `Job Arrays on Slurm <https://slurm.schedmd.com/job_array.html>`_:

..code:: bash
    sbatch --array=1-50 run_tasks.sh


Gathering Results
-----------------

In addition to the input data, the ``Warehouse`` directory contains every ``ProtocolUnitResult`` created during execution.

Because task-based execution is currently under development, there is not yet a direct command to output the results.
To enable complete workflows in the meantime, we provide the ``to-legacy-json`` CLI command that takes in a Warehouse directory and outputs the results in a format identical to the format used by ``openfe quickrun``.

This enables use of ``openfe gather`` (and ``openfe gather-septop``, ``openfe gather-abfe``).

.. code:: bash

    > openfe to-legacy-json warehouse_tyk2/ -o tyk2_result_jsons


The ``results`` directory may now be used as input to ``openfe gather``.


