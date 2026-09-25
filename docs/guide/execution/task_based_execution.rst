.. _userguide_task_based_execution:

.. module:: openfe
    :noindex:

Task-based Execution
====================

In contrast to :ref:`quickrun execution <userguide_quickrun>`, task-based execution does not require that you explicitly define the ``Transformation`` to be executed.

Instead, an :class:`AlchemicalNetwork` is used to create a series of tasks corresponding to each transformation, with each task representing a single :class:`ProtocolUnit` to be executed.

.. add a figure relating Transformation -> ProtocolDAG -> ProtocolUnits?

This means that you can execute an entire ``AlchemicalNetwork``\'s campaign just by calling the ``openfe run-task`` command iteratively until all tasks are complete, without needing to track specific Transformation JSON files.

See below for details on how to run an openfe campaign using task-based execution using either the CLI or the Python API.

Task-based Execution with the CLI
---------------------------------

Setting up a Campaign
~~~~~~~~~~~~~~~~~~~~~

An ``AlchemicalNetwork`` will be our input for executing a campaign.
Refer to the cookbook `Create an AlchemicalNetwork <../../cookbook/create_alchemical_network.nblink>`_ for guidance on getting to this step.

If you are accustomed to using the existing ``openfe plan-rbfe-network`` or ``openfe plan-rhfe-network`` CLI commands to create, you can simply add the ``--networks-only`` argument to your existing call, and use the output ``AlchemicalNetwork`` (``tyk2.json`` here) as a starting point.

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


You should see a ``Warehouse`` (``warehouse_tyk2/``) in the form of a directory and a ``TaskStatusDB`` (``tasks_tyk2.db``) file as output.

.. code:: bash

    > ls
    warehouse_tyk2/    tasks_tyk2.db

The ``TaskStatusDB`` is the source of truth for tracking the execution status of the tasks in this campaign.

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
    │ BLOCKED          │       8 │
    │ AVAILABLE        │       4 │
    │ IN_PROGRESS      │       0 │
    │ COMPLETED        │       0 │
    │ TOO_MANY_RETRIES │       0 │
    │ ERROR            │       0 │
    └──────────────────┴─────────┘

.. warning ::

    The ``Warehouse`` is a directory with a specific structure, and **should not be edited manually**.

At this stage, we can see that the following directories have been populated:

- ``setup/``: Stores the input :class:`AlchemicalNetwork`, deconstructed into its sub-components.
- ``tasks/``: Stores the :class:`ProtocolUnit` tasks that are to be executed by ``openfe run-task``.
- ``protocol_dags/``: Stores the :class:`ProtocolDAG`\s that the task :class:`ProtocolUnit`/s correspond to. These can be thought of as "bookkeeping" done by **openfe**.

Note that ``results/`` and ``shared/`` are empty, as no tasks have been executed yet.

.. code:: bash

    > tree warehouse_tyk2/
    ├── protocol_dags/
    │   ├── ProtocolDAG-62fb10eb3a78ec5568e5992688dd0b4a
    │   ├── ProtocolDAG-901e038b0360e59c930dcfed633ff72a
            ...
    ├── results/
    ├── setup/
    │   ├── AlchemicalNetwork-f7b23b93aebaa38a4fa6282ed9a2ee9b
    │   ├── ChemicalSystem-042fade29db8a39616f5a902e2da7ae3
    │   ├── ChemicalSystem-092349f47625178f21bc28888fc27d6a
            ...
    │   ├── LigandAtomMapping-0cdd365bd42a6a6dcc223a297aeba078
            ...
    │   ├── ProteinComponent-ee7ec9e5f1904266e3b388963bdcfe3c
    │   ├── RelativeHybridTopologyProtocol-59d604ce336f12d44f6e4647a6f7e616
    │   ├── SmallMoleculeComponent-8f5b6a9848f63293c9b2361b51b466ab
            ...
    │   ├── SolventComponent-26b4034ad9dbd9f908dfc298ea8d449f
    │   ├── Transformation-622f1d19bea91cd47a7e478e42376305
    │   ├── Transformation-a8ccb7f841aafdc860143dad3f2bd82a
    │   ├── Transformation-ae340a619b2794a9a3da4e72ef977885
    │   └── Transformation-d51c5a0397bfe45865a78edd1935e3b0
    ├── shared/
    └── tasks/
        ├── HybridTopologyMultiStateAnalysisUnit-1fda3bd83b584e0f9b4c5a511c04634d
        ├── HybridTopologyMultiStateAnalysisUnit-b28f3adb441a458c8b8be15fb44f6eaa
            ...
        ├── HybridTopologyMultiStateSimulationUnit-1f012c080bdc49e98695d2897b789c4f
        ├── HybridTopologyMultiStateSimulationUnit-4948383ff0d84b89b8d214dfff7c260b
            ...
        ├── HybridTopologySetupUnit-1714964f74c14753992582208540338c
        ├── HybridTopologySetupUnit-2732c040c9fa47b8bec6717da38ab839
            ...

.. note::

    If you're migrating from Transformation-based execution with quickrun, know that the ``Warehouse`` directory contains all the information that a directory of transformation.json files store, just in a different directory structure.


Running the Campaign
~~~~~~~~~~~~~~~~~~~~

To execute the campaign all we need is the ``Warehouse`` and a ``TaskStatusDB`` which handle storage and orchestration, respectively.

To execute a single ``task`` (where here a ``task`` is one ``ProtocolUnit``), you can simply run:

.. code:: bash

    > openfe run-task --warehouse warehouse_tyk2/ --task-db tasks_tyk2.db

**openfe** finds next available ``task`` in the ``TaskStatusDB``, retrieves the necessary data from the ``Warehouse`` to execute the task, then executes the task.

Now, you will see that a ``scratch/`` directory has been created locally, which is needed for quick read/write operations during execution.



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

As units are executed, you'll see rest of the ``Warehouse`` directories be populated:

- ``shared``: Data that may be shared between ProtocolUnits.
- ``results``: :class:`ProtocolUnitResult`\s corresponding 1:1 with the :class:`ProtocolUnit`\s in the ``tasks/`` directory.

.. code:: bash

    > tree warehouse_tyk2/shared warehouse_tyk2/results
    shared/
    ├── protocol_unit_results/
    │   ├── HybridTopologyMultiStateAnalysisUnit-69fe776a38d8498b9ddf44b88ca41825
    │   ├── HybridTopologyMultiStateAnalysisUnit-7fc2904754f54e9eb536f5edf0db6dc3
        ...
    │   ├── HybridTopologyMultiStateSimulationUnit-0ca902f13ea143b0a6dae64f489c0fb9
    │   ├── HybridTopologyMultiStateSimulationUnit-aa6efcc8b1e743e78d873bd8958e9541
        ...
    │   ├── HybridTopologySetupUnit-1093980e07b045cbaaec354177710e62
    │   ├── HybridTopologySetupUnit-580edcdab34a4b07bc8ba201f54c679d
        ...
    └── task_workdirs/
        ├── HybridTopologyMultiStateAnalysisUnit-69fe776a38d8498b9ddf44b88ca41825
        │   ├── forward_reverse_convergence.png
        │   ├── mbar_overlap_matrix.png
        │   ├── replica_exchange_matrix.png
        │   └── replica_state_timeseries.png
        ├── HybridTopologyMultiStateAnalysisUnit-7fc2904754f54e9eb536f5edf0db6dc3
            ...
        ├── HybridTopologyMultiStateSimulationUnit-0ca902f13ea143b0a6dae64f489c0fb9
        │   ├── checkpoint.chk
        │   └── simulation.nc
        ├── HybridTopologyMultiStateSimulationUnit-aa6efcc8b1e743e78d873bd8958e9541
            ...
        ├── HybridTopologySetupUnit-1093980e07b045cbaaec354177710e62
        │   ├── A_db.json
        │   ├── B_db.json
        │   ├── hybrid_positions.npy
        │   ├── hybrid_system.pdb
        │   └── hybrid_system.xml.bz2
        ├── HybridTopologySetupUnit-580edcdab34a4b07bc8ba201f54c679d
            ...
    results/
    ├── ProtocolUnitResult-21195221b4fd45a6bbb022936fd4d23b
    ├── ProtocolUnitResult-2726e87afe534c69b863bff9d141f9d7
        ...


To run an entire campaign this way, you would have to run ``openfe run-task`` *many* times.

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

.. code:: bash

    sbatch --array=1-50 run_tasks.sh


Gathering Results
~~~~~~~~~~~~~~~~~

In addition to the input data, the ``Warehouse`` directory contains every ``ProtocolUnitResult`` created during execution.

Because task-based execution is currently under development, there is not yet a direct command to output the results.
To enable complete workflows in the meantime, we provide the ``to-legacy-json`` CLI command that takes in a Warehouse directory and outputs the results in a format identical to the format used by ``openfe quickrun``.

This enables use of ``openfe gather`` (and ``openfe gather-septop``, ``openfe gather-abfe``).

.. code:: bash

    > openfe to-legacy-json warehouse_tyk2/ -o tyk2_result_jsons


The ``results`` directory may now be used as input to ``openfe gather``.

.. TODO: how to cross-link to openfe gather?

.. code:: bash

    > openfe gather tyk2_result_jsons/ --report=raw
    ┏━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
    ┃         ┃          ┃          ┃ DG(i->j)        ┃ MBAR uncertainty ┃
    ┃ leg     ┃ ligand_i ┃ ligand_j ┃ (kcal/mol)      ┃ (kcal/mol)       ┃
    ┡━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
    │ complex │ ligand_1 │ ligand_2 │ 0.0             │ 1.0              │
    │ solvent │ ligand_1 │ ligand_2 │ 3.0             │ 2.0              │
    │ complex │ ligand_2 │ ligand_3 │ 2.0             │ 2.0              │
    │ solvent │ ligand_2 │ ligand_3 │ 2.0             │ 1.0              │
    ...
