.. userguide_exorcist:

Execution with Exorcist Workers
===============================

Using the API to execute an Alchemical Network
----------------------------------------------

You can execute the network of simulation units defined by an ``AlchemicalNetwork`` (see :any:`create_alchemical_network`) using ``openfe.orchestration``:


First, we build a graph of tasks to be executed from the ``AlchemicalNetwork``:

.. code:: bash

    from openfe.orchestration import build_task_db_from_alchemical_network
    from openfe.storage import warehouse

    alchemical_network = openfe.AlchemicalNetwork.from_json("alchemicalNetwork.json")

    # create a Warehouse to define where simulation data is stored
    my_warehouse = FileSystemWarehouse()

    # store the AlchemicalNetwork in the Warehouse
    my_warehouse.store_setup_tokenizable(alchemical_network)

    # build a database of tasks from the AlchemicalNetwork
    db_path = Path(warehouse.root_dir) / "tasks.db"
    task_db = build_task_db_from_alchemical_network(alchemical_network, my_warehouse, db_path)


Next, we call ``worker.execute_unit()`` to execute the next available task in the warehouse:

.. code:: bash
    # execution: build the worker
    from openfe.orchestration import Worker

    worker = Worker(warehouse=my_warehouse, task_db_path=db_path)

    execution = worker.execute_unit(scratch=pathlib.Path("path/to/local/scratch"))


Each time ``worker.execute_unit`` is called, the worker will pick up the next valid task in the AlchemicalNetwork's task graph.


Using the CLI
-------------

.. Note: this is a proof-of-concept for use with RBFEs, tbd if we want to expose this right now.

.. code:: bash
    openfe plan-rbfe-network ... --warehouse


.. code:: bash
    openfe worker warehouse/

To run a single task to completion.
To take full advantage of the worker model, you can run multiple ``openfe worker`` commands concurrently.

The following is an example script that runs up to 4 workers at a time, with each automatically picking up the next valid unit to be executed.

.. code:: bash
    #!/usr/bin/env bash
    set -euo pipefail

    # Command + args as an array (so spaces are handled correctly)
    CMD=(openfe worker warehouse)

    MAX_JOBS=4
    pids=()

    cleanup() {
    echo "Stopping workers..."
    for pid in "${pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit 0
    }
    trap cleanup INT TERM

    run_job() {
    while true; do
        "${CMD[@]}"
    done
    }

    # Start initial workers
    for ((i=0; i<MAX_JOBS; i++)); do
    run_job &
    pids+=("$!")
    done

    # Keep 4 running; if one exits, start another
    while true; do
    wait -n
    run_job &
    pids+=("$!")
    done

