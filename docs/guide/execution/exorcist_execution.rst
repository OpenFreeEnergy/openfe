.. userguide_exorcist:

Execution with Exorcist Workers
===============================

Using the CLI
-------------

Note: this is a proof-of-concept for use with RBFEs, tbd if we want to expose this right now.

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

Using the API
-------------