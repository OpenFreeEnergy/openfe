.. _userguide_worker:

Task-based Execution
====================

In contrast to quickrun execution, task-based execution does not require that you explicitly define the Transformation to be executed.
In worker-based execution, an OpenFE ``Worker`` is pointed to a ``Warehouse`` and ``TaskStatusDB`` which handle storage and orchestration, respectively.
This means that you can execute an entire ``AlchemicalNetwork``'s campaign just by calling the ``openfe worker`` command iteratively until all tasks are complete!


See below for details on how to run an openfe campaign using worker-based execution using either the CLI.

Setting up a Campaign
---------------------

If you are accustomed to using the existing ``openfe plan-rbfe-network`` or ``openfe plan-rhfe-network`` CLI commands, you can simply add the ``--warehouse`` argument to your existing call, and the Warehouse and TaskStatusDB needed for Worker-based execution will be created.

.. code:: bash

    > openfe plan-rbfe-network --warehouse

Alternatively, you can pass any ``AlchemicalNetwork`` (as a JSON file) to the ``setup-warehouse-db`` CLI command, and a Warehouse and TaskDB will be created with the same name.
This allows for more flexibility, as you can use any protocol (such as Separated Topologies).


.. code:: bash

    > openfe setup-warehouse-db --alchemical-network tyk2_alchemical_network.json


Either way, you should see a ``Warehouse`` directory and a ``TaskStatusDB`` file.

Running the Campaign
--------------------

To execute a single ``task`` (where here a ``task`` is one ``ProtocolUnit``), you can simply run:

.. code:: bash

    > openfe run-task tyk2/ tyk2.db


However, to run an entire campaign you would have to run this single command _many_ times.

In practice, you will likely be submitting many workers simultaneously using SLURM or similar.
You can call this command loop to automatically run a new worker after the previous has completed.

To run multiple workers in parallel, submit ``run_worker.sh`` multiple times.

.. literalinclude:: run_worker.sh
    :caption: Example SLURM submission script for worker-based execution
    :linenos:
    :language: bash

Gathering Results
-----------------

The ``Warehouse`` directory contains every ``ProtocolUnitResult`` created during execution.

Because Worker-based execution is currently under development, there is not yet a direct command to output the results.
To enable complete workflows in the meantime, we provide the ``output-as-legacy-json`` CLI command that takes in a Warehouse directory and outputs the results in a format identical to the format used by ``openfe quickrun``.

This enables use of ``openfe gather`` (and ``openfe gather-septop``, ``openfe gather-abfe``).

.. code:: bash

    > openfe output-as-legacy-json warehouse


The ``results`` directory may now be used as input to ``openfe gather``.

