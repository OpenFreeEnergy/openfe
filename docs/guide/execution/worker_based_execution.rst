.. _userguide_worker:

Worker-based Execution
=======================

.. This tutorial demonstrates how to setup a campaign, run simulations, and gather results using **openfe**'s worker-based CLI execution.


Setting up a Campaign
---------------------

If you are used to using the existing ``openfe plan-rbfe-network`` or ``openfe plan-rhfe-network``, you can simply add the ``--warehouse`` argument, and a directory and task db will be created.

.. code:: bash

    > openfe plan-rbfe-network --warehouse


Alternatively, you pass any ``AlchemicalNetwork`` (as a JSON file) to the ``setup-warehouse-db`` CLI command, and a Warehouse and TaskDB will be created with the same name.

OR

.. code:: bash

    > openfe setup-warehouse-db --alchemical-network tyk2_alchemical_network.json


Running the Campaign
------------------------

To execute a single ``task`` (where a ``task`` is one ``ProtocolUnit``), you can simply run:

.. code:: bash

    > openfe run-worker tyk2.db tyk2.db


But, to run an entire campaign you would have to run this single command _many_ times.

In practice, you will likely be submitting many workers simultaneously using SLURM or similar.
You can call this command loop to automatically run a new worker after the previous has completed.

To run multiple workers in parallel, submit ``run_worker.sh`` multiple times.

... insert run_worker.sh

Gathering Results
-----------------

.. note::

    Because Worker-based execution is currently under development, there is not yet a

The `Warehouse` directory contains all the ``ProtocolUnitResult``s created during execution.
To enable use of ``openfe gather`` (and ``openfe gather-septop``, ``openfe gather-abfe``), we provide a helper functionality that takes in a Warehouse and outputs the results in a format identical to the format used by ``openfe quickrun``.


.. code:: bash

    > openfe output-as-legacy-json warehouse


The ``results`` directory may now be used as input to ``openfe gather``.

