.. _howto_recover_quickrun:

Recover from a failed or interrupted run
========================================

A quickrun job can stop early in two ways:

* **A task failed.** quickrun finished, but printed an error naming the failed task.
  The results file exists, with ``estimate`` and ``uncertainty`` set to ``null``.
* **The job was interrupted.** quickrun was stopped before it finished, for example by a wall-time limit,
  a node failure or a cancelled job. There is no results file, and a cache file
  ``<work-dir>/quickrun_cache/dag-cache-<key>.json`` is left behind.

Rerunning a failed task
----------------

If a task fails, quickrun retries it up to two more times, each in its own directory
(``..._attempt_1/`` and ``..._attempt_2/``).
If it still fails, quickrun stops, writes the results file with ``estimate`` and ``uncertainty`` set to ``null``,
and exits with an error naming the failed task.

The full traceback of each failed attempt is stored under ``unit_results`` in the results file.
To print it, use ``jq`` (see :doc:`jq_inspection`):

.. code-block:: bash

   jq -r '.unit_results[] | select(.__qualname__ == "ProtocolUnitFailure") | .name, .traceback' \
      results.json

Once you have fixed the cause (see :doc:`/guide/troubleshooting`), run the Transformation again with a new
work directory and results file.

quickrun refuses to start if the results file already exists.
For example, attempting to reuse ``results.json`` produces:

.. code-block:: text

   Error: Invalid value for '-o': Path 'results.json' is a file.

This error occurs because the ``-o`` option does not accept an existing file.
Choose a different output filename for the new run, or remove the existing results file.

``--resume`` only works for interrupted runs, because quickrun removes the cache once a task has failed.
By then the task has already been retried twice, so resuming would most likely fail in the same way.

.. _howto_quickrun_resume:

Resuming an interruped job
--------------------------

When ``openfe quickrun`` starts, it saves a plan of the simulation to a cache file before execution begins:

.. code-block:: text

   <work-dir>/quickrun_cache/dag-cache-<key>.json

where ``<key>`` is a unique identifier based on the ``-o`` file path and the Transformation.
This cache is removed once quickrun finishes, including after a failed task.

If a job is interrupted (e.g. due to a wall-time limit, node failure, or manual cancellation),
you can resume it by passing the ``--resume`` flag:

.. code-block:: bash

   openfe quickrun transformation.json -d workdir/ -o workdir/results.json --resume

quickrun then uses the cache to skip the tasks that had already finished.
The hybrid topology, separated topologies and absolute protocols continue an interrupted simulation from its last
checkpoint, so only the time since that checkpoint is run again.
How often checkpoints are written is set by :attr:`~.MultiStateOutputSettings.checkpoint_interval`
in the protocol's output settings (see :doc:`choose_protocol` for how to change settings).

.. note::

   The same ``-d`` / ``--work-dir`` and ``-o`` arguments used in the original run must be specified
   so that quickrun can locate the cache file.

If you pass ``--resume`` but no cache file is found (e.g. the job never started), the following warning is printed
and a new run begins:

.. code-block:: text

   openfe quickrun was run with --resume, but no cached results found at
   <path-to-cache-file>. Starting new execution.

If the cache file is corrupted (e.g. due to an incomplete write at the moment of interruption),
``quickrun --resume`` raises the following error.
In that case, start the run again in a new work directory.

.. code-block:: text

   Recovery failed, please remove <work-dir>/quickrun_cache/dag-cache-<key>.json
   before executing a new transformation simulation.

If you do not pass the ``--resume`` flag, quickrun detects the partially complete Transformation and refuses
to start, so that you do not accidentally start a duplicate run:

.. code-block:: text

   Transformation has been started but is incomplete. Please remove
   <work-dir>/quickrun_cache/dag-cache-<key>.json and rerun, or resume
   execution using the ``--resume`` flag.

See also
--------

* :doc:`run_quickrun` – running a Transformation with quickrun.
* :doc:`/guide/troubleshooting` – common errors and how to fix them.
