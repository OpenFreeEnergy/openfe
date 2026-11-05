.. _howto_run_quickrun:

Run a transformation with quickrun
==================================

``openfe quickrun`` runs a single :class:`.Transformation` from its JSON file.
You can create these files with :ref:`cli_plan-rbfe-network` or :ref:`cli_plan-rhfe-network`,
or from Python with :meth:`.Transformation.to_json` (see :doc:`dumping_transformation`).

What one file covers depends on the protocol:

* For the Relative Hybrid Topology Protocol, each file is one leg of an edge, for example its complex or its solvent leg,
  so you need to run every leg.
* For the Separated Topologies, Absolute Binding and Absolute Solvation Protocols, one file contains the whole
  thermodynamic cycle.

To run a Transformation:

.. code-block:: bash

   openfe quickrun transformation.json -d workdir/ -o workdir/results.json

``-d`` / ``--work-dir``
   The directory for the simulation files (system setup, trajectories, checkpoints and analysis).
   It is created if it does not exist. If omitted, the current directory is used.

``-o``
   The path of the results JSON file. Its parent directory is created if needed.
   If omitted, the results are written to ``<work-dir>/<transformation key>_results.json``.

Several Transformations can share a work directory, because each task writes to its own subdirectory.
Each Transformation needs its own results file, however: quickrun refuses to start if the results file already exists.

All repeats of the Transformation (set by ``protocol_repeats``) run one after another within the same quickrun call.
To run them as separate jobs instead, see :doc:`run_repeats_in_parallel`.

When the run finishes, quickrun prints the free energy estimate and its uncertainty and writes them to the results file.
To collect the results of many Transformations, see :ref:`openfe gather <cli_gather>`.

If the run fails or is interrupted, see :doc:`recover_quickrun`.

.. TODO: link the "Gather free energy results" how-to instead of the CLI reference once it exists

See also
--------

* :ref:`userguide_quickrun` – how quickrun executes a Transformation.
* :doc:`run_on_slurm` – submitting quickrun jobs to a cluster.
* :ref:`cli_quickrun` – all quickrun options.
