.. _howto_run_quickrun:

Run a transformation with quickrun
==================================

``openfe quickrun`` takes a :class:`.Transformation` saved as JSON and runs it.
If you planned your network with :ref:`cli_plan-rbfe-network` or :ref:`cli_plan-rhfe-network`,
you already have one JSON file per Transformation.
From Python, you can write them yourself with :meth:`.Transformation.to_json` (see :doc:`dumping_transformation`).

How much of the calculation a single file covers depends on the protocol.
With the Relative Hybrid Topology Protocol, each file is one leg of an edge, such as the complex or the solvent leg,
so you need to run both legs of every edge.
With the Separated Topologies, Absolute Binding and Absolute Solvation Protocols, one file holds the whole
thermodynamic cycle.

To run a Transformation:

.. code-block:: bash

   openfe quickrun transformation.json -d workdir/ -o workdir/results.json

``-d`` / ``--work-dir``
   Where quickrun writes the simulation files: system setup, trajectories, checkpoints and analysis output.
   The directory is created if it doesn't exist, and defaults to the current directory.

``-o``
   Where quickrun writes the results JSON file. Any missing parent directories are created for you.
   By default, the file goes in the work directory as ``<transformation key>_results.json``.

You can point several Transformations at the same work directory, since each task writes to its own subdirectory.
Each Transformation does need its own results file, and quickrun won't start if the results file already exists.

All repeats of the Transformation (set by ``protocol_repeats``) run one after another within the same quickrun call.
To run them as separate jobs instead, see :doc:`run_repeats_in_parallel`.

When the run finishes, quickrun prints the free energy estimate and its uncertainty and writes them to the results file.
To collect the results of many Transformations, see :ref:`openfe gather <cli_gather>`.

If the run fails or is interrupted, :doc:`resume_quickrun` explains what to do next.

.. TODO: link the "Gather free energy results" how-to instead of the CLI reference once it exists

See also
--------

* :ref:`userguide_quickrun` – how quickrun executes a Transformation.
* :doc:`run_on_slurm` – submitting quickrun jobs to a cluster.
* :ref:`cli_quickrun` – all quickrun options.
