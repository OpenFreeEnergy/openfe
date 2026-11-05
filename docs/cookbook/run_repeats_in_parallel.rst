.. _howto_run_repeats_in_parallel:

Run repeats in parallel
=======================

By default, ``openfe quickrun`` runs all repeats of a Transformation one after another (three by default for most protocols).
When simulations are long, you can instead run each repeat as a separate job.

Set the number of repeats per Transformation to 1
-------------------------------------------------

Each quickrun call then runs a single repeat. Set this when you plan the Transformations:

* with the CLI, pass ``--n-protocol-repeats 1`` to :ref:`cli_plan-rbfe-network` or :ref:`cli_plan-rhfe-network`, or
* in Python, set ``settings.protocol_repeats = 1`` before creating the Protocol (see :doc:`choose_protocol`).

Otherwise every job runs all repeats, and you end up with three times as many repeats as intended.

Submit one job per repeat
-------------------------

Each transformation can then be executed multiple times via the ``openfe quickrun`` command to produce a set of repeats.
However, **you must use unique results files for each repeat to ensure they don't overwrite each other**.
We recommend using folders named ``results_x`` where x is 0-2 to store the repeated calculations as our :ref:`openfe gather <cli_gather>` command also supports this file structure.

Below is an example of a simple script that will create and submit a separate job script (``\*.job`` named file) for every alchemical transformation (for the simplest SLURM use case) in a network running each repeat in parallel and writing the results to a unique folder:

.. code-block:: bash

   for file in network_setup/transformations/*.json; do
     relpath="${file:30}"  # strip off "network_setup/"
     dirpath=${relpath%.*}  # strip off final ".json"
     jobpath="network_setup/transformations/${dirpath}.job"
     if [ -f "${jobpath}" ]; then
       echo "${jobpath} already exists"
       exit 1
     fi
     for repeat in {0..2}; do
       cmd="openfe quickrun ${file} -o results_${repeat}/${relpath} -d results_${repeat}/${dirpath}"
       echo -e "#!/usr/bin/env bash\n${cmd}" > "${jobpath}"
       sbatch "${jobpath}"
     done
   done

Each repeat must write to its own results file.
quickrun only checks for an existing results file when it starts, so jobs that run at the same time with the same ``-o``
overwrite each other's results.

This gives one directory per repeat, each with a results file and a work directory for every Transformation:

.. code-block:: text

   results_0/
   ├── rbfe_lig_ejm_31_complex_lig_ejm_42_complex.json
   ├── rbfe_lig_ejm_31_complex_lig_ejm_42_complex/
   │   ├── shared_HybridTopologySetupUnit-<key>_attempt_0/
   │   ├── shared_HybridTopologyMultiStateSimulationUnit-<key>_attempt_0/
   │   └── shared_HybridTopologyMultiStateAnalysisUnit-<key>_attempt_0/
   └── ...
   results_1/
   └── ...
   results_2/
   └── ...

Gather the results
------------------
Pass the repeat directories to :ref:`openfe gather <cli_gather>`, for example as ``openfe gather results_*``.
It searches them recursively and combines the repeats of each Transformation.

.. TODO: link the "Gather free energy results" how-to instead of the CLI reference once it exists.

See also
--------

* :doc:`run_on_slurm` – submitting quickrun jobs to a cluster.
* :doc:`run_with_mps` – running several jobs on one GPU.