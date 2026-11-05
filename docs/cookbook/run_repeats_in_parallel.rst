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

Each Transformation can be executed multiple times with ``openfe quickrun`` to produce independent repeats.

Each repeat must use its own results file and work directory.
We recommend using directories named ``results_0``, ``results_1``,
and ``results_2``, since :ref:`openfe gather <cli_gather>` supports
this directory structure.

On a SLURM cluster, you can extend the job script from :doc:`run_on_slurm` to accept the repeat number as a second argument.

Keep the SLURM resource requests and environment setup, and replace the final two lines with:

.. code-block:: bash

   name=$(basename "$1" .json)
   repeat=$2
   openfe quickrun "$1" -d "results_${repeat}/${name}" -o "results_${repeat}/${name}.json"

Then submit three jobs for each Transformation:

.. code-block:: bash

   for file in network_setup/transformations/*.json; do
     for repeat in 0 1 2; do
       sbatch run_quickrun.sh "$file" "$repeat"
     done
   done

Each job runs one repeat and writes its output to a separate directory.
Make sure the number of protocol repeats was set to 1 when planning the Transformations, as described above.

.. warning::

   Although quickrun does not start if the results file already
   exists, two jobs started concurrently with the same output path
   may both pass this check and overwrite each other's results.

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
