.. _howto_run_on_slurm:

Run quickrun on a SLURM cluster
===============================

On a [SLURM](https://slurm.schedmd.com/) cluster, we recommend running each Transformation as its own SLURM job by using ``openfe quickrun`` in a job script.

For example, the script below runs the Transformation JSON file given as its argument on one GPU, and follows the convention of using the input file's basename as the output JSON's basename:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=openfe
   #SBATCH --gres=gpu:1
   #SBATCH --mem-per-cpu=2G
   #SBATCH --time=24:00:00

   # activate an appropriate conda environment, or run any "module load" commands required
   conda activate openfe_env

   name=$(basename "$1" .json)
   openfe quickrun "$1" -d "workdir/${name}" -o "results/${name}.json" --resume

Adjust the ``#SBATCH`` options to your cluster, for example to add a partition or account.

Save the script as ``run_quickrun.sh`` and submit one job for each Transformation JSON, either manually with

` > sbatch run_quickrun.sh </path/to/transformation.json>`

or by looping over all the Transformation JSONs in a directory.
For example, to submit jobs for all Transformations  written by :ref:`cli_plan-rbfe-network` to the directory `network_setup/transformations/`:

.. code-block:: bash

   for file in network_setup/transformations/*.json; do
     sbatch run_quickrun.sh "$file"
   done

Once all jobs have finished, collect the results from ``results/`` with :ref:`openfe gather <cli_gather>`.

.. TODO: link the "Gather free energy results" how-to instead of the CLI reference once it exists

The ``--resume`` flag lets you resubmit a job that hit its wall-time limit: it continues where it stopped.
On the first submission, when there is nothing to resume, quickrun prints a warning and starts a new run.
See :doc:`resume_quickrun` for details.

To run the repeats of each Transformation as separate jobs, see :doc:`run_repeats_in_parallel`.

See also
--------

* :doc:`run_quickrun` – running a Transformation with quickrun.
* :doc:`run_with_mps` – running several Transformations on one GPU.
