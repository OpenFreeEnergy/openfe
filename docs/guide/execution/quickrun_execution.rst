.. _userguide_quickrun:

Execution with ``openfe quickrun``
==================================

While **openfe** intends to make the *planning* and *preparation* of an free energy campaign achievable on a local workstation in a matter of minutes,
the *execution* of these simulations requires significant computational power.
**openfe** is designed for distributing simulation execution across HPC environment(s).

The :ref:`Transformation.to_json()<dumping_transformations>` method JSON serializes all the information needed to run a ``Transformation`` to a JSON-formatted file, so that you can plan and prepare your campaign locally, then send these JSON files to an HPC to execute the simulations.

The ``openfe quickrun`` command is a simple way to execute a single transformation JSON. It takes in a JSON-serialized ``Transformation`` as input, then executes the simulation.

The following command would execute a simulation saved to a file called "transformation.json" and write the results of the simulation to ``results.json``.

::

  openfe quickrun transformation.json -o results.json


For full details, see :ref:`the CLI reference section<cli_quickrun>`.

Executing within a job submission script
----------------------------------------

It is likely that computational jobs will be submitted to a queueing engine, such as slurm.
The ``quickrun`` command can be integrated into as:

::

  #!/bin/bash

  #SBATCH --job-name="openfe job"
  #SBATCH --mem-per-cpu=2G

  # activate an appropriate conda environment, or any "module load" commands required to
  conda activate openfe_env

  openfe quickrun transformation.json -o results.json

Parallel execution of repeats with Quickrun
===========================================

Serial execution of multiple repeats of a transformation can be inefficient when simulation times are long.
Higher throughput can be achieved with parallel execution by running one repeat per HPC job. Most protocols are set up to
run three repeats in serial by default, but this can be changed by either:
 
 1. Defining the protocol setting ``protocol_repeats`` - see the :ref:`protocol configuration guide <cookbook/choose_protocol.nblink>` for more details.
 2. Using the ``openfe plan-rhfe-network`` (or ``plan-rbfe-network``) command line flag ``--n-protocol-repeats``.

Each transformation can then be executed multiple times via the
``openfe quickrun`` command to produce a set of repeats, however, you need to ensure to use unique results
files for each repeat to ensure they don't overwrite each other. We recommend using folders named ``results_x`` where x is 0-2
to store the repeated calculations as our :ref:`openfe gather <cli_gather>` command also supports this file structure.

Here is an example of a simple script that will create and submit a separate job script (``\*.job`` named file)
for every alchemical transformation (for the simplest SLURM use case) in a network running each repeat in parallel and writing the
results to a unique folder:

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
       cmd="openfe quickrun ${file} -o results_${repeat}/${relpath} -d results_${repeat}/${dirpath} --n-protocol-repeats 1"
       echo -e "#!/usr/bin/env bash\n${cmd}" > "${jobpath}"
       sbatch "${jobpath}"
     done
   done

This should result in the following file structure after execution:

::

    results_parallel/
    ├── results_0
    │   ├── rbfe_lig_ejm_31_complex_lig_ejm_42_complex
    │   │   └── shared_RelativeHybridTopologyProtocolUnit-79c279f04ec84218b7935bc0447539a9_attempt_0
    │   │       ├── checkpoint.nc
    │   │       ├── simulation.nc
    │   ├── rbfe_lig_ejm_31_complex_lig_ejm_42_complex.json
    ├── results_1
    │   ├── rbfe_lig_ejm_31_complex_lig_ejm_42_complex
    │   │   └── shared_RelativeHybridTopologyProtocolUnit-a3cef34132aa4e9cbb824fcbcd043b0e_attempt_0
    │   │       ├── checkpoint.nc
    │   │       ├── simulation.nc
    │   ├── rbfe_lig_ejm_31_complex_lig_ejm_42_complex.json
    └── results_2
        ├── rbfe_lig_ejm_31_complex_lig_ejm_42_complex
        │   └── shared_RelativeHybridTopologyProtocolUnit-abb2b104151c45fc8b0993fa0a7ee0af_attempt_0
        │       ├── checkpoint.nc
        │       ├── simulation.nc
        └── rbfe_lig_ejm_31_complex_lig_ejm_42_complex.json

The results of which can be gathered from the CLI using the ``openfe gather`` command, in this case you should direct
it to the root directory which includes the repeat results and it will automatically collate the information

::

 openfe gather results_parallel

See Also
--------

For details on inspecting these results, refer to :ref:`userguide_results`.
