.. _userguide_quickrun:

Execution with Quickrun
=======================

The planning and preparation of a campaign of alchemical simulations using the ``openfe`` package is intended to be
achievable on a local workstation in a matter of minutes.
The **execution** of these simulations however requires a large amount of computational power,
and beyond running single calculations locally,
is intended to be distributed across a HPC environment.
Doing this requires storing and sending the details of the simulation from the local workstation to a HPC environment,
this can be done via the :func:`.Transformation.dump` function which
:ref:`creates a saved "json" version of the data<dumping_transformations>`.
These serialised "json" files are the currency of executing a campaign of simulations,
and contain all the information required to execute a single simulation.

To read this information and execute the simulation, the command line interface provides a ``quickrun`` command,
the full details of which are given in :ref:`the CLI reference section<cli_quickrun>`.
Briefly, this command takes a "json" simulation as an input and will then execute the simulation contained within,
therefore this command would execute a simulation saved to a file called "transformation.json".

::

  openfe quickrun transformation.json -o results.json


Which will produce a results file called ``results.json``.

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
 2. Using the ``openfe plan-rhfe-network`` (or ``plan-rbfe-network``) command line flag ``--n-protocol-repeats``.  Each transformation can then be executed multiple times via the
``openfe quickrun`` command to produce a set of repeats, however, you need to ensure to use unique results
files for each repeat to ensure they don't overwrite each other. We recommend using folders named ``results_x`` where x is 0-2
to store the repeated calculations as our :ref:`openfe gather <cli_gather>` command also supports this file structure.

Here is an example of a simple script that will create and submit a separate job script (\*.job named file)
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
    │   ├── easy_rbfe_lig_ejm_31_complex_lig_ejm_42_complex
    │   │   └── shared_RelativeHybridTopologyProtocolUnit-79c279f04ec84218b7935bc0447539a9_attempt_0
    │   │       ├── checkpoint.nc
    │   │       ├── simulation.nc
    │   ├── easy_rbfe_lig_ejm_31_complex_lig_ejm_42_complex.json
    ├── results_1
    │   ├── easy_rbfe_lig_ejm_31_complex_lig_ejm_42_complex
    │   │   └── shared_RelativeHybridTopologyProtocolUnit-a3cef34132aa4e9cbb824fcbcd043b0e_attempt_0
    │   │       ├── checkpoint.nc
    │   │       ├── simulation.nc
    │   ├── easy_rbfe_lig_ejm_31_complex_lig_ejm_42_complex.json
    └── results_2
        ├── easy_rbfe_lig_ejm_31_complex_lig_ejm_42_complex
        │   └── shared_RelativeHybridTopologyProtocolUnit-abb2b104151c45fc8b0993fa0a7ee0af_attempt_0
        │       ├── checkpoint.nc
        │       ├── simulation.nc
        └── easy_rbfe_lig_ejm_31_complex_lig_ejm_42_complex.json

The results of which can be gathered from the CLI using the ``openfe gather`` command, in this case you should direct
it to the root directory which includes the repeat results and it will automatically collate the information

::

 openfe gather results_parallel

See Also
--------

For details on inspecting these results, refer to :ref:`userguide_results`.
