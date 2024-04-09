.. _userguide_quickrun:

Execution with Quickrun
=======================

The planning and preparation of a campaign of alchemical simulations using the ``openfe`` package is intended to be
achievable on a local workstation in a matter of minutes.
The **execution** of these simulations however requires a large amount of computational power and is intended to be
distributed across a HPC environment.
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

  openfe quickrun transformation.json


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
