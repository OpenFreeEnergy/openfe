.. _workflow:

Calculating Free Energies with OpenFE
=====================================

Here we present the workflow for calculating free energies in OpenFE, in the
broadest strokes possible This workflow is reflected in both the Python API
and in the command line interface, and so we have a section for each.

Workflow overview
-----------------

The overall workflow of OpenFE involves three stages:

1. **Setup**: Defining the simulation campaign you are going to run.
2. **Executation**: Running and performing initial analysis of your
   simulation campaign.
3. **Gather results**: Assembling the results from the simulation
   campaign for further analysis.

In many use cases, these stages may be done on different machines -- for
example, you are likely to make use of HPC or cloud computing resources to
run the simulation campaign. Because of this, each stage has a certain type
of output, which is the input to the next stage.

.. .. figure:: ???
    :alt: Setup -> (AlchemicalNetwork) -> Execution -> (ProtocolResults) -> Gather

    The main stages of a free energy calculation in OpenFE, and the intermediates between them.

The output of **setup** is an :class:`.AlchemicalNetwork`. This contains all
the information about what is being simulated (e.g., what ligands) and the
information about how to perform the simulation (the protocol).

The output of the **executation** stage is the basic results from each edge;
the ???

The **gather results** stage ???


CLI Workflow
------------

We have separate CLI commands for each stage of setup, running, and
gathering results. With the CLI, the Python objects of
:class:`.AlchemicalNetwork` and :class:`.ProtocolResult` are stored to disk
in an intermediate representation between the commands.

.. .. figure:: ???
   :alt: [NetworkPlanner -> AlchemicalNetwork] -> Transformation JSON -> quickrun -> Result JSON -> gather

   The CLI workflow, with intermediates. The setup stage uses a network
   planner to generate the network, before saving each transformation as a
   JSON file.

The commands used for set up the CLI are:

* the :ref:`cli_plan-rbfe-network`
* the :ref:`cli_plan-rhfe-network`

These will save the alchemical network represented as a JSON file for each
edge of the :class:`.AlchemicalNetwork` (i.e., each leg of the simulation).

To run a given transformation, use the :ref:`cli_quickrun`; for example:

.. code:: bash

    $ openfe quickrun mytransformation.json -d dir_for_files -o output.json

In many cases, you will want to create a job script for a queuing system
(e.g., SLURM) that wraps that command. You can do this for all JSON files
from the network planning command with something like this:

.. TODO

Finally, to gather the results of that, use the ref:`cli_gather`:

.. code:: bash

    $ openfe gather -o 

This will output a tab-separated file with the ligand pair, the 


All stages of this can be easily customized via the Python API. The
following sections will provide an overview of how customize your
simulations.
