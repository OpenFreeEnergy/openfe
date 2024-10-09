.. _guide-introduction:

Introduction 
============

Here we present an overview of the workflow for calculating free energies in
OpenFE in the broadest strokes possible. This workflow is reflected in both
the Python API and in the command line interface, and so we have a section
for each.

Workflow overview
-----------------

The overall workflow of OpenFE involves three stages:

1. **Setup**: Defining the simulation campaign you are going to run.
2. **Execution**: Running and performing initial analysis of your
   simulation campaign.
3. **Gather results**: Assembling the results from the simulation
   campaign for further analysis.

In many use cases, these stages may be done on different machines -- for
example, you are likely to make use of HPC or cloud computing resources to
run the simulation campaign. Because of this, each stage has a certain type
of output, which is the input to the next stage.

.. TODO make figure
.. .. figure:: ???
    :alt: Setup -> (AlchemicalNetwork) -> Execution -> (ProtocolResults) -> Gather

    The main stages of a free energy calculation in OpenFE, and the intermediates between them.

The output of **setup** is an :class:`.AlchemicalNetwork`. This contains all
the information about what is being simulated (e.g., what ligands, host proteins, solvation details etc) and the
information about how to perform the simulation (the Protocol).

The output of the **execution** stage is the basic results from each edge.
This can depend on the specific analysis intended, but will either involve a
:class:`.ProtocolResult` representing the calculated :math:`\Delta G` for
each edge or the :class:`.ProtocolDAGResult` linked to the data needed to
calculate that :math:`\Delta G`.

The **gather results** stage takes these results and produces something
useful to the user. For example, the CLI's ``gather`` command will create a
table of the :math:`\Delta G` for each leg.


CLI Workflow
------------

We have separate CLI commands for each stage of setup, running, and
gathering results. With the CLI, the Python objects of
:class:`.AlchemicalNetwork` and :class:`.ProtocolResult` are stored to disk
in an intermediate representation between the commands.

.. TODO make figure
.. .. figure:: ???
   :alt: [NetworkPlanner -> AlchemicalNetwork] -> Transformation JSON -> quickrun -> Result JSON -> gather

   The CLI workflow, with intermediates. The setup stage uses a network
   planner to generate the network, before saving each transformation as a
   JSON file.

The commands used to generate an :class:`AlchemicalNetwork` using the CLI are:

* :ref:`cli_plan-rbfe-network`
* :ref:`cli_plan-rhfe-network`

For example, you can create a relative binding free energy (RBFE) network using

.. code:: bash

    $ openfe plan-rbfe-network -p protein.pdb -M dir_with_sdfs/

These will save the alchemical network represented as a JSON file for each
edge of the :class:`.AlchemicalNetwork` (i.e., each leg of the alchemical cycle).

To run a given transformation, use the :ref:`cli_quickrun`; for example:

.. code:: bash

    $ openfe quickrun mytransformation.json -d dir_for_files -o output.json

In many cases, you will want to create a job script for a queuing system
(e.g., SLURM) that wraps that command. You can do this for all JSON files
from the network planning command with something like this:

.. TODO Link to example here. I think this is waiting on the CLI example
   being merged into example notebooks?

Finally, to gather the results of that, assuming all results (and only
results) are in the `results/` direcory, use the :ref:`cli_gather`:

.. code:: bash

    $ openfe gather ./results/ -o final_results.tsv

This will output a tab-separated file with the ligand pair, the estimated
:math:`\Delta G` and the uncertainty in that estimate.

The CLI provides a very straightforward user experience that works with the
most simple use cases. For use cases that need more workflow customization,
the Python API makes it relatively straightforward to define exactly the
simulation you want to run. The next sections of this user guide will
illustrate how to customize the behavior to your needs.
