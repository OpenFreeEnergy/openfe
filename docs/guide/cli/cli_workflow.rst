
CLI Workflow
------------

We have separate CLI commands for each stage of setup, execution, and
gathering results. With the CLI, the Python objects of
:class:`.AlchemicalNetwork` and :class:`.ProtocolResult` are stored to disk
in an intermediate representation between the commands.

.. TODO make figure
.. .. figure:: ???
   :alt: [NetworkPlanner -> AlchemicalNetwork] -> Transformation JSON -> quickrun -> Result JSON -> gather

   The CLI workflow, with intermediates. The setup stage uses a network
   planner to generate the network, before saving each transformation as a
   JSON file.

The commands used to generate an :class:`.AlchemicalNetwork` using the CLI are:

* :ref:`cli_plan-rbfe-network`
* :ref:`cli_plan-rhfe-network`

.. note::

   To ensure a consistent set of partial charges are used for each molecule across different transformations, the CLI
   network planners will now automatically generate charges ahead of planning the network. The partial charge generation
   scheme can be configured using the :ref:`YAML settings <userguide_cli_yaml_interface>`. We also provide tooling to
   generate the partial charges as a separate CLI step which can be run before network planning, see the :ref:`tutorial <charge_molecules_cli_tutorial>`
   for more details.


For example, you can create a relative binding free energy (RBFE) network using

.. code:: bash

    $ openfe plan-rbfe-network -p protein.pdb -M dir_with_sdfs/

This will save the alchemical network represented as a JSON file for each
edge of the :class:`.AlchemicalNetwork` (i.e., each leg of the alchemical cycle).

To run a given transformation, use the :ref:`cli_quickrun`; for example:

.. code:: bash

    $ openfe quickrun mytransformation.json -d dir_for_files -o output.json

In many cases, you will want to create a job script for a queuing system
(e.g., SLURM) that wraps that command. You can do this for all JSON files
from the network planning command with something like this:

.. TODO Link to example here. I think this is waiting on the CLI example
   being merged into example notebooks?

Finally, assuming all results (and only results) are in the `results/` directory,
use the :ref:`cli_gather` to generate a summary table:

.. code:: bash

    $ openfe gather ./results/ -o final_results.tsv

This will output a tab-separated file with the ligand pair, the estimated
:math:`\Delta G` and the uncertainty in that estimate.

The CLI provides a very straightforward user experience that works with the
most simple use cases. For use cases that need more workflow customization,
the Python API makes it relatively straightforward to define exactly the
simulation you want to run. The next sections of this user guide will
illustrate how to customize the behavior to your needs.
