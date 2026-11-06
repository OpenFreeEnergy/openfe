.. _howto_gather_results:

Gather free energy results
==========================

Once your quickrun calculations have finished, ``openfe gather`` combines their results files into one table.
It supports relative binding and relative hydration free energies calculated with the Relative Hybrid Topology Protocol.
For other protocols, see `Separated Topologies and Absolute Binding results`_.

Pass the directory containing the results files, and choose a report:

.. code-block:: bash

   openfe gather results/ --report ddg -o ddg.tsv

``openfe gather`` searches directories recursively for results JSON files.
Files must end in ``.json``; unrelated JSON files are ignored.
Results can be spread over several directories, for example one per repeat
(see :doc:`run_repeats_in_parallel`).
You can also pass multiple directories or individual results files.

For the ``dg`` and ``ddg`` reports, repeats of the same ligand pair and leg are combined,
whether they are stored in one file or several.
The ``raw`` report keeps each repeat separate.

Choose a report
---------------

``--report ddg``
   The relative binding or hydration free energy of each ligand pair,
   ``DDG(i->j) = DG(j) - DG(i)``, and its uncertainty.

``--report dg`` (default)
   The free energy of each ligand in a network, estimated using a maximum likelihood estimator.
   These values are centred around zero.
   The network must be connected, contain at least three edges, and have
   at least two repeats per leg. See :ref:`userguide_MLE`.

``--report raw``
   The free energy of every leg and repeat separately, with its MBAR uncertainty (bootstrapped).
   Use this to check whether the repeats of a leg agree.

Without ``-o``, the table is printed to the terminal.
Add ``--tsv`` to print it as tab-separated values.

Missing legs and failed runs
----------------------------

If an edge is missing a leg, for example because its complex leg failed,
``openfe gather`` reports the incomplete edges and stops when generating
a ``dg`` or ``ddg`` report.

Use ``--allow-partial`` to gather the available results anyway:

.. code-block:: bash

   openfe gather results/ --report ddg --allow-partial -o ddg.tsv

Incomplete edges that cannot produce a relative free energy are marked as
``Error`` in the ``ddg`` report.
Failed runs may also be reported as ``Error`` if the ligand pair and leg can
be identified.

Use the ``raw`` report to inspect individual legs and repeats, including
failed runs, even when an edge is incomplete.

Separated Topologies and Absolute Binding results
-------------------------------------------------

To gather results from the Separated Topologies or Absolute Binding Free Energy Protocols,
use :ref:`openfe gather-septop <gather-septop>` or :ref:`openfe gather-abfe <gather-abfe>`.
They accept similar options, but ``gather-abfe`` only has the ``dg`` and ``raw`` reports.
Both commands are experimental and may change in future releases.

See also
--------

* :ref:`cli_gather` – all gather options.
* :doc:`/how_to_guides/plotting_with_cinnabar` – plotting the results against experimental data.
