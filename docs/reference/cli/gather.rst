.. _cli_gather:

``openfe gather``
=================

Currently, ``openfe gather`` is only able to gather results from Relative Binding Free Energy (RBFE) calculations.

To gather results from ABFE or SepTop protocols, you may use the experimental :ref:`openfe gather-abfe <gather-abfe>` and
:ref:`openfe gather-septop <gather-septop>` CLI commands, but please note that these commands are still under development
and liable to change in future releases, and meant to be used only for exploratory work.

Reports
-------

``ddg``
   The relative binding or hydration free energy of each ligand pair, ``DDG(i->j) = DG(j) - DG(i)``,
   and its uncertainty.

``dg`` (default)
   The free energy of each ligand, estimated from the relative free energies with a maximum likelihood estimator
   (see :ref:`userguide_MLE`). The values are centred around zero. The network must be connected and contain at
   least three edges, and every leg needs at least two repeats.

``raw``
   The free energy of every leg and repeat separately, with its MBAR uncertainty.

.. click:: openfecli.commands.gather:gather
   :prog: openfe gather


.. _gather-abfe:

.. click:: openfecli.commands.gather_abfe:gather_abfe
   :prog: openfe gather-abfe


.. _gather-septop:

.. click:: openfecli.commands.gather_septop:gather_septop
   :prog: openfe gather-septop
