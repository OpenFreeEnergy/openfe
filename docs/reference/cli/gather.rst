.. _cli_gather:

``gather`` command
====================

Currently, ``openfe gather`` is only able to gather results from Relative Binding Free Energy (RBFE) calculations.

To gather results from ABFE or SepTop protocols, you may use the experimental ``openfe gather-abfe`` and ``openfe gather-septop`` CLI commands, but please note that these commands are still under development and liable to change in future releases, and should only be used for exploratory work.

.. click:: openfecli.commands.gather:gather
   :prog: openfe gather

.. click:: openfecli.commands.gather_septop:gather_septop
   :prog: openfe gather-septop

.. click:: openfecli.commands.gather_abfe:gather_abfe
   :prog: openfe gather-abfe
