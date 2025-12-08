.. _cli_gather:

``gather`` command
====================

Currently, ``openfe gather`` is only able to gather results from Relative Binding Free Energy (RBFE) calculations.

To gather results from ABFE or SepTop protocols, you may use the experimental :ref:`openfe gather-abfe <gather-abfe>` and  :ref:`openfe gather-septop <gather-septop>` CLI commands, but please note that these commands are still under development and liable to change in future releases, and meant to be used only for exploratory work.

.. click:: openfecli.commands.gather:gather
   :prog: openfe gather


.. _gather-abfe:

.. click:: openfecli.commands.gather_abfe:gather_abfe
   :prog: openfe gather-abfe


.. _gather-septop:

.. click:: openfecli.commands.gather_septop:gather_septop
   :prog: openfe gather-septop
