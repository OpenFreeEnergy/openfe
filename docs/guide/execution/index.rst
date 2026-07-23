.. _userguide_execution:

Execution
=========
The planning and preparation of a campaign of alchemical simulations using ``openfe`` is intended to be achievable on a local workstation in a matter of minutes.

The *execution* of these simulations however requires a large amount of computational power, and beyond running single calculations locally, is intended to be distributed across a HPC environment.

The simplest way to run a Transformation is to use the :ref:`quickrun CLI tool <userguide_quickrun>`.

More advanced options are available through first considering the
:ref:`theory of the execution model<userguide_execution_theory>`
then :ref:`reading on the available Python functions<reference_execution>`.

.. toctree::
   quickrun_execution
   exorcist_execution
   execution_theory
