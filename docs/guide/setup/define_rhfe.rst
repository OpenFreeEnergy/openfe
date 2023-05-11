.. _define-rsfe:

Defining Relative Hydration Free Energy Calculations
====================================================

Relative hydration free energy calculations are very similar to
:ref:`relative binding free energy calculations <define-rbfe>`. The
main difference is that there is no protein component.

You can easily set up an :class:`.AlchemicalNetwork` for an RHFE with the
:class:`.RHFEAlchemicalNetworkPlanner`.

Just as with the RBFE, the RHFE involves setting up a
:class:`.LigandNetwork` and a :class:`.ChemicalSystem` for each ligand, and
then using these (along with the provided :class:`.Protocol`) to create the
associated :class:`.AlchemicalNetwork`.

To customize beyond what the RHFE planner can do, many of the same documents
that help with customizing RBFE setups are also relevant:

* :ref:`define_ligand_network`
