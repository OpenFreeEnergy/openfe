.. _userguide_chemicalsystems_and_components:

ChemicalSystems, Components and thermodynamic cycles
====================================================

When thinking about computing estimates in the free energy differences between different states,
we need an object model to represent the chemical models of the chemistry we are considering.
In ``openfe`` a :class:`.ChemicalSystem` is used to capture this information.
and represents

A ``ChemicalSystem`` **does** include information, where present, on:

* exact protonation state of both the protein and ligand
* the exact pose of a ligand or conformation of a protein
* the presence of any crystallographic waters

It **does not** include any information on:

* forcefield applied to any component, including details on water model or virtual particles
* thermodynamic conditions, i.e. temperature and pressure

Components
----------

A ``ChemicalSystem`` is composed of many ``Component`` objects,
each representing a single ''piece'' of the overall system.
Examples of components include:

* :class:`.ProteinComponent`
* :class:`.SmallMoleculeComponent`
* :class:`.SolventComponent`

Splitting the total system into components serves two purposes:

* the thermodynamic cycles can be easily understood by comparing the differences in Components
* ``Protocol`` \s can know to treat different components differently, for example applying different force fields

Thermodynamic Cycles
--------------------

With

< image of solvation free energy >


< image of rbfe >

See Also
--------

* To see how to construct ``ChemicalSystem`` \s from your files, see :ref:`the cookbook entry on loading molecules <Loading Molecules>`
* For details of what thermodynamic cycles to construct, consult the pages for each specific ``Protocol``