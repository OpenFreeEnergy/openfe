.. _Loading Molecules:

Loading your data into ChemicalSystems
======================================

One of the first tasks you'll likely want to do is loading your various input files.
In ``openfe`` the entire contents of a simulation volume, for example the ligand, protein and water is referred to
as the :class:`openfe.ChemicalSystem`.

A free energy difference is defined as being between two such :class:`ChemicalSystem` objects.
To make expressing free energy calculations easier,
this ``ChemicalSystem`` is broken down into various ``Component`` objects.
It is these ``Component`` objects that are then transformed, added or removed when performing a free energy calculation.

.. note::
   Once chemical models are loaded into Components they are **read only** and cannot be modified.
   This means that any modification/tweaking of the inputs must be done **before** the Component objects are created.
   This is done so that any data cannot be accidentally modified, ruining the provenance chain.


As these all behave slightly differently to accomodate their contents,
there are specialised versions of Component to handle the different items in your system.
We will walk through how different items can be loaded,
and then how these are assembled to form ``ChemicalSystem`` objects.


Loading small molecules
-----------------------

Small molecules, such as ligands, are handled using the :py:class:`openfe.SmallMoleculeComponent` class.
These are lightweight wrappers around RDKit Molecules and can be created directly
from an RDKit molecule:

.. code::

    from rdkit import Chem
    import openfe

    m = Chem.MolFromMol2File('myfile.mol2', removeHs=False)

    smc = openfe.SmallMoleculeComponent(m, name='')


.. warning::
    Remember to include the ``removeHs=False`` keyword argument so that RDKit does not strip your hydrogens!


As these types of structures are typically stored inside sdf files, there is a ``from_sdf_file`` convenience class method:

.. code::

    import openfe

    smc = openfe.SmallMoleculeComponent.from_sdf_file('file.sdf')


.. note::
   The ``from_sdf_file`` method will only read the first molecule in a multi-molecule MolFile.
   To load multiple molcules, use RDKit's ``Chem.SDMolSupplier`` to iterate over the contents,
   and create a ``SmallMoleculeComponent`` from each.


Loading proteins
----------------

Proteins are handled using an :class:`openfe.ProteinComponent`.
Like ``SmallMoleculeComponent``, these are based upon RDKit Molecules,
however these are expected to have the `MonomerInfo` struct present on all atoms.
This struct contains the residue and chain information and is essential to apply many popular force fields.
A "protein" here is considered as the fully modelled entire biological assembly,
i.e. all chains and structural waters and ions etc.

To load a protein, use the :func:`openfe.ProteinComponent.from_pdb_file` or :func:`openfe.ProteinComponent.from_pdbx_file` classmethod

.. code::

    import openfe

    p = openfe.ProteinComponent.from_pdb_file('file.pdb')


Defining solvents
-----------------

The bulk solvent phase is defined using a :class:`openfe.SolventComponent` object.
Unlike the previously detailed Components, this does not have any explicit molecules or coordinates,
but instead represents the way that the overall system will be solvated.
This information is then interpreted inside the ``Protocol`` when solvating the system.

By default, this solvent is water with 0.15 M NaCl salt.
All parameters; the positive and negative ion as well as the ion concentration (which must be specified along with the unit)
can be freely defined.

.. code::

    import openfe
    from openff.units import unit

    solv = openfe.SolventComponent(ion_concentation=0.15 * unit.molar)


Assembling into ChemicalSystems
-------------------------------

With individual components defined, we can then proceed to assemble combinations of these into
a description of an entire **system**, called a :class:`openfe.ChemicalSystem`.
The end result of this is a chemical model
which describes the chemical topology (e.g. bonds, formal charges) and atoms' positions
but does not describe the force field aspects, and therefore any energetic terms.

The input to the `ChemicalSystem` constructor is a dictionary mapping string labels (e.g. 'ligand' or 'protein') to individual Components.
The nature of these labels must match the labels that a given `Protocol` expects.
For free energy calculations we often want to describe two systems which feature many similar components
but differ in one component, which is the subject of the free energy perturbation.
For example we could define two `ChemicalSystem` objects which we could perform a relative binding free energy calculation between
as:

.. code::

    from openfe import ChemicalSystem, ProteinComponent, SmallMoleculeComponent, SolventComponent

    # define the solvent environment and protein structure, these are common across both systems
    sol = SolventComponent()
    p = ProteinComponent()

    # define the two ligands we are interested in
    m1 = SmallMoleculeComponent()
    m2 = SmallMoleculeComponent()

    # construct two systems, these only differ in the ligand input
    cs1 = ChemicalSystem({'ligand': m1, 'solvent': sol, 'protein': p})
    cs2 = ChemicalSystem({'ligand': m2, 'solvent': sol, 'protein': p})
