# Loading your data

One of the first tasks you'll likely want to do is loading your various input files.
In ``openfe`` the entire contents of a simulation volume, for example the ligand, protein and water is referred to
as the ``ChemicalSystem``.
A free energy difference is defined as being between two such ``ChemicalSystem`` objects.
To make expressing free energy calculations easier,
this ``ChemicalSystem`` is broken down into various ``Component`` objects.
It is these ``Component`` objects that are then transformed, added or removed when performing a free energy calculation.


:::{note}
Once chemical models are loaded into Components they are **read only** and cannot be modified.
This means that any modification/tweaking of the inputs must be done **before** the Component objects are created. 
This is done so that any data cannot be accidentally modified, ruining the provenance chain.
:::

As these all behave slightly differently to accomodate their contents,
there are specialised versions of Component to handle the different items in your system.
We will walk through how different items can be loaded,
and then how these are assembled to form ``ChemicalSystem`` objects.


## Small molecules

Small molecules, such as ligands, are handled using the ``openfe.SmallMoleculeComponent`` class.
These are lightweight wrappers around RDKit Molecules and can be created directly
from an rdkit molecule:

```python
from rdkit import Chem
import openfe

m = Chem.MolFromMol2File('myfile.mol2', removeHs=False)

smc = openfe.SmallMoleculeComponent(m, name='')
```

:::{caution}
Remember to include the `removeHs=False` keyword argument so that RDKit does not strip them!
:::

As these types of structures are typically stored inside sdf files, there is a ``from_sdf`` convenience class method:

```python
import openfe

smc = openfe.SmallMoleculeComponent.from_sdf_file('file.sdf')
```

## Proteins

Proteins are handled using an ``openfe.ProteinComponent``.
Like ``SmallMoleculeComponent``, these are based upon RDKit Molecules,
however these are expected to have the `MonomerInfo` struct present on all atoms.
This struct contains the residue and chain information and is essential to apply many popular force fields.
A "protein" here is considered as the fully modelled entire biological assembly, i.e. all chains and structural waters.

To load a protein, use the ``from_pdb_file`` or ``from_pdbx_file`` classmethod

```python
import openfe

p = openfe.ProteinComponent.from_pdb_file('file.pdb')
```


## Solvents

The bulk solvent phase is defined using a ``openfe.SolventComponent`` object.
Unlike the previously detailed Components, this does not have any explicit molecules or coordinates,
but instead represents the way that the overall system will be solvated.
This information is then interpreted inside the ``Protocol`` when solvating the system.

By default, this solvent is water.  The positive and negative ion can be defined, as well as the ion concentration
which must be specified along with the unit.

```python
import openfe
from openff.units import unit

solv = openfe.SolventComponent(ion_concentation=0.15 * unit.molar) 
```

## Assembling into ChemicalSystems

dict constructor

Box information

This chemical model has positions defined

This chemical model does not have any force field applied to it!