
Alchemical Data Objects
-----------------------

Chemical Systems
~~~~~~~~~~~~~~~~

We describe a chemical system as being made up of one or more "components," e.g., solvent, protein, or small molecule. The :class:`.ChemicalSystem` object joins components together into a simulation system.

.. module:: openfe
   :noindex:

.. autosummary::
   :nosignatures:
   :toctree: generated/

	SmallMoleculeComponent
	ProteinComponent
	SolventComponent
	ChemicalSystem


Atom Mappings
~~~~~~~~~~~~~

Tools for mapping atoms in one molecule to those in another. Used to generate efficient ligand networks.

.. autosummary::
   :nosignatures:
   :toctree: generated/

	LigandAtomMapper
	LigandAtomMapping
   LigandNetwork


Alchemical Simulations
~~~~~~~~~~~~~~~~~~~~~~

Descriptions of anticipated alchemical simulation campaigns.

.. autosummary::
   :nosignatures:
   :toctree: generated/

	Transformation
	AlchemicalNetwork
