
Alchemical Data Objects
-----------------------

Chemical Systems
~~~~~~~~~~~~~~~~

We describe a chemical system as being made up of one or more "components,"
e.g., solvent, protein, or small molecule. The :class:`.ChemicalSystem`
object joins them together.

.. autoclass:: openfe.SmallMoleculeComponent

.. autoclass:: openfe.ProteinComponent

.. autoclass:: openfe.SolventComponent

.. autoclass:: openfe.ChemicalSystem


Atom Mappings
~~~~~~~~~~~~~

.. autoclass:: openfe.LigandAtomMapper
           :members:

.. autoclass:: openfe.LigandAtomMapping
	       :members:


Alchemical Simulations
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: openfe.Transformation

.. autoclass:: openfe.AlchemicalNetwork
