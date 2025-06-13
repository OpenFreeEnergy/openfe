OpenMM Separated Topologies Protocol
====================================

.. _septop protocol api:

This section provides details about the OpenMM Separated Topologies Protocol
implemented in OpenFE.

Protocol API specification
--------------------------

.. module:: openfe.protocols.openmm_septop

.. autosummary::
   :nosignatures:
   :toctree: generated/

   SepTopProtocol
   SepTopComplexSetupUnit
   SepTopComplexRunUnit
   SepTopSolventSetupUnit
   SepTopSolventRunUnit
   SepTopProtocolResult

Protocol Settings
-----------------

Below are the settings which can be tweaked in the protocol. The default settings (accessed using :meth:`SepTopProtocol.default_settings`) will automatically populate settings which we have found to be useful for running a Separated Topologies free energy calculation. There will however be some cases (such as when calculating difficult to converge systems) where you will need to tweak some of the following settings.


.. module:: openfe.protocols.openmm_septop.equil_septop_settings

.. autopydantic_model:: SepTopSettings
   :model-show-json: False
   :model-show-field-summary: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :exclude-members: get_defaults
   :member-order: bysource


Protocol Specific Settings Classes
----------------------------------

Below are Settings classes which are unique to the `SepTopProtocol`.


.. autopydantic_model:: AlchemicalSettings
   :model-show-json: False
   :model-show-field-summary: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :member-order: bysource

.. autopydantic_model:: LambdaSettings
   :model-show-json: False
   :model-show-field-summary: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :member-order: bysource

.. autopydantic_model:: SepTopEquilOutputSettings
   :model-show-json: False
   :model-show-field-summary: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :member-order: bysource
