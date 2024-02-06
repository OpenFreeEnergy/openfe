OpenMM Absolute Solvation Free Energy Protocol
==============================================

This section provides details about the OpenMM Absolute Solvation Free Energy Protocol
implemented in OpenFE.

Protocol API specification
--------------------------

.. module:: openfe.protocols.openmm_afe.equil_solvation_afe_method

.. autosummary::
   :nosignatures:
   :toctree: generated/

   AbsoluteSolvationProtocol
   AbsoluteSolvationProtocolResult

Protocol Settings
-----------------


Below are the settings which can be tweaked in the protocol. The default settings (accessed using :meth:`AbsoluteSolvationProtocol.default_settings`) will automatically populate settings which we have found to be useful for running solvation free energy calculations. There will however be some cases (such as when calculating difficult to converge systems) where you will need to tweak some of the following settings.


.. module:: openfe.protocols.openmm_afe.equil_afe_settings

.. autopydantic_model:: OpenMMSystemGeneratorFFSettings
   :model-show-json: False
   :model-show-field-summary: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :member-order: bysource
   :noindex:

.. autopydantic_model:: ThermoSettings
   :model-show-json: False
   :model-show-field-summary: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :member-order: bysource
   :noindex:

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
   :noindex:

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
   :noindex:

.. autopydantic_model:: OpenMMEngineSettings
   :model-show-json: False
   :model-show-field-summary: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :member-order: bysource
   :noindex:

.. autopydantic_model:: IntegratorSettings
   :model-show-json: False
   :model-show-field-summary: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :member-order: bysource
   :noindex:

.. autopydantic_model:: MultiStateSimulationSettings
   :model-show-json: False
   :model-show-field-summary: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :member-order: bysource
   :noindex:
