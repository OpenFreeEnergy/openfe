OpenMM Relative Free Energy Protocol
====================================

This section provides details about the OpenMM Relative Free Energy Protocol
implemented in OpenFE.

Protocol Settings
-----------------


Below are the settings which can be tweaked in the protocol. The default settings (accessed using :meth:`RelativeHybridTopologyProtocol.default_settings`) will automatically populate a settings which we have found to be useful for running relative binding free energies using explicit solvent. There will however be some cases (such as when doing gas phase calculations) where you will need to tweak some of the following settings.

.. module:: openfe.protocols.openmm_rfe.equil_rfe_settings

.. autopydantic_model:: RelativeHybridTopologyProtocolSettings
   :model-show-json: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :exclude-members: get_defaults

.. autopydantic_model:: OpenMMSystemGeneratorFFSettings
   :model-show-json: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel

.. autopydantic_model:: ThermoSettings
   :model-show-json: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel

.. autopydantic_model:: AlchemicalSamplerSettings
   :model-show-json: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel

.. autopydantic_model:: AlchemicalSettings
   :model-show-json: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel

.. autopydantic_model:: OpenMMEngineSettings
   :model-show-json: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel

.. autopydantic_model:: IntegratorSettings
   :model-show-json: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel

.. autopydantic_model:: SimulationSettings
   :model-show-json: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel

.. autopydantic_model:: SolvationSettings
   :model-show-json: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel

.. autopydantic_model:: SystemSettings
   :model-show-json: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel

Protocol API specification
--------------------------

.. module:: openfe.protocols.openmm_rfe

.. autoclass:: RelativeHybridTopologyProtocol
   :no-members:

   .. automethod:: default_settings

   .. automethod:: create

   .. automethod:: gather

.. autoclass:: RelativeHybridTopologyProtocolResult
