OpenMM Protocol Settings
========================

This page documents settings re-used by several OpenMM-based Protocols.

Settings which are specific to a Protocol (e.g. LambdaSettings) are documented
in the Protocol's individual documentation page.


Shared OpenMM Protocol Settings
-------------------------------

.. module:: openfe.protocols.openmm_utils.omm_settings

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

.. autopydantic_model:: MDOutputSettings
   :model-show-json: False
   :model-show-field-summary: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :member-order: bysource

.. autopydantic_model:: MDSimulationSettings
   :model-show-json: False
   :model-show-field-summary: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :member-order: bysource

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

.. autopydantic_model:: OpenFFPartialChargeSettings
   :model-show-json: False
   :model-show-field-summary: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :member-order: bysource

.. autopydantic_model:: OpenMMSolvationSettings
   :model-show-json: False
   :model-show-field-summary: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :member-order: bysource

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


Shared MultiState OpenMM Protocol Settings
------------------------------------------

Protocol Settings shared between MultiState simulation protocols.

These currently include the OpenMM Absolute and Relative Free Energy Protocols.


.. seealso:
   :class:`openfe.protocol.openmm_rfe.RelativeHybridTopologyProtocol`
   :class:`openfe.prtoocol.openmm_afe.AbsoluteSolvationProtocol`


.. autopydantic_model:: MultiStateOutputSettings
   :model-show-json: False
   :model-show-field-summary: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :member-order: bysource

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

