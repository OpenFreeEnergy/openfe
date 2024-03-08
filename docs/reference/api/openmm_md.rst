OpenMM Molecular Dynamics Protocol
==================================

This Protocol implements a standard MD execution for a solvated protein system.


Protocol API Specification
--------------------------

.. module:: openfe.protocols.openmm_md

.. autosummary::
   :nosignatures:
   :toctree: generated/

   PlainMDProtocol
   PlainMDProtocolResult


Protocol Settings
-----------------

.. module:: openfe.protocols.openmm_md.plain_md_settings

.. autopydantic_model:: PlainMDProtocolSettings
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

