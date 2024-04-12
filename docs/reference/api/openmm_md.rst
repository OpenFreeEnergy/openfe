OpenMM Molecular Dynamics (MD) Protocol
=======================================

.. _md protocol api:

A Protocol for running MD simulation using OpenMM.


Protocol API Specification
--------------------------

.. module:: openfe.protocols.openmm_md

.. autosummary::
   :nosignatures:
   :toctree: generated/

   PlainMDProtocol
   PlainMDProtocolUnit
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

