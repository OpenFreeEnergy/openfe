OpenMM Absolute Solvation Free Energy Protocol
==============================================

.. _afe solvation protocol api:

This section provides details about the OpenMM Absolute Solvation Free Energy Protocol
implemented in OpenFE.

Protocol API specification
--------------------------

.. module:: openfe.protocols.openmm_afe.equil_solvation_afe_method

.. autosummary::
   :nosignatures:
   :toctree: generated/

   AbsoluteSolvationProtocol
   AbsoluteSolvationVacuumUnit
   AbsoluteSolvationSolventUnit
   AbsoluteSolvationProtocolResult

Protocol Settings
-----------------


Below are the settings which can be tweaked in the protocol. The default settings (accessed using :meth:`AbsoluteSolvationProtocol.default_settings`) will automatically populate settings which we have found to be useful for running solvation free energy calculations. There will however be some cases (such as when calculating difficult to converge systems) where you will need to tweak some of the following settings.


.. module:: openfe.protocols.openmm_afe.equil_afe_settings

.. autopydantic_model:: AbsoluteSolvationSettings
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

Below are Settings classes which are unique to the `AbsoluteSolvationProtocol`.


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
