OpenMM Relative Free Energy Protocol
====================================

This section provides details about the OpenMM Relative Free Energy Protocol
implemented in OpenFE.


Protocol Settings
-----------------

Below are the settings which can be tweaked in the protocol. The default settings
(accessed using :class:`RelativeHybridTopologyProtocol.default_settings` will
automatically populate a settings which we have found to be useful for running
relative binding free energies using explicit solvent. There will however be some
cases (such as when doing gas phase calculations) where you will need to tweak
some of the following settings.

.. autopydantic_settings:: openfe.protocols.openmm_rfe.equil_rfe_settings.RelativeHybridTopologyProtocolSettings
   :settings-show-json: False
   :settings-show-config-member: False
   :settings-show-config-summary: False
   :settings-show-validator-members: False
   :settings-show-validator-summary: False
   :field-list-validators: False

.. autopydantic_settings:: openfe.protocols.openmm_rfe.equil_rfe_settings.AlchemicalSamplerSettings
   :settings-show-json: False
   :settings-show-config-member: False
   :settings-show-config-summary: False
   :settings-show-validator-members: False
   :settings-show-validator-summary: False
   :field-list-validators: False

.. autopydantic_settings:: openfe.protocols.openmm_rfe.equil_rfe_settings.AlchemicalSettings
   :settings-show-json: False
   :settings-show-config-member: False
   :settings-show-config-summary: False
   :settings-show-validator-members: False
   :settings-show-validator-summary: False
   :field-list-validators: False

.. autopydantic_settings:: openfe.protocols.openmm_rfe.equil_rfe_settings.OpenMMEngineSettings
   :settings-show-json: False
   :settings-show-config-member: False
   :settings-show-config-summary: False
   :settings-show-validator-members: False
   :settings-show-validator-summary: False
   :field-list-validators: False

.. autopydantic_settings:: openfe.protocols.openmm_rfe.equil_rfe_settings.IntegratorSettings
   :settings-show-json: False
   :settings-show-config-member: False
   :settings-show-config-summary: False
   :settings-show-validator-members: False
   :settings-show-validator-summary: False
   :field-list-validators: False

.. autopydantic_settings:: openfe.protocols.openmm_rfe.equil_rfe_settings.SimulationSettings
   :settings-show-json: False
   :settings-show-config-member: False
   :settings-show-config-summary: False
   :settings-show-validator-members: False
   :settings-show-validator-summary: False
   :field-list-validators: False

.. autopydantic_settings:: openfe.protocols.openmm_rfe.equil_rfe_settings.SolvationSettings
   :settings-show-json: False
   :settings-show-config-member: False
   :settings-show-config-summary: False
   :settings-show-validator-members: False
   :settings-show-validator-summary: False
   :field-list-validators: False

.. autopydantic_settings:: openfe.protocols.openmm_rfe.equil_rfe_settings.SystemSettings
   :settings-show-json: False
   :settings-show-config-member: False
   :settings-show-config-summary: False
   :settings-show-validator-members: False
   :settings-show-validator-summary: False
   :field-list-validators: False

.. autopydantic_settings:: openfe.protocols.openmm_rfe.equil_rfe_settings.ThermoSettings
   :settings-show-json: False
   :settings-show-config-member: False
   :settings-show-config-summary: False
   :settings-show-validator-members: False
   :settings-show-validator-summary: False
   :field-list-validators: False

.. autopydantic_settings:: openfe.protocols.openmm_rfe.equil_rfe_settings.OpenMMSystemGeneratorFFSettings
   :settings-show-json: False
   :settings-show-config-member: False
   :settings-show-config-summary: False
   :settings-show-validator-members: False
   :settings-show-validator-summary: False
   :field-list-validators: False


Protocol API specification
--------------------------


.. autoclass:: openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol
     :members:

