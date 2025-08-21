.. _jq_inspection:

Using ``jq`` to inspect OpenFE JSONS
==============================================

`jq <https://github.com/jqlang/jq>`_ is a helpful command-line tool for quickly inspecting JSON files.

.. code:: bash

    $ jq "keys[]" rbfe_lig_1a_lig_03.json
    ":version:"
    "__module__"
    "__qualname__"
    "mapping"
    "name"
    "protocol"
    "stateA"
    "stateB"

.. code:: bash

    $ jq ".protocol | keys[]" rbfe_lig_1a_lig_03.json
    ":version:"
    "__module__"
    "__qualname__"
    "settings"

.. code:: bash

    $ jq ".protocol.settings | keys[]" rbfe_lig_1a_lig_03.json
    ":is_custom:"
    "__class__"
    "__module__"
    "alchemical_settings"
    "complex_equil_output_settings"
    "complex_equil_simulation_settings"
    "complex_lambda_settings"
    "complex_output_settings"
    "complex_restraint_settings"
    "complex_simulation_settings"
    "complex_solvation_settings"
    "engine_settings"
    "forcefield_settings"
    "integrator_settings"
    "partial_charge_settings"
    "protocol_repeats"
    "solvent_equil_output_settings"
    "solvent_equil_simulation_settings"
    "solvent_lambda_settings"
    "solvent_output_settings"
    "solvent_restraint_settings"
    "solvent_simulation_settings"
    "solvent_solvation_settings"
    "thermo_settings"


.. code:: bash

    $ jq ".protocol.settings.protocol_repeats" rbfe_lig_1a_lig_03.json
    1
