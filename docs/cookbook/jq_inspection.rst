.. _jq_inspection:

Using ``jq`` to inspect OpenFE JSONs
==============================================
Sometimes you may want to get a sense of the contents of JSON files, but the files are too unwieldy to inspect one-by-one in a code editor.

`jq <https://github.com/jqlang/jq>`_ is a helpful command-line tool that we recommend for for quickly inspecting JSON files.

Below is a common use-case to get you started, but you can do much more by checking out the `jq manual <https://jqlang.org/manual/>`_.

To view all the top-level JSON keys, use ``jq "keys" filename.json``, for example with a results JSON from the tutorial:

.. code:: bash

    $ jq "keys" rbfe_lig_ejm_46_solvent_lig_jmc_28_solvent.json
    [
    "estimate",
    "protocol_result",
    "uncertainty",
    "unit_results"
    ]

.. note::

    You can use ``"keys[]"`` instead of ``"keys"`` for a cleaner output.

Now that you know ``estimate`` is at the top-level of the JSON, you can use the following pattern to see the next level of keys:

.. code:: bash

    $ jq ".estimate | keys " rbfe_lig_ejm_46_solvent_lig_jmc_28_solvent.json
    {
    "magnitude",
    "unit":,
    ":is_custom:":,
    "pint_unit_registry":
    }


If you want to show all the keys _and_ their values, simply omit ``| key`` from the query:

.. code:: bash

    $ jq ".estimate" rbfe_lig_ejm_46_solvent_lig_jmc_28_solvent.json
    {
    "magnitude": 23.347074789078682,
    "unit": "kilocalorie / mole",
    ":is_custom:": true,
    "pint_unit_registry": "openff_units"
    }


This can be very helpful for quickly checking results for many files, for example:

.. code:: bash

    $ jq ".estimate.magnitude" rbfe*.json
    -14.925911852820793
    -40.72063957254803
    -27.76541486479537
    -16.023754604070007
    -57.38608716292447
    -15.748326155729705
    -39.933880531487326
    -27.780933075807425
    -16.76023951588401
    -58.36294851896545
    -19.038006312251575
    -20.26856586311034
    17.338257573349775
    15.775784163095102
    23.134622420900932
    17.071712542470248
    15.873122071409249
    23.347074789078682
