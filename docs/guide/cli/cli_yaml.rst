Customising CLI planning with YAML settings
===========================================

The planning commands in the CLI can be made more powerful by supplying
YAML-formatted files to customise the planning algorithms.

This settings file has a series of sections for customising the different algorithms.
For example, the settings file which re-specifies the default behaviour would look like ::

  network:
    method: plan_minimal_spanning_tree
  mapper:
    method: LomapAtomMapper
    settings:
      time: 1
      threed: True
      max3d: 0.95
      element_change: True
  partial_charge:
    method: am1bcc
    settings:
      off_toolkit_backend: ambertools

The name of the algorithm is given behind the ``method:`` key and the arguments to the
algorithm are then optionally given behind the ``settings:`` key.
All sections of the file ``network:``, ``mapper:``  and ``partial_charge:`` are optional.

The settings YAML file is then provided to the ``-s`` option of ``openfe plan-rbfe-network``: ::

  openfe plan-rbfe-network -M molecules.sdf -P protein.pdb -s settings.yaml

Customising the atom mapper
---------------------------

There is a choice to be made as to which atom mapper is used,
currently included are the :class:`.LomapAtomMapper` and the :class:`.KartografAtomMapper` (full details in the `Kartograf documentation`_.)

.. _Kartograf documentation: https://kartograf.readthedocs.io/en/latest/api/kartograf.mappers.html#kartograf.atom_mapper.KartografAtomMapper

For example, to switch to using the ``Kartograf`` atom mapper, this settings YAML could be used ::

  mapper:
    method: KartografAtomMapper
    settings:
      atom_max_distance: 0.95
      atom_map_hydrogens: True
      map_hydrogens_on_hydrogens_only: False
      map_exact_ring_matches_only: True


Customising the network planner
-------------------------------

There are a variety of network planning options available, including
:func:`.generate_radial_network`,
:func:`.generate_minimal_spanning_network`, and
:func:`.generate_minimal_redundant_network`.

For example, to plan a radial network using a ligand called 'CHEMBL1078774' as the central ligand, this settings YAML
could be given ::

  network:
    method: generate_radial_network
    settings:
      central_ligand: CHEMBL1078774

Where the required ``central_ligand`` argument has been passed inside the ``settings:`` section.

Note that there is a subtle distinction when ligand names could be interpreted as integers.
To select the first ligand, the **integer** 0 can be given ::

  network:
    method: generate_radial_network
    settings:
      central_ligand: 0

Whereas if we wanted to specify the ligand named "0", we would instead explicitly pass this as **a string** to the YAML
settings file ::

  network:
    method: generate_radial_network
    settings:
      central_ligand: '0'

Customising the partial charge generation
-----------------------------------------

There are a range of partial charge generation schemes available, including

    - ``am1bcc``
    - ``am1bccelf10`` (only possible if ``off_toolkit_backend`` in settings is set to ``openeye``)
    - ``nagl`` (must have ``openff-nagl`` installed)
    - ``espaloma`` (must have ``espaloma_charge`` installed)

The following settings can also be set

    - ``off_toolkit_backend`` The backend to use for partial charge generation. Choose from  ``ambertools`` (default), ``openeye`` or ``rdkit``.
    - ``number_of_conformers`` The number of conformers to use for partial charge generation. If unset (default), the input conformer will be used.
    - ``nagl_model``: The NAGL model to use. If unset (default), the latest available production charge model will be used.

For example, to generate the partial charges using the ``am1bccelf10`` method from ``openeye`` the following should be added to the YAML settings file ::

 partial_charge:
   method: am1bccelf10
   settings:
     off_toolkit_backend: openeye

For more information on the different options, please refer to the :class:`.OpenFFPartialChargeSettings`.
