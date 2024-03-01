Customising CLI planning with yaml settings
===========================================

The planning commands in the CLI can be made more powerful by supplying
"yaml" formatted files to customise the planning algorithms.
This settings file has a series of sections for customising the different algorithms,
as an example, the settings file which re-specifies the default behaviour would look like ::

  network:
    method: plan_minimal_spanning_tree
  mapper:
    method: LomapAtomMapper
    settings:
      time: 1
      threed: True
      max3d: 0.95
      element_change: True

The name of the algorithm is given behind the ``method:`` key and the arguments to the
algorithm are then optionally given behind the ``settings:`` key.
Both the `network:` and `mapper:` sections are optional.

This is then provided to the ``openfe plan-rbfe-network`` command as ::

  openfe plan-rbfe-network -M molecules.sdf -P protein.pdb -s settings.yaml

Customising the atom mapper
---------------------------

There is a choice to be made as to which atom mapper is used,
currently included are the :class:`.LomapAtomMapper` and the :class:`.KartografAtomMapper`
For example to switch to using the ``Kartograf`` atom mapper, this settings yaml could be used ::

  mapper:
    method: KartografAtomMapper
    settings:
      atom_max_distance: 0.95
      atom_map_hydrogens: True
      map_hydrogens_on_hydrogens_only: False
      map_exact_ring_matches_only: True

Full details on these options can be found in the `Kartograf documentation`_.

.. _Kartograf documentation: https://kartograf.readthedocs.io/en/latest/api/kartograf.mappers.html#kartograf.atom_mapper.KartografAtomMapper

Customising the network planner
-------------------------------

There are a variety of network planning options available, including
:func:`.generate_radial_network`,
:func:`.generate_minimal_spanning_network`, and
:func:`.generate_minimal_redundant_network`.

For example to plan a radial network using a ligand called 'CHEMBL1078774' as the central ligand, this settings yaml
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

Whereas if we wanted to specify the ligand named "0", we would instead explicitly pass this as **a string** to the yaml
settings file ::

  network:
    method: generate_radial_network
    settings:
      central_ligand: '0'
