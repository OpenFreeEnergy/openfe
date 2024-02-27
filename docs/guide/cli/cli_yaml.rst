Customising planning with yaml
==============================

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

The name of the algorithm is given behind the ``method:`` key and the arguments to the
algorithm are then optionally given behind the ``settings:`` key.
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
      max3d: 0.95  # todo; lookup actual options


Customising the network planner
-------------------------------

There are a variety of network planning options available, including
``plan_radial_network``,
``plan_minimal_spanning_tree``, and
``plan_minimal_redundant_network``.

For example to plan a radial network using a ligand called 'CHEMBL102451' as the central ligand, this settings yaml could be given ::



Note that there is a subtle distinction when ligand names could be interpretted as integers.
To select the first ligand, the **integer** 0 can be given ::

  network:
    method: plan_radial_network
    settings:
      central_ligand: 0

Whereas if we wanted to specify the ligand named "0", we would instead explicitly pass this as a string to the yaml settings file ::

  network:
    method: plan_radial_network
    settings:
      central_ligand: '0'

