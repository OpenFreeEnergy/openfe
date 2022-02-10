import openfe.setup


def test_radial_graph(lomap_basic_test_files):
    others = [v for (k, v) in lomap_basic_test_files.items()
              if k != 'toluene']
    toluene = lomap_basic_test_files['toluene']
    mapper = openfe.setup.LomapAtomMapper()

    network = openfe.setup.network_planning.generate_radial_graph(
        ligands=others, central_ligand=toluene,
        mappers=[mapper], scorer=None,
    )

    assert len(network.nodes) == len(others) + 1
    assert len(network.edges) == len(others)

    ligands_in_network = {mol.name for mol in network.nodes}
    assert ligands_in_network == set(lomap_basic_test_files.keys())
