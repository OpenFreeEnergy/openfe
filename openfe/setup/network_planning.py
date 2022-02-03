import math


class Network:
    @classmethod
    def from_lomap(self, lomap_file):
        pass


def generate_radial_graph(ligands, central_ligand, mappers, extra_scorers=None):
    """Radial Network generator

    Parameters
    ----------
    ligands : list of rdkit Molecules
      the ligands to arrange around the central ligand
    central_ligand : rdkit Molecule
      the ligand to use as the hub/central ligand
    mappers : iterable of AtomMappers
      mappers to use, at least 1 required
    extra_scorers : iterable of Scorers, optional
      extra ways to assign scores

    Returns
    -------
    network : Network
      will have an edge between each ligand and the central ligand, with the mapping
      being the best possible mapping found using the supplied atom mappers.
    """
    n = Network()

    for l in ligands:
        best_score = inf
        best_mapping = None

        for mapper in mappers:
            for mapping in mapper.suggest_mappings(central_ligand, l):
                score = mapper.score_mapping(mapping)

                if score < best_score:
                    best_mapping = mapping
                    best_score = score

            if best_mapping is None:
                raise ValueError("No mapping found!")
            n.add_edge(best_mapping)

    raise NotImplementedError


def minimal_spanning_graph(ligands, mappers, extra_scorers=None):
    raise NotImplementedError
