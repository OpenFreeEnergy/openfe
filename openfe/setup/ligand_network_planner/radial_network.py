# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import math
from typing import Iterable, Callable, Union
import itertools

from .abstract_network_planner import AbstractRelativeLigandNetworkPlanner

from gufe import SmallMoleculeComponent
from openfe.setup import LigandNetwork
from openfe.setup.atom_mapping import LigandAtomMapper


class RadialNetworkPlanner(AbstractRelativeLigandNetworkPlanner):
    def __init__(self, mappers: Iterable[LigandAtomMapper], mapping_scorer:Callable):
        """

        Parameters
        ----------
        mappers : iterable of LigandAtomMappers
          mappers to use, at least 1 required
        mapping_scorer : Union[None, Callable], optional
          a callable which returns a float for any LigandAtomMapping.  Used to
          assign scores to potential mappings, higher scores indicate worse
          mappings.
        """
        self._mappers = mappers
        self._mapping_scorer = mapping_scorer

    def __call__(
        self,
        ligands: Iterable[SmallMoleculeComponent],
        central_ligand: SmallMoleculeComponent,
    ) -> LigandNetwork:
        """Generate a radial network with all ligands connected to a central node

        Also known as hub and spoke or star-map, this plans a Network where
        all ligands are connected via a central ligand.

        Parameters
        ----------
        ligands : iterable of SmallMoleculeComponents
          the ligands to arrange around the central ligand
        central_ligand : SmallMoleculeComponent
          the ligand to use as the hub/central ligand

        Raises
        ------
        ValueError
          if no mapping between the central ligand and any other ligand can be
          found

        Returns
        -------
        network : LigandNetwork
          will have an edge between each ligand and the central ligand, with the
          mapping being the best possible mapping found using the supplied atom
          mappers.
          If no scorer is supplied, the first mapping provided by the iterable
          of mappers will be used.
        """

        edges = []
        self._mappings = []
        for ligand in ligands:
            best_score = math.inf
            best_mapping = None

            for mapping in itertools.chain.from_iterable(
                mapper.suggest_mappings(central_ligand, ligand)
                for mapper in self._mappers
            ):
                self._mappings.append(mapping)
                if self._mapping_scorer is None:
                    best_mapping = mapping
                    break

                score = self._mapping_scorer(mapping)
                mapping = mapping.with_annotations({"score": score})

                if score < best_score:
                    best_mapping = mapping
                    best_score = score

            if best_mapping is None:
                raise ValueError(f"No mapping found for {ligand}")
            edges.append(best_mapping)

        return LigandNetwork(edges)
