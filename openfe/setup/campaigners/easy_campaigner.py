# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Iterable, Callable

from rdkit import Chem

from gufe import Protocol, AlchemicalNetwork
from gufe.mapping.atom_mapper import AtomMapper

from ._abstract_campaigner import _abstract_campaigner
from .. import LomapAtomMapper, Transformation
from ..atom_mapping.lomap_scorers import default_lomap_score
from ..ligand_network_planning import minimal_spanning_graph
from ...protocols.openmm_rbfe.equil_rbfe_methods import RelativeLigandTransform

from .building_blocks import load_files, build_transformations_from_edges

"""
    This is a draft!
"""


class easy_campaigner(_abstract_campaigner):

    def __init__(self,
                 mapper: AtomMapper = LomapAtomMapper,
                 mapping_scorers: Iterable[Callable] = [default_lomap_score],
                 networker: Callable = minimal_spanning_graph,
                 protocol:Protocol =RelativeLigandTransform) -> AlchemicalNetwork:
        """
        a simple strategy for executing a given protocol with mapper, mapping_scorers and networks for relative FE approaches.

        Parameters
        ----------
        mapper
        mapping_scorers
        networker
        protocol
        """
        self.mapper = mapper
        self.mapping_scorers = mapping_scorers
        self.networker = networker
        self.protocol = protocol

    def __call__(self, small_components, receptor_component):
        # components might be given differently!

        #throw into Networker
        self.network = self.networker(ligands=small_components,
                            mappers=[self.mapper],
                            mapping_scorers=self.mapping_scorers)


        #build transformations
        alchemical_network = build_transformations_from_edges(paths=self.network.edges, protein_component=receptor_component,
                                                           protocol=self.protocol)

        return alchemical_network

class rbfe_campainger(easy_campaigner):

    def __init__(self, mapper:AtomMapper=LomapAtomMapper,
                       mapping_scorers:Iterable[Callable]=[default_lomap_score],
                       networker:Callable=minimal_spanning_graph):

        super().__init__(mapper=mapper, mapping_scorers=mapping_scorers,
                         networker=networker, protocol=RelativeLigandTransform)


