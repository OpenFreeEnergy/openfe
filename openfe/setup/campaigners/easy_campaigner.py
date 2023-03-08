# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Iterable, Callable

from gufe import Protocol, AlchemicalNetwork, LigandAtomMapping
from gufe import SmallMoleculeComponent, ProteinComponent, SolventComponent

from openfe.setup import Network as LigandNetwork

from gufe.mapping.atom_mapper import AtomMapper

from ._abstract_campaigner import _abstract_campaigner

from .. import LomapAtomMapper
from ..atom_mapping.lomap_scorers import default_lomap_score
from ..ligand_network_planning import minimal_spanning_graph
from ..transformers.simple_rbfe_transformer import simple_rbfe_transformer
from ...protocols.openmm_rbfe.equil_rbfe_methods import RelativeLigandTransform
from ..chem_sys_generators.rbfe_system_generators import rbfe_system_generator

"""
    This is a draft!
"""

protocol_generator = {
    RelativeLigandTransform: rbfe_system_generator
}

}

class relative_campaigner(_abstract_campaigner):

    def __init__(self,
                 mapper: AtomMapper = LomapAtomMapper(),
                 mapping_scorers: Iterable[Callable] = [default_lomap_score],
                 networker: Callable = minimal_spanning_graph,
                 protocol: Protocol = RelativeLigandTransform()) -> AlchemicalNetwork:
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
        self.generator_type = protocol_generator[protocol.__class__]


class easy_rbfe_campainger(relative_campaigner):
    mappings: LigandAtomMapping
    ligand_network: LigandNetwork
    alchemical_network: AlchemicalNetwork

    def __init__(self, mapper: AtomMapper = LomapAtomMapper,
                 mapping_scorers: Iterable[Callable] = [default_lomap_score],
                 networker: Callable = minimal_spanning_graph):
        super().__init__(mapper=mapper, mapping_scorers=mapping_scorers,
                         networker=networker, protocol=RelativeLigandTransform)

    def __call__(self, ligands:SmallMoleculeComponent , solvent: SolventComponent = None, receptor: ProteinComponent = None,):
        # components might be given differently!

        # throw into Networker
        # TODO: decompose networker here into the nice helpers below
        self.ligand_network = self.networker(ligands=ligands,
                                             mappers=[self.mapper],
                                             mapping_scorers=self.mapping_scorers)

        # Prepare system generation
        self.chemical_system_generator = self.generator_type(solvent=solvent,
                                                             protein=receptor)

        # build transformations
        self.transformer = simple_rbfe_transformer(protocol=self.protocol,
                                                   system_generator=self.chemical_system_generator)
        self.alchemical_network = self.transformer(alchemical_network_edges=self.ligand_network.edges)

        return self.alchemical_network

    def _generate_all_mappings(self):
        raise NotImplementedError()

    def _score_all_mappings(self):
        raise NotImplementedError()

    def _plan_network(self):
        raise NotImplementedError()

    def _generate_alchemical_network(self):
        raise NotImplementedError()
