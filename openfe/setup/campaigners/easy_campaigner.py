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
from ..transformers.easy_transformer import rfe_transformer, afe_transformer
from ...protocols.openmm_rbfe.equil_rbfe_methods import RelativeLigandTransform
from ..chem_sys_generators.easy_system_generator import chem_system_generator

"""
    This is a draft!
"""
future_abfe_protocol = "future" # TODO: replace with real protocol
protocol_generator = {
    RelativeLigandTransform: chem_system_generator,
    future_abfe_protocol:chem_system_generator
}


"""
    easy relative campaigner
"""
class relative_campaigner(_abstract_campaigner):
    mappings: LigandAtomMapping
    ligand_network: LigandNetwork
    alchemical_network: AlchemicalNetwork

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


    #make clases below use these functions!
    def _generate_all_mappings(self):
        raise NotImplementedError()

    def _score_all_mappings(self):
        raise NotImplementedError()

    def _plan_network(self):
        raise NotImplementedError()

    def _generate_alchemical_network(self):
        raise NotImplementedError()

class rhfe_campaigner(relative_campaigner):
    def __init__(self, mapper: AtomMapper = LomapAtomMapper,
                 mapping_scorers: Iterable[Callable] = [default_lomap_score],
                 networker: Callable = minimal_spanning_graph):
        super().__init__(mapper=mapper, mapping_scorers=mapping_scorers,
                         networker=networker, protocol=RelativeLigandTransform)

    def __call__(self,  ligands:Iterable[SmallMoleculeComponent], solvent: SolventComponent = None,)->AlchemicalNetwork:
        # components might be given differently!

        # throw into Networker
        # TODO: decompose networker here into the nice helpers above
        self.ligand_network = self.networker(ligands=ligands,
                                             mappers=[self.mapper],
                                             mapping_scorers=self.mapping_scorers)

        # Prepare system generation
        self.chemical_system_generator = self.generator_type(solvent=solvent,
                                                             do_vacuum=True)

        # build transformations
        self.transformer = rfe_transformer(protocol=self.protocol,
                                           system_generator=self.chemical_system_generator)
        self.alchemical_network = self.transformer(alchemical_network_edges=self.ligand_network.edges)

        return self.alchemical_network

class rbfe_campaigner(relative_campaigner):

    def __init__(self, mapper: AtomMapper = LomapAtomMapper,
                 mapping_scorers: Iterable[Callable] = [default_lomap_score],
                 networker: Callable = minimal_spanning_graph):
        super().__init__(mapper=mapper, mapping_scorers=mapping_scorers,
                         networker=networker, protocol=RelativeLigandTransform)

    def __call__(self, ligands:Iterable[SmallMoleculeComponent], solvent: SolventComponent = None, receptor: ProteinComponent = None,):
        # components might be given differently!

        # throw into Networker
        # TODO: decompose networker here into the nice helpers above
        self.ligand_network = self.networker(ligands=ligands,
                                             mappers=[self.mapper],
                                             mapping_scorers=self.mapping_scorers)

        # Prepare system generation
        self.chemical_system_generator = self.generator_type(solvent=solvent,
                                                             protein=receptor)

        # build transformations
        self.transformer = rfe_transformer(protocol=self.protocol,
                                           system_generator=self.chemical_system_generator)
        self.alchemical_network = self.transformer(alchemical_network_edges=self.ligand_network.edges)

        return self.alchemical_network


"""
    easy absolute campaigner
"""
class absolute_campaigner(_abstract_campaigner):
    alchemical_network: AlchemicalNetwork

    def __init__(self,
                 protocol: Protocol = None) -> AlchemicalNetwork:
        """
        a simple strategy for executing a given protocol with mapper, mapping_scorers and networks for relative FE approaches.

        Parameters
        ----------
        mapper
        mapping_scorers
        networker
        protocol
        """
        self.protocol = protocol
        self.generator_type = protocol_generator[protocol.__class__]



class ahfe_campaigner(absolute_campaigner):
    def __call__(self,  ligands:Iterable[SmallMoleculeComponent], solvent: SolventComponent = None,)->AlchemicalNetwork:
        # components might be given differently!
        # Prepare system generation
        self.chemical_system_generator = self.generator_type(solvent=solvent,
                                                             do_vacuum=True)

        # build transformations
        self.transformer = afe_transformer(protocol=self.protocol,
                                           system_generator=self.chemical_system_generator)
        self.alchemical_network = self.transformer(alchemical_network_edges=ligands)

        return self.alchemical_network

class abfe_campaigner(absolute_campaigner):
    def __call__(self,  ligands:Iterable[SmallMoleculeComponent], solvent: SolventComponent = None,  receptor: ProteinComponent = None)->AlchemicalNetwork:
        # components might be given differently!
        # Prepare system generation
        self.chemical_system_generator = self.generator_type(solvent=solvent,
                                                             protein=receptor)

        # build transformations
        self.transformer = afe_transformer(protocol=self.protocol,
                                           system_generator=self.chemical_system_generator)
        self.alchemical_network = self.transformer(alchemical_network_edges=ligands)

        return self.alchemical_network