# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Iterable, Callable

from gufe import Protocol, AlchemicalNetwork, LigandAtomMapping
from gufe import SmallMoleculeComponent, ProteinComponent, SolventComponent


from gufe.mapping.atom_mapper import AtomMapper

from .abstract_alchemical_network_planner import AbstractAlchemicalNetworkPlanner

from .. import LomapAtomMapper
from ..ligand_network import LigandNetwork
from ..atom_mapping.lomap_scorers import default_lomap_score
from ..ligand_network_planning import generate_minimal_spanning_network
from ..transformation_factory import RFETransformationFactory
from ..chemicalsystem_generator import EasyChemicalSystemGenerator
from ...protocols.openmm_rbfe.equil_rbfe_methods import RelativeLigandProtocol

"""
    This is a draft!
"""
# TODO: move/or find better structure for protocol_generator combintations!
protocol_generator = {
    RelativeLigandProtocol: EasyChemicalSystemGenerator,
}


"""
    easy relative campaigner
"""


class RelativeAlchemicalNetworkPlanner(AbstractAlchemicalNetworkPlanner):
    mappings: LigandAtomMapping
    ligand_network: LigandNetwork
    alchemical_network: AlchemicalNetwork

    def __init__(
        self,
        mapper: AtomMapper = LomapAtomMapper(),
        mapping_scorers: Callable = default_lomap_score,
        networker: Callable = generate_minimal_spanning_network,
        protocol: Protocol = RelativeLigandProtocol(
            RelativeLigandProtocol._default_settings()
        ),
    ):
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


class RHFEAlchemicalNetworkPlanner(RelativeAlchemicalNetworkPlanner):
    def __init__(
        self,
        mapper: AtomMapper = LomapAtomMapper(),
        mapping_scorers: Callable = default_lomap_score,
        networker: Callable = generate_minimal_spanning_network,
    ):
        super().__init__(
            mapper=mapper,
            mapping_scorers=mapping_scorers,
            networker=networker,
            protocol=RelativeLigandProtocol(RelativeLigandProtocol._default_settings()), #Todo: this default is currently a bit hacky
        )

    def __call__(
        self,
        ligands: Iterable[SmallMoleculeComponent],
        solvent: SolventComponent = None,
    ) -> AlchemicalNetwork:
        # components might be given differently!

        # throw into Networker
        # TODO: decompose networker here into the nice helpers above
        self.ligand_network = self.networker(
            ligands=ligands, mappers=[self.mapper], scorer=self.mapping_scorers
        )

        # Prepare system generation
        self.chemical_system_generator = self.generator_type(
            solvent=solvent, do_vacuum=True
        )

        # build transformations
        self.transformer = RFETransformationFactory(
            protocol=self.protocol,
            chemical_system_generator=self.chemical_system_generator,
        )
        self.alchemical_network = self.transformer(
            alchemical_network_edges=self.ligand_network.edges
        )

        return self.alchemical_network


class RBFEAlchemicalNetworkPlanner(RelativeAlchemicalNetworkPlanner):
    def __init__(
        self,
        mapper: AtomMapper = LomapAtomMapper(),
        mapping_scorers: Callable = default_lomap_score,
        networker: Callable = generate_minimal_spanning_network,
    ):
        super().__init__(
            mapper=mapper,
            mapping_scorers=mapping_scorers,
            networker=networker,
            protocol=RelativeLigandProtocol(RelativeLigandProtocol._default_settings()),
        )

    def __call__(
        self,
        ligands: Iterable[SmallMoleculeComponent],
        solvent: SolventComponent = None,
        receptor: ProteinComponent = None,
    ):
        # components might be given differently!

        # throw into Networker
        self.ligand_network = self.networker(
            ligands=ligands, mappers=[self.mapper], scorer=self.mapping_scorers
        )

        # Prepare system generation
        self.chemical_system_generator = self.generator_type(
            solvent=solvent, protein=receptor
        )

        # build transformations
        self.transformer = RFETransformationFactory(
            protocol=self.protocol,
            chemical_system_generator=self.chemical_system_generator,
        )
        self.alchemical_network = self.transformer(
            alchemical_network_edges=self.ligand_network.edges
        )

        return self.alchemical_network
