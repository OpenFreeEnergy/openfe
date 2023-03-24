# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import abc
from typing import Iterable, Callable, Union, List, Type

from gufe import Protocol, AlchemicalNetwork, LigandAtomMapping, Transformation
from gufe import SmallMoleculeComponent, ProteinComponent, SolventComponent


from .abstract_alchemical_network_planner import AbstractAlchemicalNetworkPlanner

from .. import LomapAtomMapper
from ..ligand_network import LigandNetwork
from ..atom_mapping.ligandatommapper import LigandAtomMapper
from ..atom_mapping.lomap_scorers import default_lomap_score
from ..ligand_network_planning import generate_minimal_spanning_network
from ..chemicalsystem_generator.abstract_chemicalsystem_generator import (
    AbstractChemicalSystemGenerator,
)
from ..chemicalsystem_generator import EasyChemicalSystemGenerator, RFEComponentLabels
from ...protocols.openmm_rbfe.equil_rbfe_methods import RelativeLigandProtocol


# TODO: move/or find better structure for protocol_generator combintations!
PROTOCOL_GENERATOR = {
    RelativeLigandProtocol: EasyChemicalSystemGenerator,
}


"""
    easy relative campaigner
"""


class RelativeAlchemicalNetworkPlanner(AbstractAlchemicalNetworkPlanner, abc.ABC):

    """
    magics
    """
    _chemical_system_generator: AbstractChemicalSystemGenerator

    """
        public funcs
    """
    def __init__(
        self,
        name : str = "easy_rfe_calculation",
        mappers: Iterable[LigandAtomMapper] = [LomapAtomMapper()],
        mapping_scorer: Callable = default_lomap_score,
        ligand_network_planner: Callable = generate_minimal_spanning_network,
        protocol: Protocol = RelativeLigandProtocol(
            RelativeLigandProtocol._default_settings()
        ),
    ):
        """
        a simple strategy for executing a given protocol with mapper, mapping_scorers and networks for relative FE approaches.

        pattern: immutable
        Parameters
        ----------
        mapper
        mapping_scorer
        ligand_network_planner
        protocol
        """
        
        # TODO: Remove as soon as element Changes are possible. - START
        for mapper in mappers:
            mapper._no_element_changes = True
        # TODO: Remove as soon as element Changes are possible. - END
        
        self.name = name
        self._mappers = mappers
        self._mapping_scorer = mapping_scorer
        self._ligand_network_planner = ligand_network_planner
        self._protocol = protocol
        self._chemical_system_generator_type = PROTOCOL_GENERATOR[protocol.__class__]


    # TODO: Maybe abc.abstractmethod
    def __call__(self, *args, **kwargs) -> AlchemicalNetwork:
        return self._build_alchemical_network(*args, **kwargs)

    """
        Properties:
        TODO: Consider how to implement these
    """

    @property
    def mappers(self) -> Iterable[LigandAtomMapper]:
        return self._mappers

    @property
    def mapping_scorer(self) -> Callable:
        return self._mapping_scorer

    @property
    def ligand_network_planner(self) -> Callable:
        return self._ligand_network_planner
    @property
    def transformation_protocol(self) -> Protocol:
        return self._protocol

    @property
    def chemical_system_generator_type(self) -> Type[AbstractChemicalSystemGenerator]:
        return self._chemical_system_generator_type
    

    """
        private funcs
    """

    def _construct_ligand_network(
        self, ligands: Iterable[SmallMoleculeComponent]
    ) -> LigandNetwork:
        ligand_network = self._ligand_network_planner(
            ligands=ligands, mappers=self.mappers, scorer=self.mapping_scorer
        )

        return ligand_network


    def _build_transformations(
        self, ligand_network_edges, protocol, chemical_system_generator
    ) -> AlchemicalNetwork:
        edges = []
        nodes = []

        for edge in ligand_network_edges:

            for stateA_env, stateB_env in zip(
                self._chemical_system_generator(edge.componentA),
                self._chemical_system_generator(edge.componentB),
            ):
                transformation_name = (
                    self.name + "_" + stateA_env.name + "_" + stateB_env.name
                )

                #Todo: Another dirty hack!
                protocol_settings = self.transformation_protocol.settings

                if("vacuum" in transformation_name):
                    protocol_settings.system_settings.nonbonded_method = "nocutoff"

                protocol_settings.alchemical_settings.atom_overlap_tolerance = 100 # Todo: Hack to avoid protocol erros -  remove after fix was merged:  github PR #274
                
                self.protocol = self.transformation_protocol.__class__(settings=protocol_settings)

                edges.append(
                    Transformation(
                        stateA=stateA_env,
                        stateB=stateB_env,
                        mapping={RFEComponentLabels.LIGAND: edge}, #Todo: dirty hack!
                        name=transformation_name,
                        protocol=self.transformation_protocol,
                    )
                )
                nodes.extend([stateA_env, stateB_env])

        alchemical_network = AlchemicalNetwork(nodes=nodes, edges=edges, name=self.name)
        return alchemical_network

    @abc.abstractmethod
    def _build_chemicalsystem_generator(
        self, *args, **kwargs
    ) -> AbstractChemicalSystemGenerator:
        raise NotImplementedError()

    @abc.abstractmethod
    def _build_alchemical_network(self, *args, **kwargs) -> AlchemicalNetwork:
        raise NotImplementedError()


class RHFEAlchemicalNetworkPlanner(RelativeAlchemicalNetworkPlanner):
    
    def __init__(
    self,
    name : str = "easy_rhfe",
    mappers: Iterable[LigandAtomMapper] = [LomapAtomMapper()],
    mapping_scorer: Callable = default_lomap_score,
    ligand_network_planner: Callable = generate_minimal_spanning_network,
    protocol: Protocol = RelativeLigandProtocol(
            RelativeLigandProtocol._default_settings()
        ),
    ):
            super().__init__(name=name, mappers=mappers, mapping_scorer=mapping_scorer, ligand_network_planner=ligand_network_planner, protocol=protocol)
            
    def _build_chemicalsystem_generator(
        self, solvent
    ) -> AbstractChemicalSystemGenerator:
        return self._chemical_system_generator_type(solvent=solvent, do_vacuum=True)

    def _build_alchemical_network(
        self,
        ligands: Iterable[SmallMoleculeComponent],
        solvent: SolventComponent = None,
    ) -> AlchemicalNetwork:
        # components might be given differently!
        # throw into ligand_network_planning
        self._ligand_network = self._construct_ligand_network(ligands)

        # Prepare system generation
        self._chemical_system_generator = self._build_chemicalsystem_generator(solvent)

        # Build transformations
        self._alchemical_network = self._build_transformations(
            ligand_network_edges=self._ligand_network.edges,
            protocol=self.transformation_protocol,
            chemical_system_generator=self.chemical_system_generator_type,
        )

        return self._alchemical_network


class RBFEAlchemicalNetworkPlanner(RelativeAlchemicalNetworkPlanner):
    
    def __init__(
        self,
        name : str = "easy_rbfe",
        mappers: Iterable[LigandAtomMapper] = [LomapAtomMapper()],
        mapping_scorer: Callable = default_lomap_score,
        ligand_network_planner: Callable = generate_minimal_spanning_network,
        protocol: Protocol = RelativeLigandProtocol(
                RelativeLigandProtocol._default_settings()
            ),
    ):
            super().__init__(name=name, mappers=mappers, mapping_scorer=mapping_scorer, ligand_network_planner=ligand_network_planner, protocol=protocol)
            
            
    def _build_chemicalsystem_generator(
        self, solvent: SolventComponent, protein: ProteinComponent
    ) -> AbstractChemicalSystemGenerator:
        chemical_system_generator = self._chemical_system_generator_type(
            solvent=solvent, protein=protein
        )
        return chemical_system_generator

    def _build_alchemical_network(
        self,
        ligands: Iterable[SmallMoleculeComponent],
        solvent: SolventComponent = None,
        protein: ProteinComponent = None,
    ) -> AlchemicalNetwork:
        # components might be given differently!
        # throw into ligand_network_planning
        self._ligand_network = self._construct_ligand_network(ligands)

        # Prepare system generation
        self._chemical_system_generator = self._build_chemicalsystem_generator(
            solvent=solvent, protein=protein
        )

        # Build transformations
        self._alchemical_network = self._build_transformations(
            ligand_network_edges=self._ligand_network.edges,
            protocol=self._protocol,
            chemical_system_generator=self.chemical_system_generator_type,
        )

        return self._alchemical_network
