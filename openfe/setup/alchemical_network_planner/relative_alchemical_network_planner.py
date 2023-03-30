# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import abc
import copy
from typing import Iterable, Callable, Type

from gufe import (
    Protocol,
    AlchemicalNetwork,
    LigandAtomMapping,
    Transformation,
    ChemicalSystem,
)
from gufe import SmallMoleculeComponent, ProteinComponent, SolventComponent


from .abstract_alchemical_network_planner import (
    AbstractAlchemicalNetworkPlanner,
)

from .. import LomapAtomMapper
from ..ligand_network import LigandNetwork
from ..atom_mapping.ligandatommapper import LigandAtomMapper
from ..atom_mapping.lomap_scorers import default_lomap_score
from ..ligand_network_planning import generate_minimal_spanning_network
from ..chemicalsystem_generator.abstract_chemicalsystem_generator import (
    AbstractChemicalSystemGenerator,
)
from ..chemicalsystem_generator import (
    EasyChemicalSystemGenerator,
    RFEComponentLabels,
)
from ...protocols.openmm_rbfe.equil_rbfe_methods import RelativeLigandProtocol


# TODO: move/or find better structure for protocol_generator combintations!
PROTOCOL_GENERATOR = {
    RelativeLigandProtocol: EasyChemicalSystemGenerator,
}


class RelativeAlchemicalNetworkPlanner(
    AbstractAlchemicalNetworkPlanner, abc.ABC
):

    """
    magics
    """

    _chemical_system_generator: AbstractChemicalSystemGenerator

    """
        public funcs
    """

    def __init__(
        self,
        name: str = "easy_rfe_calculation",
        mappers: Iterable[LigandAtomMapper] = [LomapAtomMapper()],
        mapping_scorer: Callable = default_lomap_score,
        ligand_network_planner: Callable = generate_minimal_spanning_network,
        protocol: Protocol = RelativeLigandProtocol(
            RelativeLigandProtocol.default_settings()
        ),
    ):
        """a simple strategy for executing a given protocol with mapper, mapping_scorers and networks for relative FE approaches.

        Parameters
        ----------
        name : str, optional
            name of the approach/project the rfe, by default "easy_rfe_calculation"
        mappers : Iterable[LigandAtomMapper], optional
            mappers used to connect the ligands, by default [LomapAtomMapper()]
        mapping_scorer : Callable, optional
            scorer evaluating the quality of the atom mappings, by default default_lomap_score
        ligand_network_planner : Callable, optional
            network using mapper and smapping_scorer to build up an optimal network, by default generate_minimal_spanning_network
        protocol : Protocol, optional
            FE-protocol for each transformation (edge of ligand networ) that is required in order to calculate the FE graph, by default RelativeLigandProtocol( RelativeLigandProtocol._default_settings() )
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
        self._chemical_system_generator_type = PROTOCOL_GENERATOR[
            protocol.__class__
        ]

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> AlchemicalNetwork:
        return self._build_alchemical_network(*args, **kwargs)

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
    def chemical_system_generator_type(
        self,
    ) -> Type[AbstractChemicalSystemGenerator]:
        return self._chemical_system_generator_type

    def _construct_ligand_network(
        self, ligands: Iterable[SmallMoleculeComponent]
    ) -> LigandNetwork:
        ligand_network = self._ligand_network_planner(
            ligands=ligands, mappers=self.mappers, scorer=self.mapping_scorer
        )

        return ligand_network

    def _build_transformations(
        self,
        ligand_network_edges: Iterable[LigandAtomMapping],
        protocol: Protocol,
        chemical_system_generator: AbstractChemicalSystemGenerator,
    ) -> AlchemicalNetwork:
        """construct alchemical network by building transformations from ligand network and adding the given protocol to each transformation.

        Parameters
        ----------
        ligand_network_edges : Iterable[LigandAtomMapping]
            result from the ligand network planner connecting all Ligands, planning the transformations.
        protocol : Protocol
            fe simulation protocol for each transformation.
        chemical_system_generator : AbstractChemicalSystemGenerator
            generator, constructing all required chemical systems for each transformation.

        Returns
        -------
        AlchemicalNetwork
            knows all transformations and their states that need to be simulated.
        """
        transformation_edges = []
        end_state_nodes = []

        for ligand_mapping_edge in ligand_network_edges:
            for stateA_env, stateB_env in zip(
                chemical_system_generator(ligand_mapping_edge.componentA),
                chemical_system_generator(ligand_mapping_edge.componentB),
            ):
                transformation_edge = self._build_transformation(
                    ligand_mapping_edge=ligand_mapping_edge,
                    stateA=stateA_env,
                    stateB=stateB_env,
                    transformation_protocol=protocol,
                )

                transformation_edges.append(transformation_edge)
                end_state_nodes.extend([stateA_env, stateB_env])

        # Todo: make the code here more stable in future: Name doubling check
        all_transformation_labels = list(
            map(lambda x: x.name, transformation_edges)
        )

        if len(all_transformation_labels) != len(
            set(all_transformation_labels)
        ):
            raise ValueError(
                "There where multiple transformations with the same edge label! This might lead to overwritting your files. \n labels: "
                + str(len(all_transformation_labels))
                + "\nunique: "
                + str(len(set(all_transformation_labels)))
                + "\ngot: \n\t"
                + "\n\t".join(all_transformation_labels)
            )

        alchemical_network = AlchemicalNetwork(
            nodes=end_state_nodes, edges=transformation_edges, name=self.name
        )
        return alchemical_network

    def _build_transformation(
        self,
        ligand_mapping_edge: LigandAtomMapping,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        transformation_protocol: Protocol,
    ) -> Transformation:
        """
            This function is the core of building transformations. it builds a transformation with the given protocol.

        Parameters
        ----------
        ligand_mapping_edge: LigandAtomMapping
        stateA: ChemicalSystem
        stateB: ChemicalSystem

        Returns
        -------
        Transformation

        """
        transformation_name = self.name + "_" + stateA.name + "_" + stateB.name

        # Todo: Another dirty hack! - START
        protocol_settings = copy.deepcopy(transformation_protocol.settings)
        if "vacuum" in transformation_name:
            protocol_settings.system_settings.nonbonded_method = "nocutoff"

        protocol_settings.alchemical_settings.atom_overlap_tolerance = 100  # Todo: Hack to avoid protocol errors -  remove after fix was merged:
        # github PR #274
        # Todo: Another dirty hack! - END

        transformation_protocol = transformation_protocol.__class__(
            settings=protocol_settings
        )

        return Transformation(
            stateA=stateA,
            stateB=stateB,
            mapping={RFEComponentLabels.LIGAND: ligand_mapping_edge},
            name=transformation_name,
            protocol=transformation_protocol,
        )

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
        name: str = "easy_rhfe",
        mappers: Iterable[LigandAtomMapper] = [LomapAtomMapper()],
        mapping_scorer: Callable = default_lomap_score,
        ligand_network_planner: Callable = generate_minimal_spanning_network,
        protocol: Protocol = RelativeLigandProtocol(
            RelativeLigandProtocol._default_settings()
        ),
    ):
        super().__init__(
            name=name,
            mappers=mappers,
            mapping_scorer=mapping_scorer,
            ligand_network_planner=ligand_network_planner,
            protocol=protocol,
        )

    def __call__(
        self,
        ligands: Iterable[SmallMoleculeComponent],
        solvent: SolventComponent,
    ) -> AlchemicalNetwork:
        """plan the alchemical network for the given ligands and solvent.

        Parameters
        ----------
        ligands : Iterable[SmallMoleculeComponent]
            ligands that shall be used for the alchemical network.
        solvent : SolventComponent
            solvent for solvated simulations

        Returns
        -------
        AlchemicalNetwork
            RHFE network for the given ligands and solvent.
        """
        return self._build_alchemical_network(ligands=ligands, solvent=solvent)

    def _build_chemicalsystem_generator(
        self, solvent
    ) -> AbstractChemicalSystemGenerator:
        return self._chemical_system_generator_type(
            solvent=solvent, do_vacuum=True
        )

    def _build_alchemical_network(
        self,
        ligands: Iterable[SmallMoleculeComponent],
        solvent: SolventComponent,
    ) -> AlchemicalNetwork:
        """plan the alchemical network for the given ligands and solvent.

        Parameters
        ----------
        ligands : Iterable[SmallMoleculeComponent]
            ligands that shall be used for the alchemical network.
        solvent : SolventComponent
            solvent for solvated simulations

        Returns
        -------
        AlchemicalNetwork
            RHFE network for the given ligands and solvent.
        """
        # components might be given differently!
        # throw into ligand_network_planning
        self._ligand_network = self._construct_ligand_network(ligands)

        # Prepare system generation
        self._chemical_system_generator = self._build_chemicalsystem_generator(
            solvent=solvent
        )

        # Build transformations
        self._alchemical_network = self._build_transformations(
            ligand_network_edges=self._ligand_network.edges,
            protocol=self.transformation_protocol,
            chemical_system_generator=self._chemical_system_generator,
        )

        return self._alchemical_network


class RBFEAlchemicalNetworkPlanner(RelativeAlchemicalNetworkPlanner):
    def __init__(
        self,
        name: str = "easy_rbfe",
        mappers: Iterable[LigandAtomMapper] = [LomapAtomMapper()],
        mapping_scorer: Callable = default_lomap_score,
        ligand_network_planner: Callable = generate_minimal_spanning_network,
        protocol: Protocol = RelativeLigandProtocol(
            RelativeLigandProtocol._default_settings()
        ),
    ):
        super().__init__(
            name=name,
            mappers=mappers,
            mapping_scorer=mapping_scorer,
            ligand_network_planner=ligand_network_planner,
            protocol=protocol,
        )

    def __call__(
        self,
        ligands: Iterable[SmallMoleculeComponent],
        solvent: SolventComponent,
        protein: ProteinComponent,
    ) -> AlchemicalNetwork:
        """plan the alchemical network for RBFE calculations with the given ligands, protein and solvent.

        Parameters
        ----------
        ligands : Iterable[SmallMoleculeComponent]
            ligands that shall be used for the alchemical network.
        solvent : SolventComponent
            solvent for solvated and complex simulations
        protein : ProteinComponent
            protein for complex simulations

        Returns
        -------
        AlchemicalNetwork
            RBFE network for the given ligands, protein and solvent.
        """
        return self._build_alchemical_network(
            ligands=ligands, solvent=solvent, protein=protein
        )

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
        solvent: SolventComponent,
        protein: ProteinComponent,
    ) -> AlchemicalNetwork:
        """plan the alchemical network for RBFE calculations with the given ligands, protein and solvent.

        Parameters
        ----------
        ligands : Iterable[SmallMoleculeComponent]
            ligands that shall be used for the alchemical network.
        solvent : SolventComponent
            solvent for solvated and complex simulations
        protein : ProteinComponent
            protein for complex simulations

        Returns
        -------
        AlchemicalNetwork
            RBFE network for the given ligands, protein and solvent.
        """
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
            chemical_system_generator=self._chemical_system_generator,
        )

        return self._alchemical_network
