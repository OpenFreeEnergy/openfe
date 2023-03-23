# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import abc
from typing import Iterable, Callable, Union, Optional, List

from gufe import Protocol, AlchemicalNetwork, LigandAtomMapping
from gufe import SmallMoleculeComponent, ProteinComponent, SolventComponent


from gufe.mapping.atom_mapper import AtomMapper

from .abstract_alchemical_network_planner import AbstractAlchemicalNetworkPlanner

from .. import LomapAtomMapper
from ..ligand_network import LigandNetwork
from ..atom_mapping.ligandatommapper import LigandAtomMapper
from ..atom_mapping.lomap_scorers import default_lomap_score
from ..ligand_network_planning import generate_minimal_spanning_network
from ..transformation_factory import RFETransformationFactory
from ..chemicalsystem_generator.abstract_chemicalsystem_generator import (
    AbstractChemicalSystemGenerator,
)
from ..chemicalsystem_generator import EasyChemicalSystemGenerator
from ...protocols.openmm_rbfe.equil_rbfe_methods import RelativeLigandProtocol

"""
    This is a draft!
"""
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

    def __init__(
        self,
        mapper: AtomMapper = LomapAtomMapper(),
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
        mapper._no_element_changes = True # TODO: Remove as soon as possible!

        self._mapper = mapper
        self._mapping_scorer = mapping_scorer
        self._ligand_network_planner = ligand_network_planner
        self._protocol = protocol
        self._generator_type = PROTOCOL_GENERATOR[protocol.__class__]

        # Result props:
        self._mappings : Union[None, List[LigandAtomMapping]]= None
        self._ligand_network : Union[None, LigandNetwork]= None
        self._chemical_system_generator : Union[None, AbstractChemicalSystemGenerator] = None
        self._alchemical_network : Union[None, AlchemicalNetwork] = None

    # TODO: Maybe abc.abstractmethod
    def __call__(self, *args, **kwargs) -> AlchemicalNetwork:
        return self._build_alchemical_network(*args, **kwargs)

    """
        Properties:
        TODO: Consider how to implement these
    """

    @property
    def mapper(self) -> AtomMapper:
        return self._mapper

    @property
    def mapping_scorer(self) -> Callable:
        return self._mapping_scorer

    @property
    def ligand_network_planner(self) -> Callable:
        return self._ligand_network_planner

    @property
    def mappings(self) -> Union[None, List[LigandAtomMapping]]:
        return self._mappings  # TODO: not doable at the moment! Need it!

    @property
    def ligand_network(self) -> Union[None, LigandNetwork]:
        return self._ligand_network

    @property
    def alchemical_network(self) -> Union[None, AlchemicalNetwork]:
        return self._alchemical_network

    @property
    def chemical_system_generator(self) -> Union[None, AbstractChemicalSystemGenerator]:
        return self._chemical_system_generator

    """
        private funcs
    """

    def _construct_ligand_network(
        self, ligands: Iterable[SmallMoleculeComponent]
    ) -> LigandNetwork:
        ligand_network = self._ligand_network_planner(
            ligands=ligands, mappers=[self.mapper], scorer=self.mapping_scorer
        )

        return ligand_network

    def _build_transformations(
        self, ligand_network_edges, protocol, chemical_system_generator
    ) -> AlchemicalNetwork:
        self.transformer = RFETransformationFactory(
            protocol=protocol,
            chemical_system_generator=chemical_system_generator,
        )
        alchemical_network = self.transformer(
            alchemical_network_edges=ligand_network_edges
        )
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
    def _build_chemicalsystem_generator(
        self, solvent
    ) -> AbstractChemicalSystemGenerator:
        return self._generator_type(solvent=solvent, do_vacuum=True)

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
            protocol=self._protocol,
            chemical_system_generator=self.chemical_system_generator,
        )

        return self._alchemical_network


class RBFEAlchemicalNetworkPlanner(RelativeAlchemicalNetworkPlanner):
    def _build_chemicalsystem_generator(
        self, solvent: SolventComponent, protein: ProteinComponent
    ) -> AbstractChemicalSystemGenerator:
        chemical_system_generator = self._generator_type(
            solvent=solvent, protein=protein
        )
        return chemical_system_generator

    def _build_alchemical_network(
        self,
        ligands: Iterable[SmallMoleculeComponent],
        solvent: SolventComponent = None,
        receptor: ProteinComponent = None,
    ) -> AlchemicalNetwork:
        # components might be given differently!
        # throw into ligand_network_planning
        self._ligand_network = self._construct_ligand_network(ligands)

        # Prepare system generation
        self._chemical_system_generator = self._build_chemicalsystem_generator(
            solvent=solvent, protein=receptor
        )

        # Build transformations
        self._alchemical_network = self._build_transformations(
            ligand_network_edges=self._ligand_network.edges,
            protocol=self._protocol,
            chemical_system_generator=self.chemical_system_generator,
        )

        return self._alchemical_network
