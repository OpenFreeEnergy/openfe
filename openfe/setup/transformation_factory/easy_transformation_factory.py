# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import FrozenSet, Dict

from gufe import (
    AlchemicalNetwork,
    Protocol,
    LigandAtomMapping,
    Transformation,
    SmallMoleculeComponent,
)
from gufe import Transformation

from .abstract_transformation_factory import AbstractTransformerFactory
from ..chemicalsystem_generator.abstract_chemicalsystem_generator import (
    AbstractChemicalSystemGenerator,
)


class AbstractEasyTransformationFactory(AbstractTransformerFactory):
    def __init__(
        self,
        protocol: Protocol,
        chemical_system_generator: AbstractChemicalSystemGenerator,
    ):
        self.protocol = protocol
        self.chemical_system_generator = chemical_system_generator


class RFETransformationFactory(AbstractEasyTransformationFactory):
    """
    Simple transformer is a very simplistic transformer, that mapps a defined protocol to a given set of transformations
    """

    def __call__(
        self, alchemical_network_edges: FrozenSet[LigandAtomMapping], name="approach"
    ) -> AlchemicalNetwork:
        return self._build_alchemical_network(
            alchemical_network_edges=alchemical_network_edges, name=name
        )

    def _build_alchemical_network(
        self, alchemical_network_edges: FrozenSet[LigandAtomMapping], name="approach"
    ) -> AlchemicalNetwork:
        name = name
        edges = []
        nodes = []

        for edge in alchemical_network_edges:

            for stateA_env, stateB_env in zip(
                self.chemical_system_generator(edge.componentA),
                self.chemical_system_generator(edge.componentB),
            ):
                transformation_name = (
                    name + "_" + stateA_env.name + "_" + stateB_env.name
                )

                edges.append(
                    Transformation(
                        stateA=stateA_env,
                        stateB=stateB_env,
                        mapping={"ligand": edge}, #Todo: dirty hack!
                        name=transformation_name,
                        protocol=self.protocol,
                    )
                )
                nodes.extend([stateA_env, stateB_env])

        alchemical_network = AlchemicalNetwork(nodes=nodes, edges=edges, name=name)
        return alchemical_network
