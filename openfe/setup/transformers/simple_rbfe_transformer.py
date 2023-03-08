from typing import FrozenSet, Dict

from _abstract_transformer import _abstract_transformer
from gufe import AlchemicalNetwork, Protocol, LigandAtomMapping
from .. import Transformation


class simple_rbfe_transformer(_abstract_transformer):
    """
        Simple transformer is a very simplistic transformer, that mapps a defined protocol to a given set of transformations
    """

    def __init__(self, protocol: Protocol, system_generator: callable):
        self.protocol = protocol
        self.system_generator = system_generator

    def __call__(self, alchemical_network_edges: FrozenSet[LigandAtomMapping], name="approach") \
            -> AlchemicalNetwork:
        name = name

        edges = []
        nodes = []

        for edge in alchemical_network_edges:

            for stateA_env, stateB_env in zip(self.system_generator(edge.componentA), self.system_generator(edge.componentB)):
                transformation_name = name + "_" + stateA_env.name + "_" + stateB_env.name

                edges.append(Transformation(stateA=stateA_env,
                                            stateB=stateB_env,
                                            mapping=edge,
                                            name=transformation_name,
                                            protocol=self.protocol)
                             )
                nodes.extend([stateA_env, stateB_env])

        alchemical_network = AlchemicalNetwork(nodes=nodes, edges=edges, name=name)
        return alchemical_network
