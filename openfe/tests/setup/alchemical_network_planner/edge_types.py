from gufe import Transformation

from ..chemicalsystem_generator.component_checks import proteinC_in_chem_sys, solventC_in_chem_sys, ligandC_in_chem_sys


def both_states_proteinC_edge(edge: Transformation) -> bool:
    return proteinC_in_chem_sys(edge.stateA) and proteinC_in_chem_sys(edge.stateB)


def both_states_solventC_edge(edge: Transformation) -> bool:
    return solventC_in_chem_sys(edge.stateA) and solventC_in_chem_sys(edge.stateB)


def both_states_ligandC_edge(edge: Transformation) -> bool:
    return ligandC_in_chem_sys(edge.stateA) and ligandC_in_chem_sys(edge.stateB)


def r_vacuum_edge(edge: Transformation) -> bool:
    return (
        both_states_ligandC_edge(edge) and not both_states_solventC_edge(edge) and not both_states_proteinC_edge(edge)
    )


def r_solvent_edge(edge: Transformation) -> bool:
    return both_states_ligandC_edge(edge) and both_states_solventC_edge(edge) and not both_states_proteinC_edge(edge)


def r_complex_edge(edge: Transformation) -> bool:
    return both_states_ligandC_edge(edge) and both_states_solventC_edge(edge) and both_states_proteinC_edge(edge)
