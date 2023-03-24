from ..chemicalsystem_generator.component_checks import proteinC_in_chem_sys, solventC_in_chem_sys, ligandC_in_chem_sys

both_states_proteinC_edge = lambda e: proteinC_in_chem_sys(e.stateA) and proteinC_in_chem_sys(
    e.stateB
)
both_states_solventC_edge = lambda e: solventC_in_chem_sys(e.stateA) and solventC_in_chem_sys(
    e.stateB
)
both_states_ligandC_edge = lambda e: ligandC_in_chem_sys(e.stateA) and ligandC_in_chem_sys(
    e.stateB
)

r_vacuum_edge = (
    lambda e: both_states_ligandC_edge(e)
    and not both_states_solventC_edge(e)
    and not both_states_proteinC_edge(e)
)
r_solvent_edge = (
    lambda e: both_states_ligandC_edge(e)
    and both_states_solventC_edge(e)
    and not both_states_proteinC_edge(e)
)
r_complex_edge = (
    lambda e: both_states_ligandC_edge(e)
    and both_states_solventC_edge(e)
    and both_states_proteinC_edge(e)
)
