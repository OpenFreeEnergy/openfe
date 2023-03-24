from openfe.setup.chemicalsystem_generator import RFEComponentLabels


# Boolean Test logic lambdas:
ligandC_in_state = lambda s: RFEComponentLabels.LIGAND in s.components
solventC_in_state = lambda s: RFEComponentLabels.SOLVENT in s.components
proteinC_in_state = lambda s: RFEComponentLabels.PROTEIN in s.components

both_states_proteinC_edge = lambda e: proteinC_in_state(e.stateA) and proteinC_in_state(
    e.stateB
)
both_states_solventC_edge = lambda e: solventC_in_state(e.stateA) and solventC_in_state(
    e.stateB
)
both_states_ligandC_edge = lambda e: ligandC_in_state(e.stateA) and ligandC_in_state(
    e.stateB
)

r_vacuum_edge = (
    lambda e: both_states_ligandC_edge
    and not both_states_solventC_edge(e)
    and not both_states_proteinC_edge(e)
)
r_solvent_edge = (
    lambda e: both_states_ligandC_edge
    and both_states_solventC_edge(e)
    and not both_states_proteinC_edge(e)
)
r_complex_edge = (
    lambda e: both_states_ligandC_edge
    and both_states_solventC_edge(e)
    and both_states_proteinC_edge(e)
)
