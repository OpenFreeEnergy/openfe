from openfe.setup.chemicalsystem_generator import RFEComponentLabels

# Boolean Test logic lambdas:
ligandC_in_chem_sys = lambda s: RFEComponentLabels.LIGAND in s.components
solventC_in_chem_sys = lambda s: RFEComponentLabels.SOLVENT in s.components
proteinC_in_chem_sys = lambda s: RFEComponentLabels.PROTEIN in s.components