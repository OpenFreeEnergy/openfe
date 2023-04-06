from gufe import ChemicalSystem

from openfe.setup.chemicalsystem_generator import RFEComponentLabels


# Boolean Test logic lambdas:
def ligandC_in_chem_sys(chemical_system: ChemicalSystem) -> bool:
    return RFEComponentLabels.LIGAND in chemical_system.components


def solventC_in_chem_sys(chemical_system: ChemicalSystem) -> bool:
    return RFEComponentLabels.SOLVENT in chemical_system.components


def proteinC_in_chem_sys(chemical_system: ChemicalSystem) -> bool:
    return RFEComponentLabels.PROTEIN in chemical_system.components
