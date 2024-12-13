from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gufe import LigandAtomMapping

def get_alchemical_charge_difference(mapping: "LigandAtomMapping") -> int:
    """
    Return the difference in formal charge between stateA and stateB defined as (formal charge A - formal charge B)

    Parameters
    ----------
    mapping: LigandAtomMapping
        The mapping between the end states A and B.

    Returns
    -------
    int:
        The difference in formal charge between the end states.
    """
    from rdkit import Chem
    charge_a = Chem.rdmolops.GetFormalCharge(mapping.componentA.to_rdkit())
    charge_b = Chem.rdmolops.GetFormalCharge(mapping.componentB.to_rdkit())
    return charge_a - charge_b
