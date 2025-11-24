import warnings

from gufe import LigandAtomMapping


def get_alchemical_charge_difference(mapping: LigandAtomMapping) -> int:
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
    warnings.warn(
        "Use gufe.LigandAtomMapping.get_alchemical_charge_difference() instead.", DeprecationWarning
    )
    return mapping.get_alchemical_charge_difference()
