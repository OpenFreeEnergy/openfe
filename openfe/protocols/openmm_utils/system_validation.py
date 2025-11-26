# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Reusable utility methods to validate input systems to OpenMM-based alchemical
Protocols.
"""

from typing import Optional, Tuple

from gufe import (
    ChemicalSystem,
    Component,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
)


def get_alchemical_components(
    stateA: ChemicalSystem,
    stateB: ChemicalSystem,
) -> dict[str, list[Component]]:
    """
    Checks the equality between Components of two end state ChemicalSystems
    and identify which components do not match.

    Parameters
    ----------
    stateA : ChemicalSystem
      The chemical system of end state A.
    stateB : ChemicalSystem
      The chemical system of end state B.

    Returns
    -------
    alchemical_components : dict[str, list[Component]]
      Dictionary containing a list of alchemical components for each state.

    Raises
    ------
    ValueError
      If there are any duplicate components in states A or B.
    """
    # Check if there are any duplicate components in either state
    for state in [stateA, stateB]:
        comp_list = list(state.components.values())
        unique_comp_list = list(set(comp_list))
        if len(comp_list) != len(unique_comp_list):
            errmsg = f"Duplicate components found in ChemicalSystem: {state}"
            raise ValueError(errmsg)

    alchemical_components: dict[str, list[Component]] = {
        "stateA": [],
        "stateB": [],
    }

    diff = stateA.component_diff(stateB)
    alchemical_components["stateA"].extend(diff[0])
    alchemical_components["stateB"].extend(diff[1])

    return alchemical_components


def validate_solvent(state: ChemicalSystem, nonbonded_method: str):
    """
    Checks that the ChemicalSystem component has the right solvent
    composition for an input nonbonded_methtod.

    Parameters
    ----------
    state : ChemicalSystem
      The chemical system to inspect.
    nonbonded_method : str
      The nonbonded method to be applied for the simulation.

    Raises
    ------
    ValueError
      * If there are multiple SolventComponents in the ChemicalSystem.
      * If there is a SolventComponent and the `nonbonded_method` is
        `nocutoff`.
      * If the SolventComponent solvent is not water.
    """
    solv = state.get_components_of_type(SolventComponent)

    if len(solv) > 0 and nonbonded_method.lower() == "nocutoff":
        errmsg = "nocutoff cannot be used for solvent transformations"
        raise ValueError(errmsg)

    if len(solv) == 0 and nonbonded_method.lower() == "pme":
        errmsg = "PME cannot be used for vacuum transform"
        raise ValueError(errmsg)

    if len(solv) > 1:
        errmsg = "Multiple SolventComponent found, only one is supported"
        raise ValueError(errmsg)

    if len(solv) > 0 and solv[0].smiles != "O":
        errmsg = "Non water solvent is not currently supported"
        raise ValueError(errmsg)


def validate_protein(state: ChemicalSystem):
    """
    Checks that the ChemicalSystem's ProteinComponent are suitable for the
    alchemical protocol.

    Parameters
    ----------
    state : ChemicalSystem
      The chemical system to inspect.

    Raises
    ------
    ValueError
      If there are multiple ProteinComponent in the ChemicalSystem.
    """
    prots = state.get_components_of_type(ProteinComponent)

    if len(prots) > 1:
        errmsg = "Multiple ProteinComponent found, only one is supported"
        raise ValueError(errmsg)


ParseCompRet = Tuple[
    Optional[SolventComponent],
    Optional[ProteinComponent],
    list[SmallMoleculeComponent],
]


def get_components(state: ChemicalSystem) -> ParseCompRet:
    """
    Establish all necessary Components for the transformation.

    Parameters
    ----------
    state : ChemicalSystem
      ChemicalSystem to get all necessary components from.

    Returns
    -------
    solvent_comp : Optional[SolventComponent]
      If it exists, the SolventComponent for the state, otherwise None.
    protein_comp : Optional[ProteinComponent]
      If it exists, the ProteinComponent for the state, otherwise None.
    small_mols : list[SmallMoleculeComponent]
    """

    def _get_single_comps(comp_list, comptype):
        ret_comps = [comp for comp in comp_list if isinstance(comp, comptype)]
        if ret_comps:
            return ret_comps[0]
        else:
            return None

    solvent_comp: Optional[SolventComponent] = _get_single_comps(
        list(state.values()), SolventComponent
    )

    protein_comp: Optional[ProteinComponent] = _get_single_comps(
        list(state.values()), ProteinComponent
    )

    small_mols = []
    for comp in state.components.values():
        if isinstance(comp, SmallMoleculeComponent):
            small_mols.append(comp)

    return solvent_comp, protein_comp, small_mols
