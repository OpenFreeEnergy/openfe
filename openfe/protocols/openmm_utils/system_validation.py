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
    ProteinMembraneComponent,
    SmallMoleculeComponent,
    SolventComponent,
)
from openff.toolkit import Molecule as OFFMol


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
    matched_components: dict[Component, Component] = {}
    alchemical_components: dict[str, list[Component]] = {
        "stateA": [],
        "stateB": [],
    }

    for keyA, valA in stateA.components.items():
        for keyB, valB in stateB.components.items():
            if valA == valB:
                if valA not in matched_components.keys():
                    matched_components[valA] = valB
                else:
                    # Could be that either we have a duplicate component
                    # in stateA or in stateB
                    errmsg = (
                        f"state A components {keyA}: {valA} matches "
                        "multiple components in stateA or stateB"
                    )
                    raise ValueError(errmsg)

    # populate stateA alchemical components
    for valA in stateA.components.values():
        if valA not in matched_components.keys():
            alchemical_components["stateA"].append(valA)

    # populate stateB alchemical components
    for valB in stateB.components.values():
        if valB not in matched_components.values():
            alchemical_components["stateB"].append(valB)

    return alchemical_components


def require_components(systems, component_types, msg=None):
    """
    Ensure that every system in `systems` contains at least one component
    whose type is in `component_types`.

    Parameters
    ----------
    systems : iterable
        Chemical systems / states to check (e.g., stateA, stateB).
    component_types : tuple[type]
        Component classes to test for (e.g., (ProteinComponent, ProteinMembraneComponent)).
    msg : str, optional
        Custom error message.

    Raises
    ------
    ValueError
        If any system does not contain at least one component of the allowed types.
    """
    for i, system in enumerate(systems):
        if not any(system.contains(t) for t in component_types):
            default = (
                f"System {i} does not contain any of the required component types: "
                f"{component_types}"
            )
            raise ValueError(msg or default)


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
      * If there is no SolventComponent and no explicitly solvated
        ProteinMembraneComponent and the `nonbonded_method` is `pme`.
      * If the SolventComponent solvent is not water.
    """
    solv = [comp for comp in state.values() if isinstance(comp, SolventComponent)]
    # ToDo: Also validate the solvent in the ProteinMembraneComponent?
    protein_membrane = state.get_components_of_type(ProteinMembraneComponent)

    if len(solv) > 0 and nonbonded_method.lower() == "nocutoff":
        errmsg = "nocutoff cannot be used for solvent transformations"
        raise ValueError(errmsg)

    if len(solv) == 0 and len(protein_membrane) == 0 and nonbonded_method.lower() == "pme":
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
    nprot = sum(1 for comp in state.values() if isinstance(comp, ProteinComponent))

    if nprot > 1:
        errmsg = "Multiple ProteinComponent found, only one is supported"
        raise ValueError(errmsg)

    protein_membrane = state.get_components_of_type(ProteinMembraneComponent)

    if len(protein_membrane) == 1 and not protein_membrane[0]._periodic_box_vectors:
        errmsg = "No periodic box vector is supplied with the ProteinMembraneComponent"
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
