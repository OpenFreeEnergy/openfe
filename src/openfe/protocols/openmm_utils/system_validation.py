# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Reusable utility methods to validate input systems to OpenMM-based alchemical
Protocols.
"""

import logging
import warnings
from typing import Optional, Tuple

from gufe import (
    BaseSolventComponent,
    ChemicalSystem,
    Component,
    ProteinComponent,
    ProteinMembraneComponent,
    SmallMoleculeComponent,
    SolvatedPDBComponent,
    SolventComponent,
)
from openff.toolkit import Molecule as OFFMol

logger = logging.getLogger(__name__)


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


def validate_solvent(state: ChemicalSystem, nonbonded_method: str):
    """
    Checks that the ChemicalSystem component has the right solvent
    composition for an input nonbonded_methtod.

    Supported configurations are:
      * Vacuum (no BaseSolventComponent)
      * One BaseSolventComponent
      * One SolventComponent paired with one SolvatedPDBComponent

    Parameters
    ----------
    state : ChemicalSystem
      The chemical system to inspect.
    nonbonded_method : str
      The nonbonded method to be applied for the simulation.

    Raises
    ------
    ValueError
      * If there are more than two BaseSolventComponents in the ChemicalSystem.
      * If there are multiple SolventComponents or SolvatedPDBComponents in the ChemicalSystem.
      * If `nocutoff` is requested with any BaseSolventComponent present.
      * If there is no BaseSolventComponent and the `nonbonded_method` is `pme`.
      * If the SolventComponent solvent is not water.
    """
    nb_method = nonbonded_method.lower()
    base_solv_comps = state.get_components_of_type(BaseSolventComponent)
    solvation_comps = state.get_components_of_type(SolventComponent)
    solvated_comps = state.get_components_of_type(SolvatedPDBComponent)

    if len(solvated_comps) > 1:
        raise ValueError("Multiple SolvatedPDBComponent found, only one is supported")

    if len(solvation_comps) > 1:
        raise ValueError("Multiple SolventComponent found, only one is supported")

    # Any BaseSolventComponent present → nocutoff is invalid
    if base_solv_comps and nb_method == "nocutoff":
        raise ValueError("nocutoff cannot be used for solvent transformations")

    # Vacuum transform
    if not base_solv_comps:
        if nb_method == "pme":
            raise ValueError("PME cannot be used for vacuum transform")
        return

    # Solvent-specific checks
    if solvation_comps:
        solvent = solvation_comps[0]

        if solvent.smiles != "O":
            raise ValueError("Non water solvent is not currently supported")


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
    prot_comps = state.get_components_of_type(ProteinComponent)

    if len(prot_comps) > 1:
        errmsg = "Multiple ProteinComponent found, only one is supported"
        raise ValueError(errmsg)


def validate_barostat(state: ChemicalSystem, barostat: str):
    """
    Warn if there is a mismatch between the protein component type and barostat.

    A ProteinMembraneComponent should generally be simulated with a
    MonteCarloMembraneBarostat, while non-membrane protein systems should
    use a MonteCarloBarostat.

    Parameters
    ----------
    state : ChemicalSystem
      The chemical system to inspect.
    barostat: str
      The barostat to be applied to the simulation
    """
    prot_comps = state.get_components_of_type(ProteinComponent)
    protein_membrane = state.get_components_of_type(ProteinMembraneComponent)

    if not prot_comps:
        return

    if protein_membrane:
        if barostat != "MonteCarloMembraneBarostat":
            wmsg = (
                "A ProteinMembraneComponent is present, but a membrane-specific "
                "barostat (MonteCarloMembraneBarostat) is not specified. If you "
                "are simulating a system with a membrane, consider using "
                "integrator_settings.barostat='MonteCarloMembraneBarostat'."
            )
            warnings.warn(wmsg)
            logger.warning(wmsg)

    elif barostat == "MonteCarloMembraneBarostat":
        wmsg = (
            "A MonteCarloMembraneBarostat is specified, but no "
            "ProteinMembraneComponent is present. If you are not simulating a "
            "membrane system, consider using "
            "integrator_settings.barostat='MonteCarloBarostat'."
        )
        warnings.warn(wmsg)
        logger.warning(wmsg)


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

    def _get_single_comps(state, comptype):
        comps = state.get_components_of_type(comptype)

        if len(comps) > 0:
            return comps[0]
        else:
            return None

    solvent_comp: Optional[SolventComponent] = _get_single_comps(state, SolventComponent)

    protein_comp: Optional[ProteinComponent] = _get_single_comps(state, ProteinComponent)

    small_mols = state.get_components_of_type(SmallMoleculeComponent)

    return solvent_comp, protein_comp, small_mols
