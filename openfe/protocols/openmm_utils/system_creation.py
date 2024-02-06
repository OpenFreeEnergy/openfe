# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Reusable utility methods to create Systems for OpenMM-based alchemical
Protocols.
"""
import numpy as np
import numpy.typing as npt
from openmm import app, MonteCarloBarostat
from openmm import unit as omm_unit
from openff.toolkit import Molecule as OFFMol
from openff.units.openmm import to_openmm, ensure_quantity
from openmmforcefields.generators import SystemGenerator
from typing import Optional
from pathlib import Path
from gufe.settings import OpenMMSystemGeneratorFFSettings, ThermoSettings
from gufe import (
    Component, ProteinComponent, SolventComponent, SmallMoleculeComponent
)
from ..openmm_rfe.equil_rfe_settings import (
    SolvationSettings, IntegratorSettings,
)


def get_system_generator(
    forcefield_settings: OpenMMSystemGeneratorFFSettings,
    thermo_settings: ThermoSettings,
    integrator_settings: IntegratorSettings,
    cache: Optional[Path],
    has_solvent: bool,
) -> SystemGenerator:
    """
    Create a SystemGenerator based on Protocol settings.

    Paramters
    ---------
    forcefield_settings : OpenMMSystemGeneratorFFSettings
      Force field settings, including necessary information
      for constraints, hydrogen mass, rigid waters,
      non-ligand FF xmls, and the ligand FF name.
    integrator_settings: IntegratorSettings
      Integrator settings, including COM removal.
    thermo_settings : ThermoSettings
      Thermodynamic settings, including necessary settings
      for defining the ensemble conditions.
    integrator_settings : IntegratorSettings
      Integrator settings, including barostat control variables.
    cache : Optional[pathlib.Path]
      Path to openff force field cache.
    has_solvent : bool
      Whether or not the target system has solvent (and by extension
      might require a barostat).

    Returns
    -------
    system_generator : openmmforcefields.generator.SystemGenerator
      System Generator to use for this Protocol.

    TODO
    ----
    * Investigate how RF can be passed to non-periodic kwargs.
    """
    # get the right constraint
    constraints = {
        'hbonds': app.HBonds,
        'none': None,
        'allbonds': app.AllBonds,
        'hangles': app.HAngles
        # vvv can be None so string it
    }[str(forcefield_settings.constraints).lower()]

    # create forcefield_kwargs entry
    forcefield_kwargs = {
        'constraints': constraints,
        'rigidWater': forcefield_settings.rigid_water,
        'removeCMMotion': integrator_settings.remove_com,
        'hydrogenMass': forcefield_settings.hydrogen_mass * omm_unit.amu,
    }

    # get the right nonbonded method
    nonbonded_method = {
        'pme': app.PME,
        'nocutoff': app.NoCutoff,
        'cutoffnonperiodic': app.CutoffNonPeriodic,
        'cutoffperiodic': app.CutoffPeriodic,
        'ewald': app.Ewald
    }[forcefield_settings.nonbonded_method.lower()]

    nonbonded_cutoff = to_openmm(
        forcefield_settings.nonbonded_cutoff,  # type: ignore
    )

    # create the periodic_kwarg entry
    periodic_kwargs = {
        'nonbondedMethod': nonbonded_method,
        'nonbondedCutoff': nonbonded_cutoff,
    }

    # Currently the else is a dead branch, we will want to investigate the
    # possibility of using CutoffNonPeriodic at some point though (for RF)
    if nonbonded_method is not app.CutoffNonPeriodic:
        nonperiodic_kwargs = {
                'nonbondedMethod': app.NoCutoff,
        }
    else:  # pragma: no-cover
        nonperiodic_kwargs = periodic_kwargs

    # Add barostat if necessary
    # TODO: move this to its own place where we can handle membranes
    if has_solvent:
        barostat = MonteCarloBarostat(
            ensure_quantity(thermo_settings.pressure, 'openmm'),
            ensure_quantity(thermo_settings.temperature, 'openmm'),
            integrator_settings.barostat_frequency.m,
        )
    else:
        barostat = None

    system_generator = SystemGenerator(
        forcefields=forcefield_settings.forcefields,
        small_molecule_forcefield=forcefield_settings.small_molecule_forcefield,
        forcefield_kwargs=forcefield_kwargs,
        nonperiodic_forcefield_kwargs=nonperiodic_kwargs,
        periodic_forcefield_kwargs=periodic_kwargs,
        cache=str(cache) if cache is not None else None,
        barostat=barostat,
    )

    return system_generator


ModellerReturn = tuple[app.Modeller, dict[Component, npt.NDArray]]


def get_omm_modeller(protein_comp: Optional[ProteinComponent],
                     solvent_comp: Optional[SolventComponent],
                     small_mols: dict[SmallMoleculeComponent, OFFMol],
                     omm_forcefield : app.ForceField,
                     solvent_settings : SolvationSettings) -> ModellerReturn:
    """
    Generate an OpenMM Modeller class based on a potential input ProteinComponent,
    SolventComponent, and a set of small molecules.

    Parameters
    ----------
    protein_comp : Optional[ProteinComponent]
      Protein Component, if it exists.
    solvent_comp : Optional[ProteinCompoinent]
      Solvent Component, if it exists.
    small_mols : dict
      Small molecules to add.
    omm_forcefield : app.ForceField
      ForceField object for system.
    solvent_settings : SolvationSettings
      Solvation settings.

    Returns
    -------
    system_modeller : app.Modeller
      OpenMM Modeller object generated from ProteinComponent and
      OpenFF Molecules.
    component_resids : dict[Component, npt.NDArray]
      Dictionary of residue indices for each component in system.
    """
    component_resids = {}

    def _add_small_mol(comp,
                       mol,
                       system_modeller: app.Modeller,
                       comp_resids: dict[Component, npt.NDArray]):
        """
        Helper method to add OFFMol to an existing Modeller object and
        update a dictionary tracking residue indices for each component.
        """
        omm_top = mol.to_topology().to_openmm()
        system_modeller.add(
            omm_top,
            ensure_quantity(mol.conformers[0], 'openmm')
        )

        nres = omm_top.getNumResidues()
        resids = [res.index for res in system_modeller.topology.residues()]
        comp_resids[comp] = np.array(resids[-nres:])

    # Create empty modeller
    system_modeller = app.Modeller(app.Topology(), [])

    # If there's a protein in the system, we add it first to the Modeller
    if protein_comp is not None:
        system_modeller.add(protein_comp.to_openmm_topology(),
                            protein_comp.to_openmm_positions())
        # add missing virtual particles (from crystal waters)
        system_modeller.addExtraParticles(omm_forcefield)
        component_resids[protein_comp] = np.array(
          [r.index for r in system_modeller.topology.residues()]
        )
        # if we solvate temporarily rename water molecules to 'WAT'
        # see openmm issue #4103
        if solvent_comp is not None:
            for r in system_modeller.topology.residues():
                if r.name == 'HOH':
                    r.name = 'WAT'

    # Now loop through small mols
    for comp, mol in small_mols.items():
        _add_small_mol(comp, mol, system_modeller, component_resids)

    # Add solvent if neeeded
    if solvent_comp is not None:
        conc = solvent_comp.ion_concentration
        pos = solvent_comp.positive_ion
        neg = solvent_comp.negative_ion

        system_modeller.addSolvent(
            omm_forcefield,
            model=solvent_settings.solvent_model,
            padding=to_openmm(solvent_settings.solvent_padding),
            positiveIon=pos, negativeIon=neg,
            ionicStrength=to_openmm(conc),
            neutralize=solvent_comp.neutralize,
        )

        all_resids = np.array(
            [r.index for r in system_modeller.topology.residues()]
        )

        existing_resids = np.concatenate(
            [resids for resids in component_resids.values()]
        )

        component_resids[solvent_comp] = np.setdiff1d(
            all_resids, existing_resids
        )
        # undo rename of pre-existing waters
        for r in system_modeller.topology.residues():
            if r.name == 'WAT':
                r.name = 'HOH'

    return system_modeller, component_resids

