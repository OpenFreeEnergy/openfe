# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Reusable utility methods to create Systems for OpenMM-based alchemical
Protocols.
"""
from itertools import product
import logging
from string import ascii_uppercase
import numpy as np
import numpy.typing as npt
import openmm
from openmm import app, MonteCarloBarostat
from openmm import unit as omm_unit
from openff.toolkit import Molecule as OFFMol
from openff.interchange.components._packmol import UNIT_CUBE, pack_box
from openff.units.openmm import to_openmm, ensure_quantity
from openmmforcefields.generators import SystemGenerator
from typing import Optional
from pathlib import Path
from gufe.settings import OpenMMSystemGeneratorFFSettings, ThermoSettings
from gufe import (
    Component, ProteinComponent, SolventComponent, SmallMoleculeComponent
)
from ..openmm_rfe.equil_rfe_settings import (
    SystemSettings, SolvationSettings
)


logger = logging.getLogger(__name__)


def get_system_generator(
    forcefield_settings: OpenMMSystemGeneratorFFSettings,
    thermo_settings: ThermoSettings,
    system_settings: SystemSettings,
    cache: Optional[Path],
    has_solvent: bool,
) -> SystemGenerator:
    """
    Create a SystemGenerator based on Protocol settings.

    Paramters
    ---------
    forcefield_settings : OpenMMSystemGeneratorFFSettings
      Force field settings, including necessary information
      for constraints, hydrogen mass, rigid waters, COM removal,
      non-ligand FF xmls, and the ligand FF name.
    thermo_settings : ThermoSettings
      Thermodynamic settings, including everything necessary to
      create a barostat.
    system_settings : SystemSettings
      System settings including all necessary information for
      the nonbonded methods.
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
        'removeCMMotion': forcefield_settings.remove_com,
        'hydrogenMass': forcefield_settings.hydrogen_mass * omm_unit.amu,
    }

    # get the right nonbonded method
    nonbonded_method = {
        'pme': app.PME,
        'nocutoff': app.NoCutoff,
        'cutoffnonperiodic': app.CutoffNonPeriodic,
        'cutoffperiodic': app.CutoffPeriodic,
        'ewald': app.Ewald
    }[system_settings.nonbonded_method.lower()]

    nonbonded_cutoff = to_openmm(
        system_settings.nonbonded_cutoff
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
                     omm_forcefield: app.ForceField,
                     solvent_settings: SolvationSettings) -> ModellerReturn:
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


def create_packmol_system(
    protein_component: Optional[ProteinComponent],
    solvent_component: Optional[SolventComponent],
    smc_components: dict[SmallMoleculeComponent, OFFMol],
    system_generator: SystemGenerator,
    solvation_settings: SolvationSettings,
) -> tuple[openmm.System, app.Topology, omm_unit.Quantity, dict[Component, npt.NDArray]]:
    """
    Generate an OpenMM system using packmol.

    Parameters
    ----------
    protein_component : Optional[ProteinComponent]
      Protein Component, if it exists.
    solvent_component : Optional[ProteinCompoinent]
      Solvent Component, if it exists.
    smc_components : dict[SmallMoleculeComponent, openff.toolkit.topology.Molecule]
      Dictionary of openff Molecules to add.
    system_generator : openmmforcefields.generator.SystemGenerator
      System Generator to parameterise this unit.
    solvation_settings : SolvationSettings
      Settings detailing how to solvate the system.

    Returns
    -------
    system : openmm.System
      An OpenMM System of the alchemical system.
    topology : app.Topology
      Topology object describing the parameterized system
    positionns : openmm.unit.Quantity
      Positions of the system.
    comp_resids : dict[Component, npt.NDArray]
      Dictionary of residue indices for each component in system.
     """
    def _set_offmol_resname(offmol, resname):
        for a in offmol.atoms:
            a.metadata['residue_name'] = resname

    def _get_offmol_resname(offmol: OFFMol) -> Optional[str]:
        resname: Optional[str] = None
        for a in offmol.atoms:
            if resname is None:
                try:
                    resname = a.metadata['residue_name']
                except KeyError:
                    return None

            if resname != a.metadata['residue_name']:
                wmsg = (f"Inconsistent residue name in OFFMol: {offmol} "
                        "residue name will be overriden")
                logger.warning(wmsg)
                return None

        return resname

    # 0. Do any validation
    if protein_component is not None:
        errmsg = ("This backend is not available for simulations "
                  "involving ProteinComponents")
        raise ValueError(errmsg)

    # 1. Get the solvent components out
    if solvent_component is not None:
        solvent_offmol = OFFMol.from_smiles(solvent_component.smiles)
        solvent_offmol.generate_conformers()
        _set_offmol_resname(solvent_offmol, 'SOL')
        solvent_copies = [solvation_settings.num_solvent_molecules]
    else:
        solvent_offmol = []
        solvent_copies = []

    # 2. Asisgn residue names so we can track our components in the generated
    # topology.

    # Note: comp_resnames is dict[str, list[Component, list]] where the final list
    # is to append residues later on
    comp_resnames = {'SOL': [solvent_component, []]}
    resnames_store = [''.join(i) for i in product(ascii_uppercase, repeat=3)]

    for comp, offmol in smc_components.items():
        off_resname = _get_offmol_resname(offmol)
        if off_resname is None or off_resname in comp_resnames:
            # warn that we are overriding clashing molecule resnames
            if off_resname in comp_resnames:
                wmsg = (f"Duplicate residue name {off_resnames}, "
                        "duplicate will be renamed")
                logger.warning(wmsg)

            # just loop through and pick up a name that doesn't exist
            while (off_resname in comp_resnames) or (off_resname is None):
                off_resname = resnames_store.pop(0)

        wmsg = f"Setting component {comp} residue name to {off_resname}"
        logger.warning(wmsg)
        _set_offmol_resname(offmol, off_resname)
        comp_resnames[off_resname] = [comp, []]

    # 3. Create the packmol topology
    offmols = list(smc_components.values()) + [solvent_offmol]
    offmol_copies = [1 for _ in smc_components] + solvent_copies

    off_topology = pack_box(
        molecules=offmols,
        number_of_copies=offmol_copies,
        mass_density=solvation_settings.box_mass_density,
        box_shape=UNIT_CUBE,  # One day move away from this
    )

    # 4. Extract OpenMM objects
    omm_topology = off_topology.to_openmm()
    omm_positions = to_openmm(off_topology.get_positions())

    # 5. Assign component resids
    # Get all the matching residue indices
    for res in omm_topology.residues():
        comp_resnames[res.name][1].append(res.index)

    # Now create comp_resids dictionary
    comp_resids = {}
    for entry in comp_resnames.values():
        comp = entry[0]
        indices = np.array(entry[1])
        comp_resids[comp] = indices

    # force the creation of parameters for the small molecules
    # this is necessary because we need to have the FF generated ahead
    # of solvating the system.
    # Note by default this is cached to ctx.shared/db.json which should
    # reduce some of the costs.
    for mol in offmols:
        # don't do this if we have user charges
        if not (mol.partial_charges is not None and np.any(mol.partial_charges)):
            try:
                # try and follow official spec method
                mol.assign_partial_charges('am1bcc')
            except ValueError:  # this is what a confgen failure yields
                # but fallback to using existing conformer
                mol.assign_partial_charges('am1bcc',
                                           use_conformers=mol.conformers)

        system_generator.create_system(
            mol.to_topology().to_openmm(), molecules=[mol]
        )

    system = system_generator.create_system(
        omm_topology, molecules=offmols
    )

    return system, omm_topology, omm_positions, comp_resids
