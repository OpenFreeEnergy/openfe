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
from openff.toolkit.utils.toolkits import (
    AmberToolsToolkitWrapper,
    OpenEyeToolkitWrapper,
    RDKitToolkitWrapper,
)
from openff.units.openmm import to_openmm, ensure_quantity
from openff.units import unit as offunit
from openmmforcefields.generators import SystemGenerator
from typing import Optional, Iterable
from pathlib import Path
from gufe.settings import OpenMMSystemGeneratorFFSettings, ThermoSettings
from gufe import (
    Component, ProteinComponent, SolventComponent, SmallMoleculeComponent
)
from ..openmm_rfe.equil_rfe_settings import (
    SystemSettings, SimulationSettings, SolvationSettings
)


try:
    from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
except ImportError:
    HAS_NAGL = False
else:
    HAS_NAGL = True


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
            ionicStrength=to_openmm(conc)
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


def _get_toolkit_wrapper_charge_backend(selection: str):
    """
    Get a ToolkitWrapper for a given charge backend selection.

    Parameters
    ----------
    selection : str
      The charge backend selected. Supported entries are
      `oechem` and `ambertools`.

    Raises
    ------
    ValueError
      If an unrecognised charge backend selection is passed.
    """
    available_backends = {
        'oechem': OpenEyeToolkitWrapper,
        'ambertools': AmberToolsToolkitWrapper,
    }
    try:
        toolkitwrapper = available_backends[selection.lower()]
    except KeyError:
        errmsg = (f"An unknown charge backend was requested {selection} "
                  "available backend options are: {available_backends.keys()}")
        raise ValueError(errmsg)

    return toolkitwrapper


def assign_am1bcc_charges(
    offmol: OFFMol,
    charge_backend: str = 'ambertools',
    conformer: Optional[Iterable[offunit.Quantity]] = None
) -> None:
    """
    Assign AM1BCC charges using a given toolkit.

    Parameters
    ----------
    offmol: openff.toolkit.Molecule
      The Molecule to assign AM1BCC charges for.
    charge_backend : str
      Which backend to use to generate charges. Available options are:

      * `ambertools`: Use AmberTools' antechamber to generate charges.
      * `oechem`: Use the OpenEye toolkit to generate charges.
    """
    backend = _get_toolkit_wrapper_charge_backend(charge_backend)

    try:
        offmol.assign_partial_charges(
            partial_charge_method='am1bcc',
            use_conformers=conformer,
            toolkit_registry=backend(),
        )
    # if we get a confgen failure fallback to using the existing conformer
    except ValueError:
        offmol.assign_partial_charges(
            partial_charge_method='am1bcc',
            use_conformers=offmol.conformers,
            toolkit_registry=backend(),
        )


def assign_nagl_am1bcc_charges(
    offmol: OFFMol,
    nagl_model: Optional[str] = "openff-gnn-am1bcc-0.0.1-rc.1.pt"
) -> None:
    """
    Assign partial charges using the NAGL ML Model.

    Parameters
    ----------
    offmol: openff.toolkit.Molecule
      The Molecule to assign AM1BCC charges for.
    nagl_model : Optional[str]
      The nagl model to use for partial charge assignment.
      If None, will use latest available model.
    """
    if not HAS_NAGL:
        errmsg = ("The NAGLToolkitWrapper is not available, you may be using "
                  "and older version of the OpenFF toolkit - you need 0.14.4 "
                  "or above.")
        raise ImportError(errmsg)

    if nagl_model is None:
        # It's not fully clear that the models will always be sort ordered
        # see: https://github.com/openforcefield/openff-nagl-models/issues/12
        from openff.nagl_models import list_available_nagl_models
        nagl_model = list_available_models()[-1]

    offmol.assign_partial_charges(
        partial_charge_method=nagl_model,
        toolkit_registry=NAGLToolkitWrapper(),
    )


def assign_am1bccelf10_charges(
    offmol: OFFMol,
    charge_backend: Optional[str] = 'ambertools',
    ambertools_generate_n_conformers: int = 500,
) -> None:
    """
    Assign AM1BCC charges using a given toolkit.

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      The Molecule to assign AM1BCCELF10 charges for.
    charge_backend : str
      Which backend to use to generate charges. Available options are:

      * `ambertools`: Use AmberTools' antechamber to generate charges.
      * `oechem`: Use the OpenEye toolkit to generate charges.

    ambertools_generate_n_conformers : int
      The number of conformers to initially generate for elf10 selection.
      Note: this is only used by the `ambertools` backend option.
    """
    backend = _get_toolkit_wrapper_charge_backend(charge_backend)

    if charge_backend.lower() == 'oechem':
        offmol.assign_partial_charges(
            partial_charge_method='am1bccelf10',
            toolkit_registry=backend(),
        )

    else:
        # make a copy of the offmol to avoid overwriting conformers
        # TODO: in tests ensure that the offmol conformer doesn't change
        # after charge asisgnment for any of these methods!
        offmol_copy = OFFMol(offmol)

        # We generate conformers using RDKit to be consistent
        offmol_copy.generate_conformers(
            # The OFF recommended default for OpenEye is 500
            n_conformers=ambertools_generate_n_conformers,
            rms_cutoff=0.25 * offunit.angstrom,
            toolkit_registry=RDKitToolkitWrapper()
        )

        # Next we apply the elf10 conformer selection
        offmol_copy.apply_elf_conformer_selection(
            percentage=2,
            limit=10,
            rms_tolerance=0.05 * offunit.angstrom,  # Should this be an option?
        )

        # Then we loop over the selected conformers
        charges = np.zeros([offmol_copy.n_atoms])

        for conf in offmol_copy.conformers:
            offmol_copy.assign_partial_charges('am1bcc', use_conformers=[conf])
            charges += offmol_copy.partial_charges

        charges /= len(offmol_copy.conformers)

        offmol.partial_charges = charges


def assign_offmol_partial_charges(
    offmol,
    method: str,
    charge_backend: str = "ambertools",
    use_conformer: bool = True,
    ambertools_elf_generate_n_conformers: int = 500,
    nagl_model: Optional[str] = "openff-gnn-am1bcc-0.0.1-rc.1.pt"
) -> None:
    """
    Assign partial charges to an OpenFF Molecule based on selected method.


    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      The Molecule to assign charges for.
    method : str
      The method to use for charge assignement.
      Supported methods include: `am1bcc`, `am1bccelf10`, and `nagl`.
    charge_backend : str
      The charge backend used for `am1bcc` or `am1bccelf10` charges.
      Default is `ambertools`.
    use_conformer : bool
      If method is `am1bcc`, whether or not to use the existing
      conformer for charge generation. Default True.
    ambertools_elf_generate_n_conformers : int
      If method is `am1bccelf10` and charge_backend is `ambertools`,
      the number of initial conformers to generate prior to ELF10
      filtering. Default 500.
    nagl_model : Optional[str]
      The NAGL model to use for charge assignment if method is `nagl`.
      If None, the last model returned by
      openff.nagl_model.list_available_models will be used.
      Default to openff-gnn-am1bcc-0.0.1-rc.1.pt

    Raises
    ------
    ValueError
      If an unknown charge method is provided.
    """
    if method.lower() == 'am1bcc':
        if use_conformer:
            conformer = offmol.conformers
        else:
            conformer = None

        assign_am1bcc_charges(offmol, charge_backend, conformer)

    elif method.lower() == 'am1bccelf10':
        assign_am1bccelf10_charges(
            offmol, charge_backend, ambertools_elf_generate_n_conformers
        )
    elif method.lower() == 'nagl':
        assign_nagl_am1bcc_charges(offmol, nagl_model)
    else:
        errmsg = (f"Unknown charge method: {method} requested. "
                  "Available methods are: `am1bcc`, `am1bccelf10`, and `nagl`")
        raise ValueError(errmsg)
