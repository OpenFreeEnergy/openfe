# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import logging

import pytest
from openff.units import unit as offunit

import openfe
from openfe import setup
from openfe.protocols import openmm_rfe


@pytest.fixture()
def vac_settings():
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    settings.forcefield_settings.nonbonded_method = "nocutoff"
    settings.engine_settings.compute_platform = None
    settings.protocol_repeats = 1
    return settings


@pytest.fixture()
def solv_settings():
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    settings.engine_settings.compute_platform = None
    settings.protocol_repeats = 1
    return settings


def test_invalid_protocol_repeats():
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    with pytest.raises(ValueError, match="must be a positive value"):
        settings.protocol_repeats = -1


@pytest.mark.parametrize("state", ["A", "B"])
def test_endstate_two_alchemcomp_stateA(state, benzene_modifications):
    first_state = openfe.ChemicalSystem(
        {
            "ligandA": benzene_modifications["benzene"],
            "ligandB": benzene_modifications["toluene"],
            "solvent": openfe.SolventComponent(),
        }
    )
    other_state = openfe.ChemicalSystem(
        {
            "ligandC": benzene_modifications["phenol"],
            "solvent": openfe.SolventComponent(),
        }
    )

    if state == "A":
        args = (first_state, other_state)
    else:
        args = (other_state, first_state)

    with pytest.raises(ValueError, match="Only one alchemical component"):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_endstates(*args)


@pytest.mark.parametrize("state", ["A", "B"])
def test_endstates_not_smc(state, benzene_modifications):
    first_state = openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications["benzene"],
            "foo": openfe.SolventComponent(),
        }
    )
    other_state = openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications["benzene"],
            "foo": benzene_modifications["toluene"],
        }
    )

    if state == "A":
        args = (first_state, other_state)
    else:
        args = (other_state, first_state)

    errmsg = "only SmallMoleculeComponents transformations"
    with pytest.raises(ValueError, match=errmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_endstates(*args)


def test_validate_mapping_none_mapping():
    errmsg = "A single LigandAtomMapping is expected"
    with pytest.raises(ValueError, match=errmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_mapping(None, None)


def test_validate_mapping_multi_mapping(benzene_to_toluene_mapping):
    errmsg = "A single LigandAtomMapping is expected"
    with pytest.raises(ValueError, match=errmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_mapping(
            [benzene_to_toluene_mapping] * 2, None
        )


@pytest.mark.parametrize("state", ["A", "B"])
def test_validate_mapping_alchem_not_in(state, benzene_to_toluene_mapping):
    errmsg = f"not in alchemical components of state{state}"

    if state == "A":
        alchem_comps = {"stateA": [], "stateB": [benzene_to_toluene_mapping.componentB]}
    else:
        alchem_comps = {"stateA": [benzene_to_toluene_mapping.componentA], "stateB": []}

    with pytest.raises(ValueError, match=errmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_mapping(
            [benzene_to_toluene_mapping],
            alchem_comps,
        )


def test_vaccuum_PME_error(
    benzene_vacuum_system, toluene_vacuum_system, benzene_to_toluene_mapping, solv_settings
):
    p = openmm_rfe.RelativeHybridTopologyProtocol(settings=solv_settings)

    errmsg = "PME cannot be used for vacuum transform"
    with pytest.raises(ValueError, match=errmsg):
        p.validate(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            mapping=benzene_to_toluene_mapping,
        )


@pytest.mark.parametrize("charge", [None, "gasteiger"])
def test_smcs_same_charge_passes(charge, benzene_modifications):
    benzene = benzene_modifications["benzene"]
    if charge is None:
        smc = benzene
    else:
        offmol = benzene.to_openff()
        offmol.assign_partial_charges(partial_charge_method="gasteiger")
        smc = openfe.SmallMoleculeComponent.from_openff(offmol)

    # Just pass the same thing twice
    state = openfe.ChemicalSystem({"l": smc})
    openmm_rfe.RelativeHybridTopologyProtocol._validate_smcs(state, state)


def test_smcs_different_charges_none_not_none(benzene_modifications):
    # smcA has no charges
    smcA = benzene_modifications["benzene"]

    # smcB has charges
    offmol = smcA.to_openff()
    offmol.assign_partial_charges(partial_charge_method="gasteiger")
    smcB = openfe.SmallMoleculeComponent.from_openff(offmol)

    state = openfe.ChemicalSystem({"a": smcA, "b": smcB})

    errmsg = "isomorphic but with different charges"
    with pytest.raises(ValueError, match=errmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_smcs(state, state)


def test_smcs_different_charges_all(benzene_modifications):
    offmol = benzene_modifications["benzene"].to_openff()
    offmol.assign_partial_charges(partial_charge_method="gasteiger")
    smcA = openfe.SmallMoleculeComponent.from_openff(offmol)

    # now alter the offmol charges, scaling by 0.1
    offmol.partial_charges *= 0.1
    smcB = openfe.SmallMoleculeComponent.from_openff(offmol)

    state = openfe.ChemicalSystem({"l1": smcA, "l2": smcB})

    errmsg = "isomorphic but with different charges"
    with pytest.raises(ValueError, match=errmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_smcs(state, state)


def test_smcs_different_charges_different_endstates(benzene_modifications):
    # This should just pass, the charge is different but only
    # in the end states - which is an acceptable transformation.
    offmol = benzene_modifications["benzene"].to_openff()
    offmol.assign_partial_charges(partial_charge_method="gasteiger")
    smcA = openfe.SmallMoleculeComponent.from_openff(offmol)

    # now alter the offmol charges, scaling by 0.1
    offmol.partial_charges *= 0.1
    smcB = openfe.SmallMoleculeComponent.from_openff(offmol)

    stateA = openfe.ChemicalSystem({"l": smcA})
    stateB = openfe.ChemicalSystem({"l": smcB})

    openmm_rfe.RelativeHybridTopologyProtocol._validate_smcs(stateA, stateB)


def test_solvent_nocutoff_error(
    benzene_system,
    toluene_system,
    benzene_to_toluene_mapping,
    vac_settings,
):
    p = openmm_rfe.RelativeHybridTopologyProtocol(settings=vac_settings)

    errmsg = "nocutoff cannot be used for solvent transformation"

    with pytest.raises(ValueError, match=errmsg):
        p.validate(
            stateA=benzene_system,
            stateB=toluene_system,
            mapping=benzene_to_toluene_mapping,
        )


def test_nonwater_solvent_error(
    benzene_modifications,
    benzene_to_toluene_mapping,
    solv_settings,
):
    solvent = openfe.SolventComponent(smiles="C")
    stateA = openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications["benzene"],
            "solvent": solvent,
        }
    )

    stateB = openfe.ChemicalSystem({"ligand": benzene_modifications["toluene"], "solvent": solvent})

    p = openmm_rfe.RelativeHybridTopologyProtocol(settings=solv_settings)

    errmsg = "Non water solvent is not currently supported"

    with pytest.raises(ValueError, match=errmsg):
        p.validate(
            stateA=stateA,
            stateB=stateB,
            mapping=benzene_to_toluene_mapping,
        )


def test_too_many_solv_comps_error(
    benzene_modifications,
    benzene_to_toluene_mapping,
    solv_settings,
):
    stateA = openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications["benzene"],
            "solvent!": openfe.SolventComponent(neutralize=True),
            "solvent2": openfe.SolventComponent(neutralize=False),
        }
    )

    stateB = openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications["toluene"],
            "solvent!": openfe.SolventComponent(neutralize=True),
            "solvent2": openfe.SolventComponent(neutralize=False),
        }
    )

    p = openmm_rfe.RelativeHybridTopologyProtocol(settings=solv_settings)

    errmsg = "Multiple SolventComponent found, only one is supported"

    with pytest.raises(ValueError, match=errmsg):
        p.validate(
            stateA=stateA,
            stateB=stateB,
            mapping=benzene_to_toluene_mapping,
        )


def test_bad_solv_settings(
    benzene_system,
    toluene_system,
    benzene_to_toluene_mapping,
    solv_settings,
):
    """
    Test a case where the solvent settings would be wrong.
    Not doing every cases since those are covered under
    ``test_openmmutils.py``.
    """
    solv_settings.solvation_settings.solvent_padding = 1.2 * offunit.nanometer
    solv_settings.solvation_settings.number_of_solvent_molecules = 20

    p = openmm_rfe.RelativeHybridTopologyProtocol(settings=solv_settings)

    errmsg = "Only one of solvent_padding, number_of_solvent_molecules,"
    with pytest.raises(ValueError, match=errmsg):
        p.validate(stateA=benzene_system, stateB=toluene_system, mapping=benzene_to_toluene_mapping)


def test_too_many_prot_comps_error(
    benzene_modifications,
    benzene_to_toluene_mapping,
    T4_protein_component,
    eg5_protein,
    solv_settings,
):
    stateA = openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications["benzene"],
            "solvent": openfe.SolventComponent(),
            "protein1": T4_protein_component,
            "protein2": eg5_protein,
        }
    )

    stateB = openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications["toluene"],
            "solvent": openfe.SolventComponent(),
            "protein1": T4_protein_component,
            "protein2": eg5_protein,
        }
    )

    p = openmm_rfe.RelativeHybridTopologyProtocol(settings=solv_settings)

    errmsg = "Multiple ProteinComponent found, only one is supported"

    with pytest.raises(ValueError, match=errmsg):
        p.validate(
            stateA=stateA,
            stateB=stateB,
            mapping=benzene_to_toluene_mapping,
        )


def test_element_change_warning(atom_mapping_basic_test_files):
    # check a mapping with element change gets rejected early
    l1 = atom_mapping_basic_test_files["2-methylnaphthalene"]
    l2 = atom_mapping_basic_test_files["2-naftanol"]

    # We use the 'old' lomap defaults because the
    # basic test files inputs we use aren't fully aligned
    mapper = setup.LomapAtomMapper(
        time=20, threed=True, max3d=1000.0, element_change=True, seed="", shift=True
    )

    mapping = next(mapper.suggest_mappings(l1, l2))

    alchem_comps = {"stateA": [l1], "stateB": [l2]}

    with pytest.warns(UserWarning, match="Element change"):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_mapping(
            [mapping],
            alchem_comps,
        )


def test_charge_difference_no_corr(benzene_to_benzoic_mapping):
    wmsg = (
        "A charge difference of 1 is observed between the end states. "
        "No charge correction has been requested"
    )

    with pytest.warns(UserWarning, match=wmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_charge_difference(
            benzene_to_benzoic_mapping,
            "pme",
            False,
            openfe.SolventComponent(),
        )


def test_charge_difference_no_solvent(benzene_to_benzoic_mapping):
    errmsg = "Cannot use explicit charge correction without solvent"

    with pytest.raises(ValueError, match=errmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_charge_difference(
            benzene_to_benzoic_mapping,
            "pme",
            True,
            None,
        )


def test_charge_difference_no_pme(benzene_to_benzoic_mapping):
    errmsg = "Explicit charge correction when not using PME"

    with pytest.raises(ValueError, match=errmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_charge_difference(
            benzene_to_benzoic_mapping,
            "nocutoff",
            True,
            openfe.SolventComponent(),
        )


def test_greater_than_one_charge_difference_error(aniline_to_benzoic_mapping):
    errmsg = "A charge difference of 2"
    with pytest.raises(ValueError, match=errmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_charge_difference(
            aniline_to_benzoic_mapping,
            "pme",
            True,
            openfe.SolventComponent(),
        )


@pytest.mark.parametrize(
    "mapping_name,result",
    [
        ["benzene_to_toluene_mapping", 0],
        ["benzene_to_benzoic_mapping", 1],
        ["benzene_to_aniline_mapping", -1],
        ["aniline_to_benzene_mapping", 1],
    ],
)
def test_get_charge_difference(mapping_name, result, request, caplog):
    mapping = request.getfixturevalue(mapping_name)
    caplog.set_level(logging.INFO)

    ion = r"Na+" if result == -1 else r"Cl-"
    msg = (
        f"A charge difference of {result} is observed "
        "between the end states. This will be addressed by "
        f"transforming a water into a {ion} ion"
    )

    openmm_rfe.RelativeHybridTopologyProtocol._validate_charge_difference(
        mapping, "pme", True, openfe.SolventComponent()
    )

    if result != 0:
        assert msg in caplog.text
    else:
        assert msg not in caplog.text


def test_hightimestep(
    benzene_vacuum_system,
    toluene_vacuum_system,
    benzene_to_toluene_mapping,
    vac_settings,
):
    vac_settings.forcefield_settings.hydrogen_mass = 1.0

    p = openmm_rfe.RelativeHybridTopologyProtocol(settings=vac_settings)

    errmsg = "too large for hydrogen mass"

    with pytest.raises(ValueError, match=errmsg):
        p.validate(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            mapping=benzene_to_toluene_mapping,
            extends=None,
        )


def test_time_per_iteration_divmod(
    benzene_vacuum_system,
    toluene_vacuum_system,
    benzene_to_toluene_mapping,
    vac_settings,
):
    vac_settings.simulation_settings.time_per_iteration = 10 * offunit.ps
    vac_settings.integrator_settings.timestep = 4 * offunit.ps

    p = openmm_rfe.RelativeHybridTopologyProtocol(settings=vac_settings)

    errmsg = "does not evenly divide by the timestep"

    with pytest.raises(ValueError, match=errmsg):
        p.validate(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            mapping=benzene_to_toluene_mapping,
            extends=None,
        )


@pytest.mark.parametrize("attribute", ["equilibration_length", "production_length"])
def test_simsteps_not_timestep_divisible(
    attribute,
    benzene_vacuum_system,
    toluene_vacuum_system,
    benzene_to_toluene_mapping,
    vac_settings,
):
    setattr(vac_settings.simulation_settings, attribute, 102 * offunit.fs)
    p = openmm_rfe.RelativeHybridTopologyProtocol(settings=vac_settings)

    errmsg = "Simulation time not divisible by timestep"

    with pytest.raises(ValueError, match=errmsg):
        p.validate(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            mapping=benzene_to_toluene_mapping,
            extends=None,
        )


@pytest.mark.parametrize("attribute", ["equilibration_length", "production_length"])
def test_simsteps_not_mcstep_divisible(
    attribute,
    benzene_vacuum_system,
    toluene_vacuum_system,
    benzene_to_toluene_mapping,
    vac_settings,
):
    setattr(vac_settings.simulation_settings, attribute, 102 * offunit.ps)
    p = openmm_rfe.RelativeHybridTopologyProtocol(settings=vac_settings)

    errmsg = "should contain a number of steps divisible by the number of integrator timesteps"

    with pytest.raises(ValueError, match=errmsg):
        p.validate(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            mapping=benzene_to_toluene_mapping,
            extends=None,
        )


def test_checkpoint_interval_not_divisible_time_per_iter(
    benzene_vacuum_system,
    toluene_vacuum_system,
    benzene_to_toluene_mapping,
    vac_settings,
):
    vac_settings.output_settings.checkpoint_interval = 4 * offunit.ps
    vac_settings.simulation_settings.time_per_iteration = 2.5 * offunit.ps
    p = openmm_rfe.RelativeHybridTopologyProtocol(settings=vac_settings)

    errmsg = "does not evenly divide by the amount of time per state MCMC"

    with pytest.raises(ValueError, match=errmsg):
        p.validate(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            mapping=benzene_to_toluene_mapping,
            extends=None,
        )


@pytest.mark.parametrize("attribute", ["positions_write_frequency", "velocities_write_frequency"])
def test_pos_vel_write_frequency_not_divisible(
    benzene_vacuum_system,
    toluene_vacuum_system,
    benzene_to_toluene_mapping,
    attribute,
    vac_settings,
):
    setattr(vac_settings.output_settings, attribute, 100.1 * offunit.picosecond)

    p = openmm_rfe.RelativeHybridTopologyProtocol(settings=vac_settings)

    errmsg = f"The output settings' {attribute}"
    with pytest.raises(ValueError, match=errmsg):
        p.validate(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            mapping=benzene_to_toluene_mapping,
            extends=None,
        )


@pytest.mark.parametrize(
    "attribute", ["real_time_analysis_interval", "real_time_analysis_interval"]
)
def test_real_time_analysis_not_divisible(
    benzene_vacuum_system,
    toluene_vacuum_system,
    benzene_to_toluene_mapping,
    attribute,
    vac_settings,
):
    setattr(vac_settings.simulation_settings, attribute, 100.1 * offunit.picosecond)

    p = openmm_rfe.RelativeHybridTopologyProtocol(settings=vac_settings)

    errmsg = f"The {attribute}"
    with pytest.raises(ValueError, match=errmsg):
        p.validate(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            mapping=benzene_to_toluene_mapping,
            extends=None,
        )


def test_n_replicas_not_n_windows(
    benzene_vacuum_system,
    toluene_vacuum_system,
    benzene_to_toluene_mapping,
    vac_settings,
    tmpdir,
):
    # For PR #125 we pin such that the number of lambda windows
    # equals the numbers of replicas used - TODO: remove limitation
    vac_settings.simulation_settings.n_replicas = 13
    p = openmm_rfe.RelativeHybridTopologyProtocol(settings=vac_settings)

    errmsg = "Number of replicas in ``simulation_settings``:"

    with pytest.raises(ValueError, match=errmsg):
        p.validate(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            mapping=benzene_to_toluene_mapping,
            extends=None,
        )
