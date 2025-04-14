# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest
from openfe.protocols.restraint_utils.openmm.omm_restraints import (
    RestraintParameterState,
    HarmonicBondRestraint,
    FlatBottomBondRestraint,
    CentroidHarmonicRestraint,
    CentroidFlatBottomRestraint,
    BoreschRestraint,
    DistanceRestraintGeometry,
    BoreschRestraintGeometry,
    BoreschRestraintSettings,
    FlatBottomDistanceGeometry,

)
from openfe.protocols.restraint_utils.settings import (
    FlatBottomRestraintSettings,
    DistanceRestraintSettings
)
from ...conftest import HAS_INTERNET
from openff.units import unit
import openmm
from openmmtools.states import ThermodynamicState
from gufe import SmallMoleculeComponent
import pooch
import pathlib
import os

def test_parameter_state_default():
    param_state = RestraintParameterState()
    assert param_state.lambda_restraints is None


@pytest.mark.parametrize('suffix', [None, 'foo'])
@pytest.mark.parametrize('lambda_var', [0, 0.5, 1.0])
def test_parameter_state_suffix(suffix, lambda_var):
    param_state = RestraintParameterState(
        parameters_name_suffix=suffix, lambda_restraints=lambda_var 
    )

    if suffix is not None:
        param_name = f'lambda_restraints_{suffix}'
    else:
        param_name = 'lambda_restraints'

    assert getattr(param_state, param_name) == lambda_var
    assert len(param_state._parameters.keys()) == 1
    assert param_state._parameters[param_name] == lambda_var
    assert param_state._parameters_name_suffix == suffix


@pytest.mark.parametrize("restraint, geometry_settings", [
    pytest.param(HarmonicBondRestraint, {}, id="Harmonic"),
    pytest.param(FlatBottomBondRestraint, {"well_radius": 0.1 * unit.nanometer}, id="Flatbottom")
])
def test_single_bond_mixin(restraint, geometry_settings):
    res = restraint(
        restraint_settings=restraint._settings_cls(spring_constant=20 * unit.kilojoule_per_mole / unit.nanometer ** 2)
    )
    geometry_settings.update(
        {
            "guest_atoms": [0, 1],
            "host_atoms": [2, 3]
        }
    )
    with pytest.raises(ValueError, match="host_atoms and guest_atoms must only include a single index"):
        res._verify_geometry(
            geometry=res._geometry_cls(**geometry_settings)
        )


def test_verify_inputs():
    with pytest.raises(ValueError, match="Incorrect settings type DistanceRestraintSettings"):
        _ = FlatBottomBondRestraint(
            restraint_settings=DistanceRestraintSettings(
                spring_constant=20 * unit.kilojoule_per_mole / unit.nanometer ** 2
            )
        )

def test_verify_geometry():
    with pytest.raises(ValueError, match="Incorrect geometry class type DistanceRestraintGeometry"):
        restraint = FlatBottomBondRestraint(
            restraint_settings=FlatBottomRestraintSettings(
                spring_constant=20 * unit.kilojoule_per_mole / unit.nanometer ** 2
            )
        )
        geometry = DistanceRestraintGeometry(guest_atoms=[0], host_atoms=[1])
        restraint._verify_geometry(geometry)

POOCH_CACHE = pooch.os_cache("openfe")
zenodo_restraint_data = pooch.create(
    path=POOCH_CACHE,
    base_url="doi:10.5281/zenodo.15212342",
    registry={
        "industry_benchmark_systems.zip": "sha256:2bb5eee36e29b718b96bf6e9350e0b9957a592f6c289f77330cbb6f4311a07bd"
    }
    ,retry_if_failed=3
)

@pytest.fixture
def tyk2_protein_ligand_system():
    zenodo_restraint_data.fetch("industry_benchmark_systems.zip", processor=pooch.Unzip())
    cache_dir = pathlib.Path(
        pooch.os_cache("openfe") / "industry_benchmark_systems.zip.unzip/industry_benchmark_systems")
    with open(str(cache_dir / "jacs_set" / "tyk2" / "protein_ligand_system.xml")) as xml:
        return openmm.XmlSerializer.deserialize(xml.read())


@pytest.fixture
def tyk2_rdkit_ligand():
    zenodo_restraint_data.fetch("industry_benchmark_systems.zip", processor=pooch.Unzip())
    cache_dir = pathlib.Path(
        pooch.os_cache("openfe") / "industry_benchmark_systems.zip.unzip/industry_benchmark_systems")
    ligand = SmallMoleculeComponent.from_sdf_file(str(cache_dir / "jacs_set" / "tyk2" / "test_ligand.sdf"))
    return ligand.to_rdkit()


@pytest.mark.skipif(not os.path.exists(POOCH_CACHE) and not HAS_INTERNET, reason="Internet seems to be unavailable and test data is not cached locally.")
def test_harmonic_add_force(tyk2_protein_ligand_system):
    restraint = HarmonicBondRestraint(
        restraint_settings=DistanceRestraintSettings(
            spring_constant=20 * unit.kilojoule_per_mole / unit.nanometer ** 2
        )
    )
    state = ThermodynamicState(
        system=tyk2_protein_ligand_system
    )
    geometry = DistanceRestraintGeometry(
            host_atoms=[0],
            guest_atoms=[4706]
        )
    restraint.add_force(
        thermodynamic_state=state,
        geometry=geometry,
        controlling_parameter_name="lambda_restraints"
    )
    system = state.system
    forces = {force.__class__.__name__: force for force in system.getForces()}
    restraint_force = forces["CustomBondForce"]
    # some other random global parameter is included in this force
    assert restraint_force.getGlobalParameterName(1) == "lambda_restraints"
    assert restraint_force.getEnergyFunction() == "lambda_restraints * ((K/2)*r^2)"
    assert restraint_force.getNumBonds() == 1
    # some other random global parameter is included in this force otherwise there should be 1
    assert restraint_force.getNumGlobalParameters() == 2
    # check the restraint parameters
    host_atom, guest_atom, params = restraint_force.getBondParameters(0)
    assert host_atom == geometry.host_atoms[0]
    assert guest_atom == geometry.guest_atoms[0]
    assert params[0] == restraint.settings.spring_constant.m


@pytest.mark.skipif(not os.path.exists(POOCH_CACHE) and not HAS_INTERNET, reason="Internet seems to be unavailable and test data is not cached locally.")
def test_flatbottom_add_force(tyk2_protein_ligand_system):
    restraint = FlatBottomBondRestraint(
        restraint_settings=FlatBottomRestraintSettings(
            spring_constant=20 * unit.kilojoule_per_mole / unit.nanometer ** 2
        )
    )
    state = ThermodynamicState(
        system=tyk2_protein_ligand_system
    )
    geometry = FlatBottomDistanceGeometry(
        host_atoms=[0],
        guest_atoms=[4706],
        well_radius=1 * unit.nanometer
    )
    restraint.add_force(
        thermodynamic_state=state,
        geometry=geometry,
        controlling_parameter_name="lambda_restraints"
    )
    system = state.system
    forces = {force.__class__.__name__: force for force in system.getForces()}
    restraint_force = forces["CustomBondForce"]
    # some other random global parameter is included in this force
    assert restraint_force.getGlobalParameterName(1) == "lambda_restraints"
    assert restraint_force.getEnergyFunction() == "lambda_restraints * (step(r-r0) * (K/2)*(r-r0)^2)"
    assert restraint_force.getNumBonds() == 1
    # some other random global parameter is included in this force otherwise there should be 1
    assert restraint_force.getNumGlobalParameters() == 2
    # check the restraint parameters
    host_atom, guest_atom, params = restraint_force.getBondParameters(0)
    assert host_atom == geometry.host_atoms[0]
    assert guest_atom == geometry.guest_atoms[0]
    assert params[0] == restraint.settings.spring_constant.m
    assert params[1] == geometry.well_radius.m


@pytest.mark.skipif(not os.path.exists(POOCH_CACHE) and not HAS_INTERNET, reason="Internet seems to be unavailable and test data is not cached locally.")
def test_centriod_harmonic_add_force(tyk2_protein_ligand_system):
    restraint = CentroidHarmonicRestraint(
        restraint_settings=DistanceRestraintSettings(
            spring_constant=20 * unit.kilojoule_per_mole / unit.nanometer ** 2
        )
    )
    state = ThermodynamicState(
        system=tyk2_protein_ligand_system
    )
    geometry = DistanceRestraintGeometry(
        host_atoms=[0, 1, 2],
        guest_atoms=[4706, 4705, 4704],
    )
    restraint.add_force(
        thermodynamic_state=state,
        geometry=geometry,
        controlling_parameter_name="lambda_restraints"
    )
    system = state.system
    forces = {force.__class__.__name__: force for force in system.getForces()}
    restraint_force = forces["CustomCentroidBondForce"]
    assert restraint_force.getGlobalParameterName(1) == "lambda_restraints"
    assert restraint_force.getEnergyFunction() == "lambda_restraints * ((K/2)*distance(g1,g2)^2)"
    assert restraint_force.getNumBonds() == 1
    # some other random global parameter is included in this force otherwise there should be 1
    assert restraint_force.getNumGlobalParameters() == 2
    # check the restraint parameters
    groups, params = restraint_force.getBondParameters(0)

    assert params[0] == restraint.settings.spring_constant.m
    host_atoms = list(restraint_force.getGroupParameters(0)[0])
    guest_atoms  = list(restraint_force.getGroupParameters(1)[0])
    assert host_atoms == geometry.host_atoms
    assert guest_atoms == geometry.guest_atoms


@pytest.mark.skipif(not os.path.exists(POOCH_CACHE) and not HAS_INTERNET, reason="Internet seems to be unavailable and test data is not cached locally.")
def test_centroid_flat_bottom_add_force(tyk2_protein_ligand_system):
    restraint = CentroidFlatBottomRestraint(
        restraint_settings=FlatBottomRestraintSettings(
            spring_constant=20 * unit.kilojoule_per_mole / unit.nanometer ** 2
        )
    )
    state = ThermodynamicState(
        system=tyk2_protein_ligand_system
    )
    geometry = FlatBottomDistanceGeometry(
        host_atoms=[0, 1 , 2],
        guest_atoms=[4706, 4705, 4704],
        well_radius=1 * unit.nanometer
    )
    restraint.add_force(
        thermodynamic_state=state,
        geometry=geometry,
        controlling_parameter_name="lambda_restraints"
    )
    system = state.system
    forces = {force.__class__.__name__: force for force in system.getForces()}
    restraint_force = forces["CustomCentroidBondForce"]
    # some other random global parameter is included in this force
    assert restraint_force.getGlobalParameterName(1) == "lambda_restraints"
    assert restraint_force.getEnergyFunction() == "lambda_restraints * (step(distance(g1,g2)-r0) * (K/2)*(distance(g1,g2)-r0)^2)"
    assert restraint_force.getNumBonds() == 1
    # some other random global parameter is included in this force otherwise there should be 1
    assert restraint_force.getNumGlobalParameters() == 2
    # check the restraint parameters
    groups, params = restraint_force.getBondParameters(0)
    assert params[0] == restraint.settings.spring_constant.m
    assert params[1] == geometry.well_radius.m
    host_atoms = list(restraint_force.getGroupParameters(0)[0])
    guest_atoms = list(restraint_force.getGroupParameters(1)[0])
    assert host_atoms == geometry.host_atoms
    assert guest_atoms == geometry.guest_atoms


@pytest.mark.skipif(not os.path.exists(POOCH_CACHE) and not HAS_INTERNET, reason="Internet seems to be unavailable and test data is not cached locally.")
def test_add_boresch_force(tyk2_protein_ligand_system, tyk2_rdkit_ligand):

    restraint = BoreschRestraint(
        restraint_settings=BoreschRestraintSettings()
    )
    # create the geometry from the saved values in the sdf file
    geometry = BoreschRestraintGeometry(
        r_aA0=tyk2_rdkit_ligand.GetDoubleProp("r_aA0"),
        theta_A0=tyk2_rdkit_ligand.GetDoubleProp("theta_A0"),
        theta_B0=tyk2_rdkit_ligand.GetDoubleProp("theta_B0"),
        phi_A0=tyk2_rdkit_ligand.GetDoubleProp("phi_A0"),
        phi_B0=tyk2_rdkit_ligand.GetDoubleProp("phi_B0"),
        phi_C0=tyk2_rdkit_ligand.GetDoubleProp("phi_C0"),
        host_atoms=[tyk2_rdkit_ligand.GetIntProp(f"Host{i}") for i in range(3)],
        guest_atoms=[tyk2_rdkit_ligand.GetIntProp(f"Guest{i}") for i in range(3)],
    )
    state = ThermodynamicState(
        system=tyk2_protein_ligand_system
    )
    restraint.add_force(
        thermodynamic_state=state,
        geometry=geometry,
        controlling_parameter_name="lambda_restraints"
    )
    system = state.system
    forces = {force.__class__.__name__: force for force in system.getForces()}
    restraint_force = forces["CustomCompoundBondForce"]
    assert restraint_force.getGlobalParameterName(0) == "lambda_restraints"
    assert "lambda_restraints" in restraint_force.getEnergyFunction()
    assert restraint_force.getNumGlobalParameters() == 1
    assert restraint_force.getNumBonds() == 1
    atoms, parameters = restraint_force.getBondParameters(0)
    assert geometry.host_atoms == list(atoms[:3][::-1])
    assert geometry.guest_atoms == list(atoms[3:])
    # check all the parameters
    for i in range(restraint_force.getNumPerBondParameters()):
        per_bond_parameter = restraint_force.getPerBondParameterName(i)
        # if we have a force constant check the settings
        if per_bond_parameter[0] == "K":
            assert parameters[i] == getattr(restraint.settings, per_bond_parameter).m
        # else check the geometry
        else:
            assert parameters[i] == getattr(geometry, per_bond_parameter).m


@pytest.mark.skipif(not os.path.exists(POOCH_CACHE) and not HAS_INTERNET, reason="Internet seems to be unavailable and test data is not cached locally.")
def test_get_boresch_state_correction(tyk2_protein_ligand_system, tyk2_rdkit_ligand):
    restraint = BoreschRestraint(
        restraint_settings=BoreschRestraintSettings()
    )
    # create the geometry from the saved values in the sdf file
    geometry = BoreschRestraintGeometry(
        r_aA0=tyk2_rdkit_ligand.GetDoubleProp("r_aA0"),
        theta_A0=tyk2_rdkit_ligand.GetDoubleProp("theta_A0"),
        theta_B0=tyk2_rdkit_ligand.GetDoubleProp("theta_B0"),
        phi_A0=tyk2_rdkit_ligand.GetDoubleProp("phi_A0"),
        phi_B0=tyk2_rdkit_ligand.GetDoubleProp("phi_B0"),
        phi_C0=tyk2_rdkit_ligand.GetDoubleProp("phi_C0"),
        host_atoms=[tyk2_rdkit_ligand.GetIntProp(f"Host{i}") for i in range(3)],
        guest_atoms=[tyk2_rdkit_ligand.GetIntProp(f"Guest{i}") for i in range(3)],
    )

    state = ThermodynamicState(
        system=tyk2_protein_ligand_system
    )
    correction = restraint.get_standard_state_correction(
        thermodynamic_state=state,
        geometry=geometry
    )
    assert pytest.approx(correction.to(unit.kilocalorie_per_mole).m) == -6.051980241975009