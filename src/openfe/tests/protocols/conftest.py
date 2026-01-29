# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import gzip
import pathlib
from importlib import resources
from typing import Optional

import MDAnalysis as mda
import openmm
import pooch
import pytest
from gufe.tests.test_tokenization import GufeTokenizableTestsMixin
from openff.units import Quantity, unit
from openff.units.openmm import from_openmm
from openmm import Platform
from rdkit import Chem
from rdkit.Geometry import Point3D

import openfe
from openfe.data._registry import (
    POOCH_CACHE,
    zenodo_industry_benchmark_systems,
    zenodo_rfe_simulation_nc,
    zenodo_t4_lysozyme_traj,
)


@pytest.fixture
def available_platforms() -> set[str]:
    return {
        Platform.getPlatform(i).getName()
        for i in range(Platform.getNumPlatforms())
    }  # fmt: skip


@pytest.fixture
def benzene_vacuum_system(benzene_modifications):
    return openfe.ChemicalSystem(
        {"ligand": benzene_modifications["benzene"]},
    )


@pytest.fixture(scope="session")
def benzene_system(benzene_modifications):
    return openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications["benzene"],
            "solvent": openfe.SolventComponent(
                positive_ion="Na",
                negative_ion="Cl",
                ion_concentration=0.15 * unit.molar,
            ),
        },
    )


@pytest.fixture
def benzene_complex_system(benzene_modifications, T4_protein_component):
    return openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications["benzene"],
            "solvent": openfe.SolventComponent(
                positive_ion="Na",
                negative_ion="Cl",
                ion_concentration=0.15 * unit.molar,
            ),
            "protein": T4_protein_component,
        }
    )


@pytest.fixture
def toluene_vacuum_system(benzene_modifications):
    return openfe.ChemicalSystem(
        {"ligand": benzene_modifications["toluene"]},
    )


@pytest.fixture(scope="session")
def toluene_system(benzene_modifications):
    return openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications["toluene"],
            "solvent": openfe.SolventComponent(
                positive_ion="Na", negative_ion="Cl", ion_concentration=0.15 * unit.molar
            ),
        },
    )


@pytest.fixture
def toluene_complex_system(benzene_modifications, T4_protein_component):
    return openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications["toluene"],
            "solvent": openfe.SolventComponent(
                positive_ion="Na", negative_ion="Cl", ion_concentration=0.15 * unit.molar
            ),
            "protein": T4_protein_component,
        }
    )


@pytest.fixture(scope="session")
def benzene_to_toluene_mapping(benzene_modifications):
    mapper = openfe.setup.LomapAtomMapper(element_change=False)

    molA = benzene_modifications["benzene"]
    molB = benzene_modifications["toluene"]

    return next(mapper.suggest_mappings(molA, molB))


@pytest.fixture
def benzene_charges():
    files = {}
    with resources.as_file(resources.files("openfe.tests.data.openmm_rfe")) as d:
        fn = str(d / "charged_benzenes.sdf")
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)
        for rdmol in supp:
            files[rdmol.GetProp("_Name")] = openfe.SmallMoleculeComponent(rdmol)
    return files


@pytest.fixture
def benzene_to_benzoic_mapping(benzene_charges):
    mapper = openfe.setup.LomapAtomMapper(element_change=False)
    molA = benzene_charges["benzene"]
    molB = benzene_charges["benzoic_acid"]
    return next(mapper.suggest_mappings(molA, molB))


@pytest.fixture
def benzoic_to_benzene_mapping(benzene_charges):
    mapper = openfe.setup.LomapAtomMapper(element_change=False)
    molA = benzene_charges["benzoic_acid"]
    molB = benzene_charges["benzene"]
    return next(mapper.suggest_mappings(molA, molB))


@pytest.fixture
def benzene_to_aniline_mapping(benzene_charges):
    mapper = openfe.setup.LomapAtomMapper(element_change=False)
    molA = benzene_charges["benzene"]
    molB = benzene_charges["aniline"]
    return next(mapper.suggest_mappings(molA, molB))


@pytest.fixture
def aniline_to_benzene_mapping(benzene_charges):
    mapper = openfe.setup.LomapAtomMapper(element_change=False)
    molA = benzene_charges["aniline"]
    molB = benzene_charges["benzene"]
    return next(mapper.suggest_mappings(molA, molB))


@pytest.fixture
def aniline_to_benzoic_mapping(benzene_charges):
    mapper = openfe.setup.LomapAtomMapper(element_change=False)
    molA = benzene_charges["aniline"]
    molB = benzene_charges["benzoic_acid"]
    return next(mapper.suggest_mappings(molA, molB))


@pytest.fixture
def benzene_many_solv_system(benzene_modifications):
    rdmol_phenol = benzene_modifications["phenol"].to_rdkit()
    rdmol_benzo = benzene_modifications["benzonitrile"].to_rdkit()

    conf_phenol = rdmol_phenol.GetConformer()
    conf_benzo = rdmol_benzo.GetConformer()

    for atm in range(rdmol_phenol.GetNumAtoms()):
        x, y, z = conf_phenol.GetAtomPosition(atm)
        conf_phenol.SetAtomPosition(atm, Point3D(x + 30, y, z))

    for atm in range(rdmol_benzo.GetNumAtoms()):
        x, y, z = conf_benzo.GetAtomPosition(atm)
        conf_benzo.SetAtomPosition(atm, Point3D(x, y + 30, z))

    phenol = openfe.SmallMoleculeComponent.from_rdkit(rdmol_phenol, name="phenol")

    benzo = openfe.SmallMoleculeComponent.from_rdkit(rdmol_benzo, name="benzonitrile")

    return openfe.ChemicalSystem(
        {
            "whatligand": benzene_modifications["benzene"],
            "foo": phenol,
            "bar": benzo,
            "solvent": openfe.SolventComponent(),
        },
    )


@pytest.fixture
def toluene_many_solv_system(benzene_modifications):
    rdmol_phenol = benzene_modifications["phenol"].to_rdkit()
    rdmol_benzo = benzene_modifications["benzonitrile"].to_rdkit()

    conf_phenol = rdmol_phenol.GetConformer()
    conf_benzo = rdmol_benzo.GetConformer()

    for atm in range(rdmol_phenol.GetNumAtoms()):
        x, y, z = conf_phenol.GetAtomPosition(atm)
        conf_phenol.SetAtomPosition(atm, Point3D(x + 30, y, z))

    for atm in range(rdmol_benzo.GetNumAtoms()):
        x, y, z = conf_benzo.GetAtomPosition(atm)
        conf_benzo.SetAtomPosition(atm, Point3D(x, y + 30, z))

    phenol = openfe.SmallMoleculeComponent.from_rdkit(rdmol_phenol, name="phenol")

    benzo = openfe.SmallMoleculeComponent.from_rdkit(rdmol_benzo, name="benzonitrile")
    return openfe.ChemicalSystem(
        {
            "whatligand": benzene_modifications["toluene"],
            "foo": phenol,
            "bar": benzo,
            "solvent": openfe.SolventComponent(),
        },
    )


@pytest.fixture
def rfe_transformation_json() -> str:
    """string of a RFE results similar to quickrun

    generated with gen-serialized-results.py
    """
    d = resources.files("openfe.tests.data.openmm_rfe")

    with gzip.open((d / "RHFEProtocol_json_results.gz").as_posix(), "r") as f:  # type: ignore
        return f.read().decode()  # type: ignore


@pytest.fixture
def afe_solv_transformation_json() -> str:
    """
    string of a Absolute Solvation result (CN in water) generated by quickrun

    generated with gen-serialized-results.py
    """
    d = resources.files("openfe.tests.data.openmm_afe")
    fname = "AHFEProtocol_json_results.gz"

    with gzip.open((d / fname).as_posix(), "r") as f:  # type: ignore
        return f.read().decode()  # type: ignore


@pytest.fixture
def abfe_transformation_json_path() -> str:
    """
    Path to an Absolute Binding result (tyk2 complex) generated by quickrun

    generated with gen-serialized-results.py
    """
    d = resources.files("openfe.tests.data.openmm_afe")
    fname = "ABFEProtocol_json_results.json.gz"

    return str(d / fname)


@pytest.fixture
def md_json() -> str:
    """
    string of a MD result (TYK ligand lig_ejm_31  in water) generated by quickrun

    generated with gen-serialized-results.py
    """
    d = resources.files("openfe.tests.data.openmm_md")
    fname = "MDProtocol_json_results.gz"

    with gzip.open((d / fname).as_posix(), "r") as f:  # type: ignore
        return f.read().decode()  # type: ignore


@pytest.fixture
def septop_json() -> str:
    """
    Path to a SepTop result (hif2a)

    generated with gen-serialized-results.py
    """
    d = resources.files("openfe.tests.data.openmm_septop")
    fname = "SepTopProtocol_json_results.gz"

    with gzip.open((d / fname).as_posix(), "r") as f:  # type: ignore
        return f.read().decode()  # type: ignore


pooch_industry_benchmark_systems = pooch.create(
    path=POOCH_CACHE,
    base_url=zenodo_industry_benchmark_systems["base_url"],
    registry={
        zenodo_industry_benchmark_systems["fname"]: zenodo_industry_benchmark_systems["known_hash"]
    },
)


@pytest.fixture
def industry_benchmark_files():
    pooch_industry_benchmark_systems.fetch(
        "industry_benchmark_systems.zip", processor=pooch.Unzip()
    )
    cache_dir = pathlib.Path(
        POOCH_CACHE / "industry_benchmark_systems.zip.unzip/industry_benchmark_systems"
    )
    return cache_dir


pooch_t4_lysozyme = pooch.create(
    path=POOCH_CACHE,
    base_url=zenodo_t4_lysozyme_traj["base_url"],
    registry={zenodo_t4_lysozyme_traj["fname"]: zenodo_t4_lysozyme_traj["known_hash"]},
)


@pytest.fixture
def t4_lysozyme_trajectory_universe():
    pooch_t4_lysozyme.fetch("t4_lysozyme_trajectory.zip", processor=pooch.Unzip())
    cache_dir = pathlib.Path(
        POOCH_CACHE / "t4_lysozyme_trajectory.zip.unzip/t4_lysozyme_trajectory"
    )
    universe = mda.Universe(
        str(cache_dir / "t4_toluene_complex.pdb"),
        str(cache_dir / "t4_toluene_complex.xtc"),
    )
    return universe


@pytest.fixture
def simulation_nc():
    return pooch.retrieve(
        url=zenodo_rfe_simulation_nc["base_url"] + zenodo_rfe_simulation_nc["fname"],
        known_hash=zenodo_rfe_simulation_nc["known_hash"],
        path=POOCH_CACHE,
    )


@pytest.fixture
def get_available_openmm_platforms() -> set[str]:
    """
    OpenMM Platforms that are available and functional on system
    """
    import openmm
    from openmm import Platform

    # Get platforms that openmm was built with
    platforms = {Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())}

    # Now check if we can actually use the platforms
    working_platforms = set()
    for platform in platforms:
        system = openmm.System()
        system.addParticle(1.0)
        integrator = openmm.VerletIntegrator(0.001)
        try:
            context = openmm.Context(system, integrator, Platform.getPlatformByName(platform))
            working_platforms.add(platform)
            del context
        except openmm.OpenMMException:
            continue
        finally:
            del system, integrator

    return working_platforms


class ModGufeTokenizableTestsMixin(GufeTokenizableTestsMixin):
    """
    A modified gufe tokenizable tests mixin which allows
    for repr to be lazily evaluated.
    """

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


def compute_energy(
    system: openmm.System,
    positions: openmm.unit.Quantity,
    box_vectors: Optional[openmm.unit.Quantity],
    context_params: Optional[dict[str, float]] = None,
    platform=None,
) -> Quantity:
    """
    Computes the potential energy of a system at a given set of positions.

    Parameters
    ----------
    system: openmm.System
      The system to compute the energy of.
    positions: openmm.unit.Quantity
      The positions to compute the energy at.
    box_vectors: Optional[openmm.unit.Quantity]
      The box vectors to use if any.
    context_params: Optional[dict[str, float]]
      Any global context parameters to set.
    platform: str
      The platform to use.

    Returns
    -------
    potential : openff.units.Quantity
        The computed potential energy in openff unit.
    """
    context_params = context_params if context_params is not None else {}

    integrator = openmm.VerletIntegrator(0.0001 * openmm.unit.femtoseconds)

    if platform is None:
        context = openmm.Context(system, integrator)
    if platform is not None:
        context = openmm.Context(system, integrator, platform)

    for key, value in context_params.items():
        context.setParameter(key, value)

    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)

    state = context.getState(getEnergy=True)
    potential = state.getPotentialEnergy()
    del context, integrator, state
    return from_openmm(potential)
