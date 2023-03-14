# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from rdkit import Chem
import pytest

from gufe import ChemicalSystem
from openfe.setup.chemicalsystem_generator.easy_chemicalsystem_generator import (
    Easy_chemicalsystem_generator,
)


from ..conftest import T4_protein_component
from gufe import SolventComponent


def test_easy_chemical_system_generator_init(T4_protein_component):

    chemSys_generator = Easy_chemicalsystem_generator(do_vacuum=True)
    chemSys_generator = Easy_chemicalsystem_generator(solvent=SolventComponent())
    chemSys_generator = Easy_chemicalsystem_generator(
        solvent=SolventComponent(), protein=T4_protein_component
    )
    chemSys_generator = Easy_chemicalsystem_generator(
        solvent=SolventComponent(), protein=T4_protein_component, do_vacuum=True
    )

    try:
        chemSys_generator = Easy_chemicalsystem_generator()
    except ValueError:
        pass


def test_build_vacuum_chemical_system(ethane):
    chemSys_generator = Easy_chemicalsystem_generator(do_vacuum=True)
    chemSys = next(chemSys_generator(ethane))

    assert chemSys is not None
    assert isinstance(chemSys, ChemicalSystem)


def test_build_solvent_chemical_system(ethane):
    chemSys_generator = Easy_chemicalsystem_generator(solvent=SolventComponent())
    chemSys = next(chemSys_generator(ethane))

    assert chemSys is not None
    assert isinstance(chemSys, ChemicalSystem)


def test_build_hydr_scenario_chemical_systems(ethane):
    chemSys_generator = Easy_chemicalsystem_generator(
        do_vacuum=True, solvent=SolventComponent()
    )
    chemSys_gen = chemSys_generator(ethane)
    chemSyss = [chemSys for chemSys in chemSys_gen]

    assert len(chemSyss) == 2
    assert all([isinstance(chemSys, ChemicalSystem) for chemSys in chemSyss])


def test_build_binding_scenario_chemical_systems(ethane, T4_protein_component):
    chemSys_generator = Easy_chemicalsystem_generator(
        solvent=SolventComponent(), protein=T4_protein_component
    )
    chemSys_gen = chemSys_generator(ethane)
    chemSyss = [chemSys for chemSys in chemSys_gen]

    assert len(chemSyss) == 2
    assert all([isinstance(chemSys, ChemicalSystem) for chemSys in chemSyss])


def test_build_hbinding_scenario_chemical_systems(ethane, T4_protein_component):
    chemSys_generator = Easy_chemicalsystem_generator(
        do_vacuum=True, solvent=SolventComponent(), protein=T4_protein_component
    )
    chemSys_gen = chemSys_generator(ethane)
    chemSyss = [chemSys for chemSys in chemSys_gen]

    assert len(chemSyss) == 3
    assert all([isinstance(chemSys, ChemicalSystem) for chemSys in chemSyss])
