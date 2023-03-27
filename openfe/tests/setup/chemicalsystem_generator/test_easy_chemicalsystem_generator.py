# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from rdkit import Chem
import pytest

from gufe import ChemicalSystem
from openfe.setup.chemicalsystem_generator.easy_chemicalsystem_generator import (
    EasyChemicalSystemGenerator,
)


from ..conftest import T4_protein_component
from gufe import SolventComponent
from .component_checks import proteinC_in_chem_sys, solventC_in_chem_sys, ligandC_in_chem_sys

def test_easy_chemical_system_generator_init(T4_protein_component):

    chem_sys_generator = EasyChemicalSystemGenerator(do_vacuum=True)
    
    chem_sys_generator = EasyChemicalSystemGenerator(solvent=SolventComponent())
    
    chem_sys_generator = EasyChemicalSystemGenerator(
        solvent=SolventComponent(), protein=T4_protein_component
    )
    
    chem_sys_generator = EasyChemicalSystemGenerator(
        solvent=SolventComponent(), protein=T4_protein_component, do_vacuum=True
    )

    with pytest.raises(ValueError, match='Chemical system generator is unable to generate any chemical systems with neither protein nor solvent nor do_vacuum'):
        chem_sys_generator = EasyChemicalSystemGenerator()


def test_build_vacuum_chemical_system(ethane):
    chem_sys_generator = EasyChemicalSystemGenerator(do_vacuum=True)
    chem_sys = next(chem_sys_generator(ethane))

    assert chem_sys is not None
    assert isinstance(chem_sys, ChemicalSystem)
    assert not proteinC_in_chem_sys(chem_sys)
    assert not solventC_in_chem_sys(chem_sys)
    assert ligandC_in_chem_sys(chem_sys)


def test_build_solvent_chemical_system(ethane):
    chem_sys_generator = EasyChemicalSystemGenerator(solvent=SolventComponent())
    chem_sys = next(chem_sys_generator(ethane))

    assert chem_sys is not None
    assert isinstance(chem_sys, ChemicalSystem)
    assert not proteinC_in_chem_sys(chem_sys)
    assert solventC_in_chem_sys(chem_sys)
    assert ligandC_in_chem_sys(chem_sys)

def test_build_protein_chemical_system(ethane, T4_protein_component):
    chem_sys_generator = EasyChemicalSystemGenerator(protein=T4_protein_component)
    chem_sys = next(chem_sys_generator(ethane))

    assert chem_sys is not None
    assert isinstance(chem_sys, ChemicalSystem)
    assert proteinC_in_chem_sys(chem_sys)
    assert not solventC_in_chem_sys(chem_sys)
    assert ligandC_in_chem_sys(chem_sys)

def test_build_hydr_scenario_chemical_systems(ethane):
    chem_sys_generator = EasyChemicalSystemGenerator(
        do_vacuum=True, solvent=SolventComponent()
    )
    chem_sys_gen = chem_sys_generator(ethane)
    chem_syss = [chem_sys for chem_sys in chem_sys_gen]

    assert len(chem_syss) == 2
    assert all([isinstance(chem_sys, ChemicalSystem) for chem_sys in chem_syss])
    assert [proteinC_in_chem_sys(chem_sys) for chem_sys in chem_syss] == [False, False]
    assert [solventC_in_chem_sys(chem_sys) for chem_sys in chem_syss] == [False, True]
    assert [ligandC_in_chem_sys(chem_sys) for chem_sys in chem_syss] == [True, True]

def test_build_binding_scenario_chemical_systems(ethane, T4_protein_component):
    chem_sys_generator = EasyChemicalSystemGenerator(
        solvent=SolventComponent(), protein=T4_protein_component
    )
    chem_sys_gen = chem_sys_generator(ethane)
    chem_syss = [chem_sys for chem_sys in chem_sys_gen]

    assert len(chem_syss) == 2
    assert all([isinstance(chem_sys, ChemicalSystem) for chem_sys in chem_syss])
    print(chem_syss)
    assert [proteinC_in_chem_sys(chem_sys) for chem_sys in chem_syss] == [False, True]
    assert [solventC_in_chem_sys(chem_sys) for chem_sys in chem_syss] == [True, True]
    assert [ligandC_in_chem_sys(chem_sys) for chem_sys in chem_syss] == [True, True]


def test_build_hbinding_scenario_chemical_systems(ethane, T4_protein_component):
    chem_sys_generator = EasyChemicalSystemGenerator(
        do_vacuum=True, solvent=SolventComponent(), protein=T4_protein_component
    )
    chem_sys_gen = chem_sys_generator(ethane)
    chem_syss = [chem_sys for chem_sys in chem_sys_gen]

    assert len(chem_syss) == 3
    assert all([isinstance(chem_sys, ChemicalSystem) for chem_sys in chem_syss])
    assert [proteinC_in_chem_sys(chem_sys) for chem_sys in chem_syss] == [False, False, True]
    assert [solventC_in_chem_sys(chem_sys) for chem_sys in chem_syss] == [False, True, True]
    assert [ligandC_in_chem_sys(chem_sys) for chem_sys in chem_syss] == [True, True, True]