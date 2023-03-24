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

    chemSys_generator = EasyChemicalSystemGenerator(do_vacuum=True)
    
    chemSys_generator = EasyChemicalSystemGenerator(solvent=SolventComponent())
    
    chemSys_generator = EasyChemicalSystemGenerator(
        solvent=SolventComponent(), protein=T4_protein_component
    )
    
    chemSys_generator = EasyChemicalSystemGenerator(
        solvent=SolventComponent(), protein=T4_protein_component, do_vacuum=True
    )

    with pytest.raises(ValueError, match='you need to provide any system generation information in the constructor'):        
        chemSys_generator = EasyChemicalSystemGenerator()


def test_build_vacuum_chemical_system(ethane):
    chemSys_generator = EasyChemicalSystemGenerator(do_vacuum=True)
    chemSys = next(chemSys_generator(ethane))

    assert chemSys is not None
    assert isinstance(chemSys, ChemicalSystem)
    assert not proteinC_in_chem_sys(chemSys)
    assert not solventC_in_chem_sys(chemSys)
    assert ligandC_in_chem_sys(chemSys)


def test_build_solvent_chemical_system(ethane):
    chemSys_generator = EasyChemicalSystemGenerator(solvent=SolventComponent())
    chemSys = next(chemSys_generator(ethane))

    assert chemSys is not None
    assert isinstance(chemSys, ChemicalSystem)
    assert not proteinC_in_chem_sys(chemSys)
    assert solventC_in_chem_sys(chemSys)
    assert ligandC_in_chem_sys(chemSys)

def test_build_protein_chemical_system(ethane, T4_protein_component):
    chemSys_generator = EasyChemicalSystemGenerator(protein=T4_protein_component)
    chemSys = next(chemSys_generator(ethane))

    assert chemSys is not None
    assert isinstance(chemSys, ChemicalSystem)
    assert proteinC_in_chem_sys(chemSys)
    assert not solventC_in_chem_sys(chemSys)
    assert ligandC_in_chem_sys(chemSys)

def test_build_hydr_scenario_chemical_systems(ethane):
    chemSys_generator = EasyChemicalSystemGenerator(
        do_vacuum=True, solvent=SolventComponent()
    )
    chemSys_gen = chemSys_generator(ethane)
    chemSyss = [chemSys for chemSys in chemSys_gen]

    assert len(chemSyss) == 2
    assert all([isinstance(chemSys, ChemicalSystem) for chemSys in chemSyss])
    assert [proteinC_in_chem_sys(chemSys) for chemSys in chemSyss] == [False, False]
    assert [solventC_in_chem_sys(chemSys) for chemSys in chemSyss] == [False, True]
    assert [ligandC_in_chem_sys(chemSys) for chemSys in chemSyss] == [True, True]

def test_build_binding_scenario_chemical_systems(ethane, T4_protein_component):
    chemSys_generator = EasyChemicalSystemGenerator(
        solvent=SolventComponent(), protein=T4_protein_component
    )
    chemSys_gen = chemSys_generator(ethane)
    chemSyss = [chemSys for chemSys in chemSys_gen]

    assert len(chemSyss) == 2
    assert all([isinstance(chemSys, ChemicalSystem) for chemSys in chemSyss])
    print(chemSyss)
    assert [proteinC_in_chem_sys(chemSys) for chemSys in chemSyss] == [False, True]
    assert [solventC_in_chem_sys(chemSys) for chemSys in chemSyss] == [True, True]
    assert [ligandC_in_chem_sys(chemSys) for chemSys in chemSyss] == [True, True]


def test_build_hbinding_scenario_chemical_systems(ethane, T4_protein_component):
    chemSys_generator = EasyChemicalSystemGenerator(
        do_vacuum=True, solvent=SolventComponent(), protein=T4_protein_component
    )
    chemSys_gen = chemSys_generator(ethane)
    chemSyss = [chemSys for chemSys in chemSys_gen]

    assert len(chemSyss) == 3
    assert all([isinstance(chemSys, ChemicalSystem) for chemSys in chemSyss])
    assert [proteinC_in_chem_sys(chemSys) for chemSys in chemSyss] == [False, False, True]
    assert [solventC_in_chem_sys(chemSys) for chemSys in chemSyss] == [False, True, True]
    assert [ligandC_in_chem_sys(chemSys) for chemSys in chemSyss] == [True, True, True]