# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest

from ..conftest import atom_mapping_basic_test_files, T4_protein_component

from gufe import ChemicalSystem, SolventComponent
from openfe.setup.alchemical_network_planner import RHFEAlchemicalNetworkPlanner, RBFEAlchemicalNetworkPlanner

def test_rhfe_alchemical_network_planner_init():
    
    alchem_planner = RHFEAlchemicalNetworkPlanner()

def test_rbfe_alchemical_network_planner_init():
    
    alchem_planner = RBFEAlchemicalNetworkPlanner()
    
def test_rbfe_alchemical_network_planner_call(atom_mapping_basic_test_files, T4_protein_component):
    alchem_planner = RBFEAlchemicalNetworkPlanner()
    alchem_network = alchem_planner(ligands=atom_mapping_basic_test_files.values(), solvent=SolventComponent(), receptor=T4_protein_component)
    
def test_rhfe_alchemical_network_planner_call(atom_mapping_basic_test_files, T4_protein_component):
    alchem_planner = RHFEAlchemicalNetworkPlanner()
    alchem_network = alchem_planner(ligands=atom_mapping_basic_test_files.values(), solvent=SolventComponent())