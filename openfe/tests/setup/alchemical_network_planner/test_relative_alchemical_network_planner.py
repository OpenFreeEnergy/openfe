# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest

from ...conftest import atom_mapping_basic_test_files, T4_protein_component

from gufe import SolventComponent, AlchemicalNetwork
from openfe.setup.alchemical_network_planner import RHFEAlchemicalNetworkPlanner, RBFEAlchemicalNetworkPlanner
from .edge_types import r_complex_edge, r_solvent_edge, r_vacuum_edge
import openfe


def test_rhfe_alchemical_network_planner_init():
    alchem_planner = RHFEAlchemicalNetworkPlanner()
    
    assert alchem_planner.name == "easy_rhfe"


def test_rbfe_alchemical_network_planner_init():
    alchem_planner = RBFEAlchemicalNetworkPlanner()
    
    assert alchem_planner.name == "easy_rbfe"


def test_rbfe_alchemical_network_planner_call(atom_mapping_basic_test_files, T4_protein_component):
    alchem_planner = RBFEAlchemicalNetworkPlanner(
        # these aren't default settings, but the test ligands aren't aligned
        mappers=[openfe.LomapAtomMapper(
            time=20,
            element_change=False,
            max3d=1,
            shift=True,
        )]
    )
    alchem_network = alchem_planner(
        ligands=atom_mapping_basic_test_files.values(),
        solvent=SolventComponent(),
        protein=T4_protein_component,
    )
    
    assert isinstance(alchem_network, AlchemicalNetwork)
    
    edges = alchem_network.edges
    assert len(edges) == 14 # we build 2envs*8ligands-2startLigands = 14 relative edges. 

    assert sum([r_complex_edge(e) for e in edges]) == 7 # half of the transformations should be complex (they always are)!
    assert sum([r_solvent_edge(e) for e in edges]) == 7 # half of the transformations should be solvent!
    assert sum([r_vacuum_edge(e) for e in edges]) == 0 # no vacuum here!


def test_rhfe_alchemical_network_planner_call_multigraph(atom_mapping_basic_test_files):
    alchem_planner = RHFEAlchemicalNetworkPlanner(
        mappers=[openfe.LomapAtomMapper(
            time=20,
            element_change=False,
            max3d=1,
            shift=True,
        )]
    )

    ligand_network = alchem_planner._construct_ligand_network(atom_mapping_basic_test_files.values())
    ligand_network_edges = list(ligand_network.edges)
    ligand_network_edges.extend(list(ligand_network.edges))

    chemical_system_generator = alchem_planner._chemical_system_generator_type(
            solvent=SolventComponent()
        )

    with pytest.raises(ValueError, match="There were multiple transformations with the same edge label! This might lead to overwritting your files."):
        alchem_network = alchem_planner._build_transformations(
            ligand_network_edges=ligand_network_edges,
            protocol=alchem_planner.transformation_protocol,
            chemical_system_generator=chemical_system_generator)


def test_rhfe_alchemical_network_planner_call(atom_mapping_basic_test_files):
    alchem_planner = RHFEAlchemicalNetworkPlanner(
        mappers=[openfe.LomapAtomMapper(
            time=20,
            element_change=False,
            max3d=1,
            shift=True,
        )]
    )
    alchem_network = alchem_planner(ligands=atom_mapping_basic_test_files.values(), solvent=SolventComponent())
    
    assert isinstance(alchem_network, AlchemicalNetwork)
    
    edges = alchem_network.edges
    assert len(edges) == 14 # we build 2envs*8ligands-2startLigands = 14 relative edges.

    assert sum([r_complex_edge(e) for e in edges]) == 0 # no complex!
    assert sum([r_solvent_edge(e) for e in edges]) == 7 # half of the transformations should be solvent!
    assert sum([r_vacuum_edge(e) for e in edges]) == 7 # half of the transformations should be vacuum!
