# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest

from gufe import ChemicalSystem
from openfe.setup.transformation_factory.easy_transformation_factory import (
    AbstractEasyTransformationFactory,
    RFETransformationFactory
    )

from openfe.setup.chemicalsystem_generator.easy_chemicalsystem_generator import (
    EasyChemicalSystemGenerator,
)
from openfe.protocols.openmm_rbfe import RelativeLigandProtocol
from ..conftest import T4_protein_component
from gufe import SolventComponent

# Todo: this is clunky!
from ..conftest import atom_mapping_basic_test_files
from ..test_network_planning import toluene_vs_others, minimal_spanning_network


def test_easy_transformation_factory_init():
    def_settings = RelativeLigandProtocol._default_settings()
    protocol = RelativeLigandProtocol(def_settings)
    system_generator = EasyChemicalSystemGenerator(do_vacuum=True, solvent=SolventComponent())
    
    trans_factory = RFETransformationFactory(protocol=protocol,
                                                  chemical_system_generator=system_generator)

def test_easy_transformation_factory_call_rhfe(minimal_spanning_network):
    def_settings = RelativeLigandProtocol._default_settings()
        
    protocol = RelativeLigandProtocol(def_settings)
    system_generator = EasyChemicalSystemGenerator(solvent=SolventComponent(), do_vacuum=True)
    
    trans_factory = RFETransformationFactory(protocol=protocol,
                                             chemical_system_generator=system_generator)
    
    alchem_net = trans_factory(name="test", 
                               alchemical_network_edges=minimal_spanning_network.edges)
    
    print(alchem_net)
    

def test_easy_transformation_factory_call_rbfe(minimal_spanning_network, T4_protein_component):
    def_settings = RelativeLigandProtocol._default_settings()
        
    protocol = RelativeLigandProtocol(def_settings)
    system_generator = EasyChemicalSystemGenerator(solvent=SolventComponent(), protein=T4_protein_component)
    
    trans_factory = RFETransformationFactory(protocol=protocol,
                                             chemical_system_generator=system_generator)
    
    alchem_net = trans_factory(name="test", 
                               alchemical_network_edges=minimal_spanning_network.edges)
    
    print(alchem_net)
    