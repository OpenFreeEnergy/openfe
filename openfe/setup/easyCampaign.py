# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Iterable, Callable, FrozenSet, Tuple

from rdkit import Chem

from gufe import Protocol
from gufe.mapping.atom_mapper import AtomMapper

from . import LomapAtomMapper, Transformation, SmallMoleculeComponent, ProteinComponent, LigandAtomMapping, ChemicalSystem
from .atom_mapping.lomap_scorers import default_lomap_score
from .ligand_network_planning import minimal_spanning_graph
from ..protocols.openmm_rbfe.equil_rbfe_methods import RelativeLigandTransform

"""
    This is a draft!
"""

def load_files(ligand_sdf_paths:Iterable[str], receptor_pdb_path:str)-> Tuple[Iterable[SmallMoleculeComponent], ProteinComponent]:
    small_components = []
    for ligand_sdf in ligand_sdf_paths:
        suppl = Chem.SDSupplier(ligand_sdf)
        small_components.extend([SmallMoleculeComponent(rdkit=mol) for mol in suppl])
    
    receptor_component = ProteinComponent.from_pdbfile(pdbfile=receptor_pdb_path)
    
    return small_components, receptor_component
    
def build_transformations(edges:FrozenSet[LigandAtomMapping], protein_component:ProteinComponent, protocol:Protocol)->Iterable[Transformation]:
    
    transformations = []
    for edge in edges:
        #build Chemical System
        stateA = ChemicalSystem(components={"compA": edge.componentA, 
                                            "receptor":protein_component})

        stateB = ChemicalSystem(components={"compA": edge.componentA, 
                                            "receptor":protein_component})
        
        #build Transformation
        edge_transformation = Transformation(stateA=stateA, stateB=stateB, mapping=edge, protocol=protocol)
        transformations.append(edge_transformation)

    return transformations

def plan_easy_campaign(
                        receptor_pdb_path:str, ligand_sdf_paths: Iterable[str],
                        mapper:AtomMapper=LomapAtomMapper,
                        mapping_scorers:Iterable[Callable]=[default_lomap_score],
                        networker:Callable=minimal_spanning_graph,
                        protocol:Protocol =RelativeLigandTransform) -> Iterable[Transformation]:
    """
        This draft should realize an easy campaign builder for a simple use case one receptor multiple ligands RBFE only.
    """
    # Implement:
    #Read files to Components - debatable if not to start from components
    small_components, receptor_component = load_files(ligand_sdf_paths=ligand_sdf_paths,
                                                      receptor_pdb_path=receptor_pdb_path)
           
    #throw into Networker
    network = networker(ligands=small_components,
                        mappers=[mapper],
                        mapping_scorers=mapping_scorers)
    
    
    #build transformations
    transformations = build_transformations(paths=network.edges, protein_component=receptor_component, protocol=RelativeLigandTransform)
   
    # Here we should add a more complex structure that wraps transformations to deal with network strategies Campaign?
    # Campaign could be a class containing transformations and strategies for the chemical systems and the tasks.
    
    return transformations
    