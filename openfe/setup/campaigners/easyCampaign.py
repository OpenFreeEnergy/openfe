# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Iterable, Callable

from rdkit import Chem

from gufe import Protocol
from gufe.mapping.atom_mapper import AtomMapper

from .. import LomapAtomMapper, Transformation
from ..atom_mapping.lomap_scorers import default_lomap_score
from ..ligand_network_planning import minimal_spanning_graph
from ...protocols.openmm_rbfe.equil_rbfe_methods import RelativeLigandTransform

from .building_blocks import load_files, build_transformations_from_edges

"""
    This is a draft!
"""

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
    transformations = build_transformations_from_edges(paths=network.edges, protein_component=receptor_component, protocol=RelativeLigandTransform)
   
    # Here we should add a more complex structure that wraps transformations to deal with network strategies Campaign?
    # Campaign could be a class containing transformations and strategies for the chemical systems and the tasks.
    
    return transformations
    