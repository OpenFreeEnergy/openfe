from typing import Iterable, FrozenSet, Tuple

from rdkit import Chem

from gufe import Protocol

from .. import Transformation, SmallMoleculeComponent, ProteinComponent, LigandAtomMapping, ChemicalSystem



def load_files(ligand_sdf_paths:Iterable[str], receptor_pdb_path:str)-> Tuple[Iterable[SmallMoleculeComponent], ProteinComponent]:
    small_components = []
    for ligand_sdf in ligand_sdf_paths:
        suppl = Chem.SDSupplier(ligand_sdf)
        small_components.extend([SmallMoleculeComponent(rdkit=mol) for mol in suppl])
    
    receptor_component = ProteinComponent.from_pdbfile(pdbfile=receptor_pdb_path)
    
    return small_components, receptor_component
    
def build_transformations_from_edges(edges:FrozenSet[LigandAtomMapping], protein_component:ProteinComponent, protocol:Protocol)->Iterable[Transformation]:
    
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
