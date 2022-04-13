import networkx as nx
from rdkit import Chem

from typing import Dict

def hydrogen_mapping(
    heavy_mapping: Dict[int, int],
    molA: Chem.Mol,
    molB: Chem.Mol
) -> Dict[int, int]:
    """Create the hydrogen mappings for a given heavy atom mapping.

    Parameters
    ----------
    heavy_mapping: Dict[int, int]
        mapping for heavy atoms. If a heavy atom is included on either side
        of the mapping, it should be included here (even if the atom it maps
        to is a hydrogen).
    molA, molB : Chem.Mol
        RDKit representation molecules for this mapping pair

    Return
    ------
    Dict[int, int]
        hydrogen-to-hydrogen mappings
    """
    hyd_mapping = {}
    graphA = nx.Graph([(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                       for b in molA.GetBonds()])
    graphB = nx.Graph([(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                       for b in molB.GetBonds()])
    for idxA, idxB in mapping.items():
        hydA = {at for at in graphA.adj[idxA]
                if molA.GetAtomWithIdx(at).GetAtomicNum() == 1}
        hydB = {at for at in graphB.adj[idxB]
                if molB.GetAtomWithIdx(at).GetAtomicNum() == 1}
        hyd_mapping.update(zip(hydA, hydB))

    return hyd_mapping

