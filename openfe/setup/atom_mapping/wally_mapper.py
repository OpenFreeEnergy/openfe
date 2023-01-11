# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import itertools
import numpy as np
import networkx as nx
from enum import Enum
from rdkit import Chem
from rdkit.Chem import rdFMCS

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from .. import SmallMoleculeComponent
from . import LigandAtomMapping, LigandAtomMapper

############# NEW IMPLEMENTATION IDEA for pure 3D mapping

#Working with graphs:
eukli = lambda x, y: np.sqrt(np.sum(np.square(y-x)))
calculate_edge_weight = eukli

def _build_graph(molA:Chem.Mol, molB:Chem.Mol, max_d:float= 0.95)-> nx.Graph:
    """
    This function builds a full graph, with the exception of filtering for max_d

    Parameters
    ----------
    molA : Chem.Mol
        _description_
    molB :  Chem.Mol
        _description_
    max_d : float, optional
        _description_, by default 0.95

    Returns
    -------
    nx.Graph
        constructed graph
    """
    
    aA = molA.GetConformer().GetPositions()
    aB = molB.GetConformer().GetPositions()

    G=nx.Graph()

    mol1_length = mol2_start = len(aA)
    mol2_length = len(aB)

    for n in range(mol1_length+mol2_length):
        G.add_node(n)

    edges = []
    for n1, atomPosA in enumerate(aA):
        for n2, atomPosB in enumerate(aB, start=mol2_start):
            dist = eukli(atomPosA, atomPosB)
            color = "red" if(dist>max_d) else "green"
            e= (n1, n2,  {"dist": eukli(atomPosA, atomPosB), "color": color})
            if(dist > max_d):
                continue
            else:
                edges.append((e[0], e[1], eukli(atomPosA, atomPosB)))
    G.add_weighted_edges_from(edges)
    
    return G

#modified MST
def _get_mst_chain_graph(graph):
    """
    This function uses a graph and returns its edges in order of an MST according to Kruskal algorithm, but filters the edges such no branching occurs in the tree (actually not really a tree anymore I guess...).
    The 'no-branching' translate to no atom can be mapped twice in the final result.

    Parameters
    ----------
    graph : nx.Graph
        _description_

    Returns
    -------
    dict[int, int]
        resulting atom mapping
    """
    gMap = {}
    min_edges = nx.minimum_spanning_edges(nx.MultiGraph(graph), weight='weight', algorithm="kruskal")
    for n1, n2, w, attr in min_edges:
        if(n1 in gMap.keys() or n1 in gMap.values() or n2 in gMap.keys() or n2 in gMap.values()):
            continue
        else:
            gMap[n1]=n2
            #print(n1, n2, attr['weight'])
            
    return gMap

def get_geom_Mapping(molA:SmallMoleculeComponent, molB:SmallMoleculeComponent, max_d:float = 0.95):
    """
    This function is a networkx graph based implementation to build up an Atom Mapping purely on 3D criteria.

    Parameters
    ----------
    molA : SmallMoleculeComponent
        _description_
    molB : SmallMoleculeComponent
        _description_
    max_d : float, optional
        _description_, by default 0.95

    Returns
    -------
    LigandAtomMapping
        resulting 3d Atom mapping
    """
    mol1_length = molA._rdkit.GetNumAtoms()
    G = _build_graph(molA=molA._rdkit, molB=molB._rdkit, max_d=max_d)
    gMap = _get_mst_chain_graph(G)
    map_dict = {k%mol1_length:v%mol1_length for k,v in gMap.items()} #cleanup step due to graph build up.
    return LigandAtomMapping(molA, molB, map_dict)


#Working with pure numpy:

import numpy as np

# MST
def mst_map(molA:SmallMoleculeComponent, molB:SmallMoleculeComponent, max_d:float = 0.95)->LigandAtomMapping:
    """
    This function is a numpy graph based implementation to build up an Atom Mapping purely on 3D criteria.

    Parameters
    ----------
    molA : SmallMoleculeComponent
        _description_
    molB : SmallMoleculeComponent
        _description_
    max_d : float, optional
        _description_, by default 0.95

    Returns
    -------
    dict[int, int]
        resulting atomMapping
    """
    mA = molA._rdkit
    mB = molB._rdkit

    aA = mA.GetConformer().GetPositions()
    aB = mB.GetConformer().GetPositions()
    num_bAtoms = aB.shape[0]

    eukli = lambda x, y: np.sqrt(np.sum(np.square(y-x), axis=1))

    #distance matrix:  - full graph
    edges = []
    for i, atomPosA in enumerate(aA):
        atomPos_distances = np.vstack([eukli(atomPosA, aB), np.full(num_bAtoms, i), np.arange(num_bAtoms)]).T  
        edges.extend(atomPos_distances)
        
    edges = np.array(edges)
    sorted_edges = edges[edges[:, 0].argsort()]

    mapping = {}
    for w, x, y in sorted_edges: 
        if(w>max_d):
            break
        else:
            if(not x in mapping.keys() and not y in mapping.values()):
                mapping[int(x)]=int(y)
    
    return LigandAtomMapping(molA, molB, mapping)

## Linear Sum Assignment
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment

#get 
def _get_connected_subsets(mol:Chem.Mol, to_be_searched:List[int])->List[Set[int]]:
    #Get connected atoms
    connected_sets = []
    for aid in to_be_searched:
        a = mol.GetAtomWithIdx(int(aid))
        bond_atoms = [b.GetOtherAtomIdx(aid) for b in a.GetBonds()]
        set_of_connected_atoms = set([aid]+bond_atoms)
        searched_set_of_connected_atoms  = set(filter(lambda x: x in to_be_searched, set_of_connected_atoms))
        
        if(len(searched_set_of_connected_atoms)==1):
            continue
        
        for i, g in enumerate(connected_sets):
            if(len(g.intersection(searched_set_of_connected_atoms))>0):
                connected_sets[i] = connected_sets[i].union(searched_set_of_connected_atoms)
                break
        else:
            connected_sets.append(searched_set_of_connected_atoms)

    #sort connected atom sets by size
    connected_sets = sorted(connected_sets, key=lambda x: len(x), reverse=True)
    return connected_sets

def _get_maximal_mapping_set_overlap(sets_a:set, sets_b:set, mapping:Dict[int,int], verbose:bool=False)->Tuple[Tuple[int, int], dict]:
    #Calculate overlaps
    max_set_combi={}
    for ida, set_a in enumerate(sets_a):
        mapped_set_a = {mapping[a] for a in set_a}
        for idb, set_b in enumerate(sets_b):
            set_intersection_size = len(mapped_set_a.intersection(set_b))
            if(set_intersection_size):
                max_set_combi[(ida, idb)] = {'overlap_count': set_intersection_size, 
                                            "set_a_size": len(set_a), 
                                            "set_b_size": len(set_b)}

    #sort overlap by size
    max_set_keys = OrderedDict(sorted(max_set_combi.items(), key=lambda x: x[1]['overlap_count'], reverse=True))
    if(verbose): print(max_set_keys)

    set_a_id, set_b_id = max_set_keys.popitem()
    return set_a_id, set_b_id

def linSumAlgorithm_map(molA, molB, max_d:float = 0.95, verbose:bool=True)->LigandAtomMapping:

    mA = molA._rdkit
    mB = molB._rdkit

    aA = mA.GetConformer().GetPositions()
    aB = mB.GetConformer().GetPositions()

    #distance matrix:  - full graph
    eukli = lambda x, y: np.sqrt(np.sum(np.square(y-x), axis=1))

    distance_matrix = []
    for i, atomPosA in enumerate(aA):
        atomPos_distances = eukli(atomPosA, aB)
        distance_matrix.append(atomPos_distances)
    distance_matrix = np.array(distance_matrix)
    #how to make max_d a thing
    distance_matrix=np.array(np.ma.where(distance_matrix<max_d, distance_matrix, np.inf))

    clipped_distance_matrix = []
    mapped_rows = {}
    for row_id, row in enumerate(distance_matrix):
        if(any([np.isfinite(y) for y in row])):
            mapped_rows[len(mapped_rows)] = row_id
            clipped_distance_matrix.append(row)
        else:
            continue
    clipped_distance_matrix = np.array(clipped_distance_matrix)

    if(verbose): print("Mapped rows: ", mapped_rows)
    # solve atom mappings
    row_ind, col_ind = linear_sum_assignment(clipped_distance_matrix)
    mapped_row_inds = list(map(lambda x: int(mapped_rows[x]), row_ind)) # map back to original row ids
    col_ind = list(map(int, col_ind))
    mapping = dict(zip(mapped_row_inds, col_ind))

    # get connected sets from mappings
    sets_a = _get_connected_subsets(mA, mapping.keys())

    sets_b = _get_connected_subsets(mB, mapping.values())

    if(verbose): print("Found connected sets: ", sets_a, sets_b, mapping, row_ind)
    # get maximally overlapping largest sets
    ((max_set_a_id, max_set_b_id),max_set) = _get_maximal_mapping_set_overlap(sets_a, sets_b, mapping)

    # filter for only mapped atoms
    found_mapping = {}
    if(max_set["set_a_size"]> max_set["set_b_size"]):
        print("b")
        found_mapping = {int(a):int(mapping[a]) for a in sets_a[max_set_a_id] if(a in mapping)}
    else:
        print("a")
        r_mapping = {v:k for k,v in mapping.items()}
        found_mapping = {int(r_mapping[b]):int(b) for b in sets_b[max_set_b_id] if(b in r_mapping)}

    if(verbose): print("Final Mapping", found_mapping)
    mapping = LigandAtomMapping(molA, molB, found_mapping)

    return mapping


# Enums:
class atom_comparisons(Enum):
    any = rdFMCS.AtomCompare.CompareAny
    heavy = rdFMCS.AtomCompare.CompareAnyHeavyAtom
    element = rdFMCS.AtomCompare.CompareElements


class bond_comparisons(Enum):
    any = rdFMCS.BondCompare.CompareAny
    order = rdFMCS.BondCompare.CompareOrder
    orderExact = rdFMCS.BondCompare.CompareOrderExact


def total_mismatch(molA: Chem.Mol, idxA: tuple[int], molB: Chem.Mol, idxB: tuple[int]) -> float:
    """Total distance between atoms in mapping

    molA/B : rdkit Mols
    idxA/B : indices of the mapping, same length

    Returns distance as float
    """
    confA = molA.GetConformer()
    confB = molB.GetConformer()

    total = 0
    for i, j in zip(idxA, idxB):
        dA = confA.GetAtomPosition(i)
        dB = confB.GetAtomPosition(j)

        dist = dA.Distance(dB)

        total += dist

    return total


def select_best_mapping(molA, molB, smarts: str) -> tuple[tuple[int], tuple[int]]:
    """work around symmetry to find best mapping in 3d

    Parameters
    ----------
    molA, molB : rdkit.Mol
    smarts : str
      smarts string of the MCS

    Returns
    -------
    mapping : pair of tuple[int]
      the best pairing of indices to use as a mapping
      this pairing has the minimum distance between atoms
    """
    query = Chem.MolFromSmarts(smarts)

    mA_matches = molA.GetSubstructMatches(query, uniquify=False)
    mB_matches = molB.GetSubstructMatches(query, uniquify=False)

    ret = tuple()
    best = float('inf')

    for mA, mB in itertools.product(mA_matches, mB_matches):
        d = total_mismatch(molA, mA, molB, mB)

        if d < best:
            best = d
            ret = (mA, mB)

    return ret


class WallyMapper(LigandAtomMapper):
    mcs_params: rdFMCS.MCSParameters

    # todo; investigate MCSParameters and hook these up here
    def __init__(self, *,
                 mcs_seed: Optional[str] = None,
                 atom_match: Optional[str] = 'any',
                 atom_match_valences: Optional[bool] = False,
                 atom_match_chiral: Optional[bool] = False,
                 atom_match_charge: Optional[bool] = False,
                 atom_ring_matches_ring: Optional[bool] = False,
                 atom_complete_rings: Optional[bool] = False,
                 atom_max_distance: Optional[float] = 1.0,
                 bond_match: Optional[str] = 'any'):
        """
        This mapper is a homebrew, that utilises rdkit in order
        to generate a mapping.

        Parameters
        ----------
        atom_match : str, optional
            one of 'Any', 'Element', 'Heavy'
            how must elements match when constructing the mapping.
            Element forces an exact match, Heavy allows any heavy
            atoms to match, and Any allows hydrogens to match heavy
            atoms.  default 'Any'
        atom_match_valences : bool, optional
            atoms must strictly match valence to be included in mapping,
            default False
        atom_match_chiral : bool, optional
            atoms must match chirality to be included in mapping
            default False
        atom_ring_matches_ring : bool, optional
            default False
        atom_max_distance : float, optional
            geometric criteria for two atoms, how far their distance
            can be maximal. Default 1.0
        """

        # Settings Catalog
        self.mcs_params = rdFMCS.MCSParameters()
        try:
            self.mcs_params.AtomTyper = atom_comparisons[str(atom_match).lower()].value
        except KeyError:
            raise ValueError("Atom comparison type was not recognized, you "
                             f"provided: {atom_match}\n"
                             "Please provide on of: "
                             f"{list(atom_comparisons._member_names_)}")
        self.mcs_params.AtomCompareParameters.MatchValences = atom_match_valences
        # TODO: Not sure this is what we want for chirality handling
        #       this will rely on CIP priority, which might get flipped
        #       on changing certain r-groups
        self.mcs_params.AtomCompareParameters.MatchChiralTag = atom_match_chiral
        self.mcs_params.AtomCompareParameters.MatchFormalCharge = atom_match_charge
        self.mcs_params.AtomCompareParameters.CompleteRingsOnly = atom_ring_matches_ring

        self.mcs_params.AtomCompareParameters.MaxDistance = atom_max_distance
        if(mcs_seed is not None):
            self.mcs_seed = str(mcs_seed)

    @property
    def atom_comparison_type(self) -> str:
        return atom_comparisons(self.mcs_params.AtomTyper).name

    #
    def atom_comparison_type(self) -> str:
        return atom_comparisons(self.mcs_params.AtomTyper).name
    
    @property
    def bond_comparison_type(self) -> str:
        return bond_comparisons(self.mcs_params.BondTyper).name

    @property
    def atom_match_valences(self) -> bool:
        return self.mcs_params.AtomCompareParameters.MatchValences

    @property
    def atom_max_distance(self) -> bool:
        return self.mcs_params.AtomCompareParameters.MaxDistance

    @property
    def atom_complete_rings(self) -> bool:
        return self.mcs_params.AtomCompareParameters.CompleteRingsOnly

    @property
    def atom_ring_matches_ring(self) -> bool:
        return self.mcs_params.AtomCompareParameters.RingMatchesRingOnly

    @property
    def atom_match_charge(self) -> bool:
        return self.mcs_params.AtomCompareParameters.MatchFormalCharge

    @property
    def atom_match_chiral(self) -> bool:
        return self.mcs_params.AtomCompareParameters.MatchChiralTag

    @property
    def mcs_seed(self) -> str:
        return self.mcs_params.InitialSeed
    
    @mcs_seed.setter
    def mcs_seed(self, value:str):
        self.mcs_params.InitialSeed=value

    def _mappings_generator(self,
                            molA: SmallMoleculeComponent,
                            molB: SmallMoleculeComponent,
                            ) -> Iterable[Dict[int, int]]:

        m = self.get_mapping(molA.to_rdkit(),
                             molB.to_rdkit())
        yield m.molA_to_molB

    def get_mapping(self,
                    mol1: Chem.Mol, mol2: Chem.Mol,
                    ) -> LigandAtomMapping:
        """
        The function effectivly maps the two molecules on to each other and
        applies the given settings by the obj.

        Parameters
        ----------
        mol1, mol2 : rdkit.Chem.Mol
            two rdkit molecules that should be mapped onto each other
        mcs_seed : str, optional
            smart string containing the MCS of the mols,
            optionally provide a starting point to improve performance over
        """
        # make a copy of the explicitH version
        mol1 = Chem.Mol(mol1)
        mol2 = Chem.Mol(mol2)

        # label the atoms to make mapping the implicit to explicit H version
        # trivial
        self._assign_idx(mol1)
        self._assign_idx(mol2)

        # create a copy without hydrogens
        mol1b = Chem.RemoveHs(mol1)
        mol2b = Chem.RemoveHs(mol2)

        # do a match on the implicit H version
        res = rdFMCS.FindMCS([mol1b, mol2b],
                             parameters=self.mcs_params,
                             )

        # convert match to mapping
        m1_idx, m2_idx = select_best_mapping(mol1b, mol2b, res.smartsString)

        # remap indices to original molecule
        m1_idx = [
            mol1b.GetAtomWithIdx(i).GetAtomMapNum() - 1
            for i in m1_idx
        ]
        m2_idx = [
            mol2b.GetAtomWithIdx(i).GetAtomMapNum() - 1
            for i in m2_idx
        ]

        heavy_mapping = LigandAtomMapping(
            SmallMoleculeComponent.from_rdkit(mol1),
            SmallMoleculeComponent.from_rdkit(mol2),
            molA_to_molB=dict(zip(m1_idx, m2_idx))
        )

        # attach hydrogens
        extras = self.add_hydrogens(heavy_mapping)
        extras.update(heavy_mapping.molA_to_molB)

        return LigandAtomMapping(
            heavy_mapping.molA,
            heavy_mapping.molB,
            molA_to_molB=extras
        )

    def add_hydrogens(self, mapping: LigandAtomMapping):
        """Take a heavy atom mapping and add Hydrogens"""
        molA = mapping.molA.to_rdkit()
        molB = mapping.molB.to_rdkit()
        A_conf = molA.GetConformer()
        B_conf = molB.GetConformer()

        element_change = (self.mcs_params.AtomTyper ==
                          atom_comparisons.any.value)
        max_atom_dist = self.mcs_params.AtomCompareParameters.MaxDistance
        extras = dict()

        for i, j in mapping.molA_to_molB.items():
            # grab neighbours of this pair in the mapping
            A_nebs = [b.GetOtherAtomIdx(i)
                      for b in molA.GetAtomWithIdx(i).GetBonds()
                      if not b.GetOtherAtomIdx(i) in mapping.molA_to_molB]
            B_nebs = [b.GetOtherAtomIdx(j)
                      for b in molB.GetAtomWithIdx(j).GetBonds()
                      if not b.GetOtherAtomIdx(j)
                      in mapping.molA_to_molB.values()]

            # for each combination of neighbours, check if they're close
            for i, ai in enumerate(A_nebs):
                atom_i = molA.GetAtomWithIdx(ai)
                for j, aj in enumerate(B_nebs):
                    atom_j = molB.GetAtomWithIdx(aj)

                    # at least one atom must be a hydrogen, or maybe both if no
                    # element changes allowed
                    if element_change:
                        if not (atom_i.GetAtomicNum() ==
                                1 or atom_j.GetAtomicNum() == 1):
                            continue
                    else:
                        if not (atom_i.GetAtomicNum() ==
                                1 and atom_j.GetAtomicNum() == 1):
                            continue

                    dist = A_conf.GetAtomPosition(ai).Distance(
                        B_conf.GetAtomPosition(aj))

                    if dist < max_atom_dist:
                        extras[ai] = aj

        return extras

    def common_core(self, molecules: List[SmallMoleculeComponent]) -> str:
        """
        Identify a common core across many molecules
        the common core is stored in mcs_seed
        """
        # TODO: We properly copy settings out here so we don't break anything
        # TODO: Works only with this weird construct for me.
        tmp_type = self.mcs_params.AtomTyper
        self.mcs_params.AtomTyper = atom_comparisons['element'].value
        self.mcs_params.Threshold = 0.75
        core = rdFMCS.FindMCS([m.to_rdkit()
                              for m in molecules], parameters=self.mcs_params)
        self.mcs_params.AtomTyper = tmp_type

        return core.smartsString

    @staticmethod
    def _assign_idx(m: Chem.Mol):
        for i, a in enumerate(m.GetAtoms()):
            # dont set to zero, this clears the tag
            a.SetAtomMapNum(i + 1)
