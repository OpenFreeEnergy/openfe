# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import copy
import itertools
import numpy as np
import networkx as nx
from enum import Enum
from collections import OrderedDict

from rdkit import Chem
from rdkit.Chem import rdFMCS

from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components

from typing import Callable
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from .. import SmallMoleculeComponent
from . import LigandAtomMapping, LigandAtomMapper

eukli = calculate_edge_weight = lambda x, y: np.sqrt(np.sum(np.square(y - x), axis=1))

############# NEW IMPLEMENTATION IDEA for pure 3D mapping


# Working with graphs:
def _build_graph(molA: Chem.Mol, molB: Chem.Mol, max_d: float = 0.95) -> nx.Graph:
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

    G = nx.Graph()

    mol1_length = mol2_start = len(aA)
    mol2_length = len(aB)

    for n in range(mol1_length + mol2_length):
        G.add_node(n)

    edges = []
    for n1, atomPosA in enumerate(aA):
        for n2, atomPosB in enumerate(aB, start=mol2_start):
            dist = eukli(atomPosA, atomPosB)
            color = "red" if (dist > max_d) else "green"
            e = (n1, n2, {"dist": eukli(atomPosA, atomPosB), "color": color})
            if dist > max_d:
                continue
            else:
                edges.append((e[0], e[1], eukli(atomPosA, atomPosB)))
    G.add_weighted_edges_from(edges)

    return G


# modified MST
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
    min_edges = nx.minimum_spanning_edges(
        nx.MultiGraph(graph), weight="weight", algorithm="kruskal"
    )
    for n1, n2, w, attr in min_edges:
        if n1 in gMap.keys() or n1 in gMap.values() or n2 in gMap.keys() or n2 in gMap.values():
            continue
        else:
            gMap[n1] = n2

    return gMap


def get_geom_Mapping(
    molA: SmallMoleculeComponent, molB: SmallMoleculeComponent, max_d: float = 0.95
):
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
    map_dict = {
        k % mol1_length: v % mol1_length for k, v in gMap.items()
    }  # cleanup step due to graph build up.
    return LigandAtomMapping(molA, molB, map_dict)


#######################################################

# Enums:
class mapping_algorithm(Enum):
    linear_sum_assignment = "LSA"
    minimal_spanning_tree = "MST"


class WallyMapper_geometry(LigandAtomMapper):
    atom_max_distance: float
    atom_ring_matches_ring: bool
    atom_map_hydrogens: bool
    mapping_algorithm: mapping_algorithm

    def __init__(
        self,
        *,
        atom_ring_matches_ring: Optional[bool] = False,
        atom_max_distance: Optional[float] = 0.95,
        atom_map_hydrogens: Optional[bool] = False,
        mapping_algorithm: Optional[mapping_algorithm] = mapping_algorithm.linear_sum_assignment,
    ):
        """
        This mapper is a homebrew, that utilises rdkit in order
        to generate a mapping.

        Parameters
        ----------
        atom_ring_matches_ring : bool, optional
            default False
        atom_max_distance : float, optional
            geometric criteria for two atoms, how far their distance
            can be maximal (in Angstrom). Default 0.95
        mapping_algorithm : str, optional
            default="LSA"

        """
        self.atom_max_distance = atom_max_distance
        self.atom_ring_matches_ring = atom_ring_matches_ring
        self.atom_map_hydrogens = atom_map_hydrogens

        # if (mapping_algorithm == mapping_algorithm.linear_sum_assignment):
        #    self.mapping_algorithm = self.linSumAlgorithm_map
        # elif (mapping_algorithm == mapping_algorithm.minimal_spanning_tree):
        #    self.mapping_algorithm = self.mst_map
        # else:
        #    raise ValueError("Mapping algorithm not implemented or unknown (options: MST or LSA). got key: " + str(mapping_algorithm))

    """
        Set Operations
    """

    @staticmethod
    def _get_connected_atom_subsets(mol: Chem.Mol, to_be_searched: List[int]) -> List[Set[int]]:
        # Get bonded atoms sets
        bonded_atom_sets = []
        for aid in to_be_searched:
            a = mol.GetAtomWithIdx(int(aid))
            bond_atoms = [b.GetOtherAtomIdx(aid) for b in a.GetBonds()]
            connected_atoms = set([aid] + bond_atoms)

            # filter to not connect non mapped atoms
            filtered_connected_atoms = set(filter(lambda x: x in to_be_searched, connected_atoms))

            # NO mapped atom connected?
            if len(filtered_connected_atoms) == 1:
                continue
            else:
                bonded_atom_sets.append(filtered_connected_atoms)
        bonded_atom_sets = list(sorted(bonded_atom_sets))

        log.debug("Bonded atom Sets: " + str(bonded_atom_sets))

        # add new set or merge with existing set
        atom_mappings = {i: a for i, a in enumerate(to_be_searched)}
        r_atom_mappings = {a: i for i, a in enumerate(to_be_searched)}
        max_n = len(to_be_searched)

        connection_matrix = np.zeros(shape=(max_n, max_n))
        for set_i in bonded_atom_sets:
            for ai in set_i:
                for aj in set_i:
                    connection_matrix[r_atom_mappings[ai]][r_atom_mappings[aj]] = 1

        log.debug("Adjacency Matrix: " + str(connection_matrix))

        # get connected components
        matrix = csr_matrix(connection_matrix)
        n_components, labels = connected_components(csgraph=matrix, directed=False)

        log.debug("Connected Components: " + str(n_components) + "\t" + str(labels))

        # Translate back
        merged_connnected_atom_sets = []
        labels_sizes = [
            label
            for label, _ in sorted(
                np.vstack(np.unique(labels, return_counts=True)).T, key=lambda x: x[1], reverse=True
            )
        ]
        for selected_label in labels_sizes:
            connected_atoms_ids = list(
                sorted(
                    [atom_mappings[i] for i, label in enumerate(labels) if label == selected_label]
                )
            )
            merged_connnected_atom_sets.append(connected_atoms_ids)

        log.debug("Merged Sets:" + str(merged_connnected_atom_sets))

        return merged_connnected_atom_sets

    @staticmethod
    def _get_maximal_mapping_set_overlap(
        sets_a: set, sets_b: set, mapping: Dict[int, int]
    ) -> Tuple[Tuple[int, int], dict]:
        # Calculate overlaps

        log.debug("Input: " + str(sets_a) + " " + str(sets_b) + " " + str(mapping))

        max_set_combi = {}
        for ida, set_a in enumerate(sets_a):
            mapped_set_a = {mapping[a] for a in set_a}
            for idb, set_b in enumerate(sets_b):
                set_intersection_size = len(mapped_set_a.intersection(set_b))
                if set_intersection_size:
                    max_set_combi[(ida, idb)] = {
                        "overlap_count": set_intersection_size,
                        "set_a_size": len(set_a),
                        "set_b_size": len(set_b),
                    }

        # sort overlap by size
        overlap_sorted_sets = OrderedDict(
            sorted(max_set_combi.items(), key=lambda x: x[1]["overlap_count"], reverse=False)
        )

        log.debug("Set overlap properties" + str(overlap_sorted_sets))

        max_overlap_set_a_id, max_overlap_set_b_id = overlap_sorted_sets.popitem()
        return max_overlap_set_a_id, max_overlap_set_b_id

    @staticmethod
    def _filter_mapping_for_set_overlap(
        set_a: Set[int], set_b: Set[int], mapping: Dict[int, int]
    ) -> Dict[int, int]:
        filtered_mapping = {}
        for atom_a, atom_b in mapping.items():
            if atom_a in set_a and atom_b in set_b:
                filtered_mapping[atom_a] = atom_b
        return filtered_mapping

    @classmethod
    def _filter_mapping_for_max_overlapping_connected_atom_set(
        cls, moleculeA: Chem.Mol, moleculeB: Chem.Mol, atom_mapping: Dict[int, int]
    ) -> Dict[int, int]:
        # get connected sets from mappings
        sets_a = cls._get_connected_atom_subsets(moleculeA, atom_mapping.keys())
        sets_b = cls._get_connected_atom_subsets(moleculeB, atom_mapping.values())

        log.debug("Found connected sets: " + "\t".join(map(str, [sets_a, sets_b, atom_mapping])))

        # get maximally overlapping largest sets
        ((max_set_a_id, max_set_b_id), max_set) = cls._get_maximal_mapping_set_overlap(
            sets_a, sets_b, atom_mapping
        )

        # filter mapping for atoms in maximally overlapping sets
        atom_mapping = cls._filter_mapping_for_set_overlap(
            sets_a[max_set_a_id], sets_b[max_set_b_id], atom_mapping
        )

        log.debug(
            "Found connected set overlaps: "
            + "\t".join(map(str, [max_set_a_id, max_set_b_id, max_set, atom_mapping]))
        )

        return atom_mapping

    """
        Utils
    """

    @staticmethod
    def _get_full_distance_matrix(
        atomA_pos: np.array,
        atomB_pos: np.array,
        metric: Callable[
            [Union[float, Iterable], Union[float, Iterable]], Union[float, Iterable]
        ] = eukli,
    ) -> np.array:
        distance_matrix = []
        for atomPosA in atomA_pos:
            atomPos_distances = metric(atomPosA, atomB_pos)
            distance_matrix.append(atomPos_distances)
        distance_matrix = np.array(distance_matrix)
        return distance_matrix

    @staticmethod
    def _mask_atoms(
        mol, mol_pos, map_hydrogens: bool = False, masked_atoms=[]
    ) -> Tuple[Dict, List]:
        pos = []
        masked_atomMapping = {}
        for atom in mol.GetAtoms():
            atom_id = atom.GetIdx()
            if (map_hydrogens or atom.GetSymbol() != "H") and atom_id not in masked_atoms:
                masked_atomMapping[len(masked_atomMapping)] = atom_id
                pos.append(mol_pos[atom_id])
        pos = np.array(pos)

        return masked_atomMapping, pos

    @classmethod
    def mst_map(
        cls,
        molA: Chem.Mol,
        molB: Chem.Mol,
        max_d: float = 0.95,
        masked_atoms_molA=[],
        masked_atoms_molB=[],
        pre_mapped_atoms={},
    ) -> LigandAtomMapping:
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

        aA = molA.GetConformer().GetPositions()
        aB = molB.GetConformer().GetPositions()
        num_bAtoms = aB.shape[0]

        # distance matrix:  - full graph
        edges = []
        for i, atomPosA in enumerate(aA):
            atomPos_distances = np.vstack(
                [eukli(atomPosA, aB), np.full(num_bAtoms, i), np.arange(num_bAtoms)]
            ).T
            edges.extend(atomPos_distances)
        edges = np.array(edges)

        # priority queue based on edge weight
        sorted_edges = edges[edges[:, 0].argsort()]

        # MST like algorithm
        mapping = {}
        for w, x, y in sorted_edges:
            if w > max_d:
                break
            else:
                if x not in mapping.keys() and y not in mapping.values():
                    mapping[int(x)] = int(y)

        # get max connected set
        mapping = cls._filter_mapping_for_max_overlapping_connected_atom_set(
            moleculeA=molA, moleculeB=molB, atom_mapping=mapping
        )

        return mapping

    @classmethod
    def linSumAlgorithm_map(
        cls,
        molA: Chem.Mol,
        molB: Chem.Mol,
        max_d: float = 0.95,
        masked_atoms_molA=[],
        masked_atoms_molB=[],
        pre_mapped_atoms={},
        map_hydrogens: bool = False,
    ) -> LigandAtomMapping:

        molA_rdkit = Chem.Mol(molA)
        molB_rdkit = Chem.Mol(molB)
        molA_pos = molA_rdkit.GetConformer().GetPositions()
        molB_pos = molB_rdkit.GetConformer().GetPositions()
        masked_atoms_molA = copy.deepcopy(masked_atoms_molA)
        masked_atoms_molB = copy.deepcopy(masked_atoms_molB)


        if len(pre_mapped_atoms) > 0:
            masked_atoms_molA.extend(pre_mapped_atoms.keys())
            masked_atoms_molB.extend(pre_mapped_atoms.values())

        log.debug("Positions lengths: " + str(len(molA_pos)) + " " + str(len(molB_pos)))
        log.debug("Positions: \n" + "\n\t".join(map(str, [molA_pos, molB_pos])))
        log.debug("masked_atoms_molA: "+ str(masked_atoms_molA))
        log.debug("masked_atoms_molB: "+ str(masked_atoms_molB))
        log.debug("pre_mapped_atoms: "+ str(pre_mapped_atoms))
        log.debug("map_hydrogens: "+ str(map_hydrogens))

        log.info("Masking Atoms")

        molA_masked_atomMapping, molA_pos = cls._mask_atoms(
            mol=molA_rdkit,
            mol_pos=molA_pos,
            masked_atoms=masked_atoms_molA,
            map_hydrogens=map_hydrogens,
        )
        molB_masked_atomMapping, molB_pos = cls._mask_atoms(
            mol=molB_rdkit,
            mol_pos=molB_pos,
            masked_atoms=masked_atoms_molB,
            map_hydrogens=map_hydrogens,
        )
        
        log.debug(
            "Remove Masked Atoms: \n"
            + "\n\t".join(
                map(str, [molA_masked_atomMapping, molB_masked_atomMapping]))
                + "\n"
                + "\n\t".join(map(str, [molA_pos, molB_pos]))
        )
        log.debug(
            "filtered Masked Positions lengths: " + "\t".join(map(str, [len(molA_pos), len(molB_pos)]))
        )

        if len(molA_pos) == 0 or len(molB_pos) == 0:  # TODO: check if this is correct
            log.warning("no mappable Atoms were found!")
            return pre_mapped_atoms

        # Calculate mapping
        log.info("Build Distance Matrix")
        # distance matrix:  - full graph
        distance_matrix = cls._get_full_distance_matrix(molA_pos, molB_pos)
        log.debug("Distance Matrix: \n" + str(distance_matrix))

        # Mask distance matrix with max_d
        # np.inf is considererd as not possible in lsa implementation - therefore use a high value
        mask_dist_val = max_d * 10**6
        masked_dmatrix = np.array(
            np.ma.where(distance_matrix < max_d, distance_matrix, mask_dist_val)
        )
        log.debug("masked Distance Matrix: " + str(max_d) + "\n\t" + str(masked_dmatrix))

        # solve atom mappings
        log.info("Calculate Mapping")
        row_ind, col_ind = linear_sum_assignment(masked_dmatrix)
        raw_mapping = list(zip(map(int, row_ind), map(int, col_ind)))
        log.debug("Raw Mapping: " + str(raw_mapping))

        # filter for mask value
        mapping = dict(filter(lambda x: masked_dmatrix[x] < mask_dist_val, raw_mapping))
        log.debug("Distance filtered Mapping: " + str(mapping))

        if len(mapping) == 0:  # TODO: check if this is correct
            log.warning("no mapping could be found!")
            return pre_mapped_atoms

        # reverse any prior masking:
        mapping = {
            molA_masked_atomMapping[k]: molB_masked_atomMapping[v] for k, v in mapping.items()
        }

        if len(pre_mapped_atoms) > 0:
            mapping.update(pre_mapped_atoms)
        log.debug("reverse Masking Mapping: " + str(mapping))

        # Reduce mapping to maximally overlapping two connected sets
        log.info("Find Maximal overlapping connected sets of mapped atoms")
        mapping = cls._filter_mapping_for_max_overlapping_connected_atom_set(
            moleculeA=molA_rdkit, moleculeB=molB_rdkit, atom_mapping=mapping
        )
        log.debug("Set overlap Mapping: " + str(mapping))

        return mapping

    """
        Mapping functions
    """

    def _mappings_generator(
        self,
        molA: SmallMoleculeComponent,
        molB: SmallMoleculeComponent,
    ) -> Iterable[Dict[int, int]]:

        m = self.get_mapping(molA.to_rdkit(), molB.to_rdkit())
        yield m.molA_to_molB

    def get_mapping(
        self,
        molA: SmallMoleculeComponent,
        molB: SmallMoleculeComponent,
        masked_atoms_molA=[],
        masked_atoms_molB=[],
        pre_mapped_atoms={},
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
        pre_mapped_atoms = copy.deepcopy(pre_mapped_atoms)

        log.info("#################################")
        log.info("Map Heavy Atoms ")
        log.info("#################################")

        mapping = self.linSumAlgorithm_map(
            molA._rdkit,
            molB._rdkit,
            max_d=self.atom_max_distance,
            masked_atoms_molA=[],
            masked_atoms_molB=[],
            pre_mapped_atoms=pre_mapped_atoms,
            map_hydrogens=False,
        )

        if self.atom_map_hydrogens:
            log.info("#################################")
            log.info("Map Hydrogen Atoms: ")
            log.info("#################################")

            pre_mapped_atoms.update(mapping)

            mapping = self.linSumAlgorithm_map(
                molA._rdkit,
                molB._rdkit,
                max_d=self.atom_max_distance,
                masked_atoms_molA=masked_atoms_molA,
                masked_atoms_molB=masked_atoms_molB,
                pre_mapped_atoms=pre_mapped_atoms,
                map_hydrogens=self.atom_map_hydrogens,
            )

        return LigandAtomMapping(molA, molB, componentA_to_componentB=mapping)


#######################################################
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
    best = float("inf")

    for mA, mB in itertools.product(mA_matches, mB_matches):
        d = total_mismatch(molA, mA, molB, mB)

        if d < best:
            best = d
            ret = (mA, mB)

    return ret


class WallyMapper_MCS(LigandAtomMapper):
    mcs_params: rdFMCS.MCSParameters

    # todo; investigate MCSParameters and hook these up here
    def __init__(
        self,
        *,
        mcs_seed: Optional[str] = None,
        atom_match: Optional[str] = "any",
        atom_match_valences: Optional[bool] = False,
        atom_match_chiral: Optional[bool] = False,
        atom_match_charge: Optional[bool] = False,
        atom_ring_matches_ring: Optional[bool] = False,
        atom_complete_rings: Optional[bool] = False,
        atom_max_distance: Optional[float] = 1.0,
        bond_match: Optional[str] = "any",
    ):
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
            raise ValueError(
                "Atom comparison type was not recognized, you "
                f"provided: {atom_match}\n"
                "Please provide on of: "
                f"{list(atom_comparisons._member_names_)}"
            )
        self.mcs_params.AtomCompareParameters.MatchValences = atom_match_valences
        # TODO: Not sure this is what we want for chirality handling
        #       this will rely on CIP priority, which might get flipped
        #       on changing certain r-groups
        self.mcs_params.AtomCompareParameters.MatchChiralTag = atom_match_chiral
        self.mcs_params.AtomCompareParameters.MatchFormalCharge = atom_match_charge
        self.mcs_params.AtomCompareParameters.CompleteRingsOnly = atom_ring_matches_ring

        self.mcs_params.AtomCompareParameters.MaxDistance = atom_max_distance
        if mcs_seed is not None:
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
    def mcs_seed(self, value: str):
        self.mcs_params.InitialSeed = value

    def _mappings_generator(
        self,
        molA: SmallMoleculeComponent,
        molB: SmallMoleculeComponent,
    ) -> Iterable[Dict[int, int]]:

        m = self.get_mapping(molA.to_rdkit(), molB.to_rdkit())
        yield m.molA_to_molB

    def get_mapping(
        self,
        mol1: Chem.Mol,
        mol2: Chem.Mol,
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
        res = rdFMCS.FindMCS(
            [mol1b, mol2b],
            parameters=self.mcs_params,
        )

        # convert match to mapping
        m1_idx, m2_idx = select_best_mapping(mol1b, mol2b, res.smartsString)

        # remap indices to original molecule
        m1_idx = [mol1b.GetAtomWithIdx(i).GetAtomMapNum() - 1 for i in m1_idx]
        m2_idx = [mol2b.GetAtomWithIdx(i).GetAtomMapNum() - 1 for i in m2_idx]

        heavy_mapping = LigandAtomMapping(
            SmallMoleculeComponent.from_rdkit(mol1),
            SmallMoleculeComponent.from_rdkit(mol2),
            molA_to_molB=dict(zip(m1_idx, m2_idx)),
        )

        # attach hydrogens
        extras = self.add_hydrogens(heavy_mapping)
        extras.update(heavy_mapping.molA_to_molB)

        return LigandAtomMapping(heavy_mapping.molA, heavy_mapping.molB, molA_to_molB=extras)

    def add_hydrogens(self, mapping: LigandAtomMapping):
        """Take a heavy atom mapping and add Hydrogens"""
        molA = mapping.molA.to_rdkit()
        molB = mapping.molB.to_rdkit()
        A_conf = molA.GetConformer()
        B_conf = molB.GetConformer()

        element_change = self.mcs_params.AtomTyper == atom_comparisons.any.value
        max_atom_dist = self.mcs_params.AtomCompareParameters.MaxDistance
        extras = dict()

        for i, j in mapping.molA_to_molB.items():
            # grab neighbours of this pair in the mapping
            A_nebs = [
                b.GetOtherAtomIdx(i)
                for b in molA.GetAtomWithIdx(i).GetBonds()
                if not b.GetOtherAtomIdx(i) in mapping.molA_to_molB
            ]
            B_nebs = [
                b.GetOtherAtomIdx(j)
                for b in molB.GetAtomWithIdx(j).GetBonds()
                if not b.GetOtherAtomIdx(j) in mapping.molA_to_molB.values()
            ]

            # for each combination of neighbours, check if they're close
            for i, ai in enumerate(A_nebs):
                atom_i = molA.GetAtomWithIdx(ai)
                for j, aj in enumerate(B_nebs):
                    atom_j = molB.GetAtomWithIdx(aj)

                    # at least one atom must be a hydrogen, or maybe both if no
                    # element changes allowed
                    if element_change:
                        if not (atom_i.GetAtomicNum() == 1 or atom_j.GetAtomicNum() == 1):
                            continue
                    else:
                        if not (atom_i.GetAtomicNum() == 1 and atom_j.GetAtomicNum() == 1):
                            continue

                    dist = A_conf.GetAtomPosition(ai).Distance(B_conf.GetAtomPosition(aj))

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
        self.mcs_params.AtomTyper = atom_comparisons["element"].value
        self.mcs_params.Threshold = 0.75
        core = rdFMCS.FindMCS([m.to_rdkit() for m in molecules], parameters=self.mcs_params)
        self.mcs_params.AtomTyper = tmp_type

        return core.smartsString

    @staticmethod
    def _assign_idx(m: Chem.Mol):
        for i, a in enumerate(m.GetAtoms()):
            # dont set to zero, this clears the tag
            a.SetAtomMapNum(i + 1)
