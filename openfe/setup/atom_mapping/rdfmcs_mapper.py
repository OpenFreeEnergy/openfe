# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from enum import Enum
import itertools
from rdkit import Chem
from rdkit.Chem import rdFMCS

from typing import Dict, Iterable, List, Optional

from . import LigandAtomMapping, LigandAtomMapper
from .. import SmallMoleculeComponent


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


class RDFMCSMapper(LigandAtomMapper):
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
            self.mcs_params.AtomTyper = atom_comparisons[str(atom_match).lower()]
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
        self.mcs_seed = mcs_seed

    @property
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
    def _assign_idx(self, m: Chem.Mol):
        for i, a in enumerate(m.GetAtoms()):
            # dont set to zero, this clears the tag
            a.SetAtomMapNum(i + 1)
