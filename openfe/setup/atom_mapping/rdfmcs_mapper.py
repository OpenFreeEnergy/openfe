# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from rdkit import Chem
from rdkit.Chem import rdFMCS

from typing import Dict, Iterable

from . import LigandAtomMapping, LigandAtomMapper
from .. import SmallMoleculeComponent


def assign_idx(m):
    for i, a in enumerate(m.GetAtoms()):
        # dont set to zero, this clears the tag
        a.SetAtomMapNum(i + 1)


def add_hydrogens(mapping,
                  atom_max_distance=0.5,
                  element_change=True):
    """Take a heavy atom mapping and add Hydrogens"""
    molA = mapping.molA.to_rdkit()
    molB = mapping.molB.to_rdkit()
    A_conf = molA.GetConformer()
    B_conf = molB.GetConformer()
    extras = dict()

    for i, j in mapping.molA_to_molB.items():
        # grab neighbours of of this pair in the mapping
        A_nebs = [b.GetOtherAtomIdx(i)
                  for b in molA.GetAtomWithIdx(i).GetBonds()
                  if not b.GetOtherAtomIdx(i) in mapping.molA_to_molB]
        B_nebs = [b.GetOtherAtomIdx(j)
                  for b in molB.GetAtomWithIdx(j).GetBonds()
                  if not b.GetOtherAtomIdx(j) in mapping.molA_to_molB.values()]

        # for each combination of neighbours, check if they're close
        for i, ai in enumerate(A_nebs):
            atom_i = molA.GetAtomWithIdx(ai)
            for j, aj in enumerate(B_nebs):
                atom_j = molB.GetAtomWithIdx(aj)

                # at least one atom must be a hydrogen, or maybe both if no element changes allowed
                if element_change:
                    if not (
                            atom_i.GetAtomicNum() == 1 or atom_j.GetAtomicNum() == 1):
                        continue
                else:
                    if not (
                            atom_i.GetAtomicNum() == 1 and atom_j.GetAtomicNum() == 1):
                        continue

                dist = A_conf.GetAtomPosition(ai).Distance(
                    B_conf.GetAtomPosition(aj))
                if dist < atom_max_distance:
                    extras[ai] = aj

    return extras


def get_mapping(
        mol1, mol2, *,
        mcs_seed=None,
        atom_match='Any',
        atom_match_valences=False,
        atom_match_chiral=False,
        atom_match_charge=False,
        atom_ring_matches_ring=False,
        atom_complete_rings=False,
        atom_match_isotope=False,
        atom_max_distance=1.0,
):
    """
    Parameters
    ----------
    mol1, mol2 : rdkit Mol
    mcs_seed : str, optional
      optionally provide a starting point to improve performance over
    atom_match : str, optional
      one of 'Any', 'Element', 'Heavy'
      how must elements match when constructing the mapping.  Element forces
      an exact match, Heavy allows any heavy atoms to match, and Any allows
      hydrogens to match heavy atoms.
      default 'Any'
    atom_match_distance : float, optional
      geometric criteria for two
    """
    # make a copy of the explicitH version
    mol1 = Chem.Mol(mol1)
    mol2 = Chem.Mol(mol2)
    # label the atoms to make mapping the implicit to explicit H version trivial
    assign_idx(mol1)
    assign_idx(mol2)
    # create a copy without hydrogens
    mol1b = Chem.RemoveHs(mol1)
    mol2b = Chem.RemoveHs(mol2)

    p = rdFMCS.MCSParameters()
    if mcs_seed:
        p.InitialSeed = mcs_seed
    atom_comp = {
        'any': rdFMCS.AtomCompare.CompareAny,
        'heavy': rdFMCS.AtomCompare.CompareAnyHeavyAtom,
        'element': rdFMCS.AtomCompare.CompareElements,
    }[atom_match.lower()]  # todo catch errors
    p.AtomTyper = atom_comp
    p.AtomCompareParameters.MatchValences = atom_match_valences
    p.AtomCompareParameters.MaxDistance = atom_max_distance

    # do a match on the implicit H version
    res = rdFMCS.FindMCS([mol1b, mol2b], p)
    # convert match to mapping
    q = Chem.MolFromSmarts(res.smartsString)
    m1_idx = mol1b.GetSubstructMatch(q)
    m2_idx = mol2b.GetSubstructMatch(q)

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
    extras = add_hydrogens(heavy_mapping,
                           element_change=(atom_match.lower() == 'any'),
                           atom_max_distance=atom_max_distance)
    extras.update(heavy_mapping.molA_to_molB)

    return LigandAtomMapping(
        heavy_mapping.molA,
        heavy_mapping.molB,
        molA_to_molB=extras
    )


class RDFMCSMapper(LigandAtomMapper):
    # todo; investigate MCSParameters and hook these up here
    def __init__(self):
        pass

    def _mappings_generator(self,
                            molA: SmallMoleculeComponent,
                            molB: SmallMoleculeComponent
                            ) -> Iterable[Dict[int, int]]:
        m = get_mapping(molA.to_rdkit(),
                        molB.to_rdkit())
        yield m.molA_to_molB

    def common_core(self, molecules: list[SmallMoleculeComponent]) -> str:
        """Identify a common core across many molecules

        Returns a smarts string of the common core
        """
        # todo: this FindMCS should also use the same parameters as the
        #  "get_mapping" call above.  I.e. the common core must follow the
        #  same rules as the pairwise mappings
        core = rdFMCS.FindMCS([m.to_rdkit() for m in molecules])

        return core.smartsString
