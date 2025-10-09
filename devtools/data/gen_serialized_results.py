"""
Dev script to generate some result jsons that are used for testing

Generates
- SepTopProtocol_json_results.gy
  - used in septop_json fixture
- AHFEProtocol_json_results.gz
  - used in afe_solvation_json fixture
- RHFEProtocol_json_results.gz
  - used in rfe_transformation_json fixture
- MDProtocol_json_results.gz
  - used in md_json fixture
"""
import gzip
import json
import logging
import pathlib
import tempfile
from rdkit import Chem
from openff.toolkit import (
    Molecule, RDKitToolkitWrapper, AmberToolsToolkitWrapper
)
from openff.toolkit.utils.toolkit_registry import (
    toolkit_registry_manager, ToolkitRegistry
)
from openff.units import unit
from kartograf.atom_aligner import align_mol_shape
from kartograf import KartografAtomMapper
import gufe
from gufe.tokenization import JSON_HANDLER
import openfe
from openfe.protocols.openmm_md.plain_md_methods import PlainMDProtocol
from openfe.protocols.openmm_afe import AbsoluteSolvationProtocol
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openfe.protocols.openmm_septop import SepTopProtocol


logger = logging.getLogger(__name__)

LIGA = "[H]C([H])([H])C([H])([H])C(=O)C([H])([H])C([H])([H])[H]"
LIGB = "[H]C([H])([H])C(=O)C([H])([H])C([H])([H])C([H])([H])[H]"

amber_rdkit = ToolkitRegistry(
    [RDKitToolkitWrapper(), AmberToolsToolkitWrapper()]
)


def get_molecule(smi, name):
    with toolkit_registry_manager(amber_rdkit):
        m = Molecule.from_smiles(smi)
        m.generate_conformers()
        m.assign_partial_charges(partial_charge_method="am1bcc")
    return openfe.SmallMoleculeComponent.from_openff(m, name=name)


def get_hif2a_inputs():
    with gzip.open('inputs/hif2a_protein.pdb.gz', 'r') as f:
        protcomp = openfe.ProteinComponent.from_pdb_file(f, name='hif2a_prot')

    with gzip.open('inputs/hif2a_ligands.sdf.gz', 'r') as f:
        smcs = [openfe.SmallMoleculeComponent(mol) for mol in
                list(Chem.ForwardSDMolSupplier(f, removeHs=False))]

    return smcs, protcomp


def execute_and_serialize(dag, protocol, simname):
    logger.info(f"running {simname}")
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = pathlib.Path(tmpdir)
        dagres = gufe.protocols.execute_DAG(
            dag,
            shared_basedir=workdir,
            scratch_basedir=workdir,
            keep_shared=True,
            n_retries=3
        )
    protres = protocol.gather([dagres])

    outdict = {
        "estimate": protres.get_estimate(),
        "uncertainty": protres.get_uncertainty(),
        "protocol_result": protres.to_dict(),
        "unit_results": {
            unit.key: unit.to_keyed_dict()
            for unit in dagres.protocol_unit_results
        }
    }

    with gzip.open(f"{simname}_json_results.gz", 'wt') as zipfile:
        json.dump(outdict, zipfile, cls=JSON_HANDLER.encoder)


def generate_md_settings():
    settings = PlainMDProtocol.default_settings()
    settings.simulation_settings.equilibration_length_nvt = 0.01 * unit.nanosecond
    settings.simulation_settings.equilibration_length = 0.01 * unit.nanosecond
    settings.simulation_settings.production_length = 0.01 * unit.nanosecond
    settings.forcefield_settings.nonbonded_method = "nocutoff"

    return settings


def generate_md_json(smc):
    protocol = PlainMDProtocol(settings=generate_md_settings())
    system = openfe.ChemicalSystem({"ligand": smc})
    dag = protocol.create(stateA=system, stateB=system, mapping=None)

    execute_and_serialize(dag, protocol, "MDProtocol")


def generate_ahfe_settings():
    settings = AbsoluteSolvationProtocol.default_settings()
    settings.solvent_equil_simulation_settings.equilibration_length_nvt = 10 * unit.picosecond
    settings.solvent_equil_simulation_settings.equilibration_length = 10 * unit.picosecond
    settings.solvent_equil_simulation_settings.production_length = 10 * unit.picosecond
    settings.solvent_simulation_settings.equilibration_length = 10 * unit.picosecond
    settings.solvent_simulation_settings.production_length = 500 * unit.picosecond
    settings.vacuum_equil_simulation_settings.equilibration_length = 10 * unit.picosecond
    settings.vacuum_equil_simulation_settings.production_length = 10 * unit.picosecond
    settings.vacuum_simulation_settings.equilibration_length = 10 * unit.picosecond
    settings.vacuum_simulation_settings.production_length = 1000 * unit.picosecond
    settings.lambda_settings.lambda_elec = [0.0, 0.25, 0.5, 0.75, 1.0, 1.0,
                                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                            1.0]
    settings.lambda_settings.lambda_vdw = [0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 0.24,
                                           0.36, 0.48, 0.6, 0.7, 0.77, 0.85,
                                           1.0]
    settings.protocol_repeats = 3
    settings.solvent_simulation_settings.n_replicas = 14
    settings.vacuum_simulation_settings.n_replicas = 14
    settings.solvent_simulation_settings.early_termination_target_error = 0.12 * unit.kilocalorie_per_mole
    settings.vacuum_simulation_settings.early_termination_target_error = 0.12 * unit.kilocalorie_per_mole
    settings.vacuum_engine_settings.compute_platform = 'CPU'
    settings.solvent_engine_settings.compute_platform = 'CUDA'

    return settings

    
def generate_ahfe_json(smc):
    protocol = AbsoluteSolvationProtocol(settings=generate_ahfe_settings())
    sysA = openfe.ChemicalSystem(
        {"ligand": smc, "solvent": openfe.SolventComponent()}
    )
    sysB = openfe.ChemicalSystem(
        {"solvent": openfe.SolventComponent()}
    )

    dag = protocol.create(stateA=sysA, stateB=sysB, mapping=None)

    execute_and_serialize(dag, protocol, "AHFEProtocol")


def generate_rfe_settings():
    settings = RelativeHybridTopologyProtocol.default_settings()
    settings.simulation_settings.equilibration_length = 10 * unit.picosecond
    settings.simulation_settings.production_length = 250 * unit.picosecond
    settings.forcefield_settings.nonbonded_method = "nocutoff"
    
    return settings


def generate_rfe_json(smcA, smcB):
    protocol = RelativeHybridTopologyProtocol(settings=generate_rfe_settings())

    a_smcB = align_mol_shape(smcB, ref_mol=smcA)
    mapper = KartografAtomMapper(atom_map_hydrogens=True)
    mapping = next(mapper.suggest_mappings(smcA, a_smcB))

    systemA = openfe.ChemicalSystem({'ligand': smcA})
    systemB = openfe.ChemicalSystem({'ligand': a_smcB})

    dag = protocol.create(
        stateA=systemA, stateB=systemB, mapping=mapping
    )

    execute_and_serialize(dag, protocol, "RHFEProtocol")


def generate_septop_settings():
    settings = SepTopProtocol.default_settings()
    settings.solvent_equil_simulation_settings.equilibration_length_nvt = 10 * unit.picosecond
    settings.solvent_equil_simulation_settings.equilibration_length = 10 * unit.picosecond
    settings.solvent_equil_simulation_settings.production_length = 10 * unit.picosecond
    settings.solvent_simulation_settings.equilibration_length = 100 * unit.picosecond
    settings.solvent_simulation_settings.production_length = 500 * unit.picosecond
    settings.solvent_simulation_settings.time_per_iteration = 2.5 * unit.ps
    settings.complex_equil_simulation_settings.equilibration_length_nvt = 10 * unit.picosecond
    settings.complex_equil_simulation_settings.equilibration_length = 10 * unit.picosecond
    settings.complex_equil_simulation_settings.production_length = 10 * unit.picosecond
    settings.complex_simulation_settings.equilibration_length = 100 * unit.picosecond
    settings.complex_simulation_settings.production_length = 500 * unit.picosecond
    settings.complex_simulation_settings.time_per_iteration = 2.5 * unit.ps
    settings.solvent_solvation_settings.box_shape = 'dodecahedron'
    settings.complex_solvation_settings.box_shape = 'dodecahedron'
    settings.solvent_solvation_settings.solvent_padding = 1.2 * unit.nanometer
    settings.complex_solvation_settings.solvent_padding = 1.0 * unit.nanometer
    settings.forcefield_settings.nonbonded_cutoff = 0.9 * unit.nanometer
    settings.protocol_repeats = 1
    settings.engine_settings.compute_platform = 'CUDA'

    return settings


def generate_septop_json():
    hif2a_ligands, hif2a_protein = get_hif2a_inputs()
    protocol = SepTopProtocol(settings=generate_septop_settings())
    sysA = openfe.ChemicalSystem(
        {
            "ligand_A": hif2a_ligands[0],
            "protein": hif2a_protein,
            "solvent": openfe.SolventComponent(),
        }
    )
    sysB = openfe.ChemicalSystem(
        {
            "ligand_B": hif2a_ligands[1],
            "protein": hif2a_protein,
            "solvent": openfe.SolventComponent(),
        }
    )

    dag = protocol.create(stateA=sysA, stateB=sysB, mapping=None)
    execute_and_serialize(dag, protocol, "SepTopProtocol")
        

if __name__ == "__main__":
    molA = get_molecule(LIGA, "ligandA")
    molB = get_molecule(LIGB, "ligandB")
    # generate_md_json(molA)
    # generate_ahfe_json(molA)
    # generate_rfe_json(molA, molB)
    generate_septop_json()
