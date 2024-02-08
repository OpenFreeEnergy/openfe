"""
Dev script to generate some result jsons that are used for testing

Generates
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
from openff.toolkit import Molecule
from openff.units import unit
from kartograf.atom_aligner import align_mol_shape
from kartograf import KartografAtomMapper
import gufe
from gufe.tokenization import JSON_HANDLER
import openfe
from openfe.protocols.openmm_md.plain_md_methods import PlainMDProtocol
from openfe.protocols.openmm_afe import AbsoluteSolvationProtocol
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol


logger = logging.getLogger(__name__)

LIGA = "[H]C([H])([H])C([H])([H])C(=O)C([H])([H])C([H])([H])[H]"
LIGB = "[H]C([H])([H])C(=O)C([H])([H])C([H])([H])C([H])([H])[H]"


def get_molecule(smi, name):
    m = Molecule.from_smiles(smi)
    m.generate_conformers()
    m.assign_partial_charges(partial_charge_method="am1bcc")
    return openfe.SmallMoleculeComponent.from_openff(m, name=name)


def execute_and_serialize(dag, protocol, simname):
    logger.info(f"running {simname}")
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = pathlib.Path(tmpdir)
        dagres = gufe.protocols.execute_DAG(
            dag,
            shared_basedir=workdir,
            scratch_basedir=workdir,
            keep_shared=False,
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


def generate_md_json(smc):
    settings = PlainMDProtocol.default_settings()
    settings.simulation_settings.equilibration_length_nvt = 0.01 * unit.nanosecond
    settings.simulation_settings.equilibration_length = 0.01 * unit.nanosecond
    settings.simulation_settings.production_length = 0.01 * unit.nanosecond
    settings.forcefield_settings.nonbonded_method = "nocutoff"
    protocol = PlainMDProtocol(settings=settings)
    system = openfe.ChemicalSystem({"ligand": smc})
    dag = protocol.create(stateA=system, stateB=system, mapping=None)

    execute_and_serialize(dag, protocol, "MDProtocol")


def generate_ahfe_json(smc):
    settings = AbsoluteSolvationProtocol.default_settings()
    settings.solvent_simulation_settings.equilibration_length = 10 * unit.picosecond
    settings.solvent_simulation_settings.production_length = 500 * unit.picosecond
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

    protocol = AbsoluteSolvationProtocol(settings=settings)
    sysA = openfe.ChemicalSystem(
        {"ligand": smc, "solvent": openfe.SolventComponent()}
    )
    sysB = openfe.ChemicalSystem(
        {"solvent": openfe.SolventComponent()}
    )

    dag = protocol.create(stateA=sysA, stateB=sysB, mapping=None)

    execute_and_serialize(dag, protocol, "AHFEProtocol")


def generate_rfe_json(smcA, smcB):
    settings = RelativeHybridTopologyProtocol.default_settings()
    settings.simulation_settings.equilibration_length = 10 * unit.picosecond
    settings.simulation_settings.production_length = 250 * unit.picosecond
    settings.forcefield_settings.nonbonded_method = "nocutoff"
    protocol = RelativeHybridTopologyProtocol(settings=settings)

    a_smcB = align_mol_shape(smcB, ref_mol=smcA)
    mapper = KartografAtomMapper(atom_map_hydrogens=True)
    mapping = next(mapper.suggest_mappings(smcA, a_smcB))

    systemA = openfe.ChemicalSystem({'ligand': smcA})
    systemB = openfe.ChemicalSystem({'ligand': a_smcB})

    dag = protocol.create(
        stateA=systemA, stateB=systemB, mapping=mapping
    )

    execute_and_serialize(dag, protocol, "RHFEProtocol")
        

if __name__ == "__main__":
    molA = get_molecule(LIGA, "ligandA")
    molB = get_molecule(LIGB, "ligandB")
    generate_md_json(molA)
    generate_ahfe_json(molA)
    generate_rfe_json(molA, molB)
