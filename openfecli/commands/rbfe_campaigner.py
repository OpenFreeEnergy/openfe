# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe


"""
Here I want to build the cmd tool for easy campaigner with RBFE. The output should be runnable with quickrun directly!
    So user experience would be:
        easy_campaign -l sdf_dir -p receptor.pdb -> Alchem network
        quickrun -i alchem_network

"""
import json
import pathlib
import click
from openfecli import OFECommandPlugin
from openfecli.parameters import MOL, MAPPER, OUTPUT_FILE_AND_EXT
from openfecli.parameters.output import ensure_file_does_not_exist

@click.command(
    'plan_rbfe',
    short_help="Run a planning session, saved as a JSON file"
)
@click.argument('in_ligand_sdf_dir', type=click.File(mode='r'),
                help=(
                        "path to directory containing sdfs"
                ),
                required=True)
@click.option(
    'in_receptor_pdb', '-d', default=None,
    type=click.Path(dir_okay=False, file_okay=True, writable=False,
                    path_type=pathlib.Path),
    help=(
        "path to the receptor pdb"
    ),
)
@click.option(
    'out_json_path', '-o', default=None,
    type=click.Path(dir_okay=False, file_okay=True, writable=True,
                    path_type=pathlib.Path),
    help="output file (JSON format) for the final results",
    callback=ensure_file_does_not_exist,
    required=True
)
def plan_rbfe_campaign(ligand_sdf_dir:str, receptor_pdb:str, out_json_path:str):

    import glob
    from gufe import SmallMoleculeComponent, ProteinComponent, SolventComponent

    small_molecule_paths = glob.glob(ligand_sdf_dir+"/*.sdf")
    small_molecules = [SmallMoleculeComponent.from_sdf_file(sdf_file_path) for sdf_file_path in small_molecule_paths]
    protein = ProteinComponent.from_pdb_file(pdb_file=receptor_pdb)
    solvent = SolventComponent()

    #define vars here to allow quick choices.
    from openfe.setup import LomapAtomMapper
    from openfe.setup.atom_mapping.lomap_scorers import default_lomap_score
    from openfe.setup.ligand_network_planning import minimal_spanning_graph

    mapper = LomapAtomMapper()
    mapping_scorers = [default_lomap_score]
    networker = minimal_spanning_graph

    from openfe.setup.campaigners.easy_campaigner import easy_rbfe_campainger
    campaigner = easy_rbfe_campainger(mapper=mapper,
                                      mapping_scorers=mapping_scorers,
                                      networker=networker)
    alchemical_network = campaigner(ligands=small_molecules, solvent=solvent, receptor=protein)

    an_dict = alchemical_network.to_dict()
    json.dump(an_dict, open(out_json_path, "w"))


PLUGIN = OFECommandPlugin(
    command=plan_rbfe_campaign,
    section="Planning",
    requires_ofe=(0, 3)
)
