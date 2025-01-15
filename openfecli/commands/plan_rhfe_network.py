# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe


import click
from typing import List

from openfecli.utils import write, print_duration
from openfecli import OFECommandPlugin
from openfecli.parameters import (
    MOL_DIR, MAPPER, OUTPUT_DIR, YAML_OPTIONS,
)

def plan_rhfe_network_main(
    mapper, mapping_scorer, ligand_network_planner, small_molecules,
    solvent, partial_charge_settings,
):
    """Utility method to plan a relative hydration free energy network.

    Parameters
    ----------
    mapper : list[LigandAtomMapper]
        list of mappers to use to generate the mapping
    mapping_scorer : Callable
        scorer, that evaluates the generated mappings
    ligand_network_planner : Callable
        function building the network from the ligands, mappers and mapping_scorer
    small_molecules : Iterable[SmallMoleculeComponent]
        molecules of the system
    solvent : SolventComponent
        Solvent component used for solvation
    partial_charge_settings : OpenFFPartialChargeSettings
        how to assign partial charges to the input ligands
        (if they don't already have partial charges).

    Returns
    -------
    Tuple[AlchemicalNetwork, LigandNetwork]
        Alchemical network with protocol for executing simulations, and the
        associated ligand network
    """
    from openfe.setup.alchemical_network_planner.relative_alchemical_network_planner import (
        RHFEAlchemicalNetworkPlanner
    )
    from openfe.protocols.openmm_utils.charge_generation import assign_offmol_partial_charges
    from openfe import SmallMoleculeComponent

    charged_small_molecules = []
    for smc im small_molecules:
        offmol = smc.to_openff()
        assign_offmol_partial_charges(
            offmol=offmol,
            overwrite=False,
            method=partial_charge_settings.partial_charge_method,
            toolkit_backend=partial_charge_settings.off_toolkit_backend,
            generate_n_conformers=partial_charge_settings.number_of_conformers,
            nagl_model=partial_charge_settings.nagl_model
        )
        charged_small_molecules.append(SmallMoleculeComponent.from_openff(offmol))

    network_planner = RHFEAlchemicalNetworkPlanner(
        mappers=mapper,
        mapping_scorer=mapping_scorer,
        ligand_network_planner=ligand_network_planner,
    )
    alchemical_network = network_planner(
        ligands=small_molecules, solvent=solvent
    )

    return alchemical_network, network_planner._ligand_network


@click.command(
    "plan-rhfe-network",
    short_help=(
        "Plan a relative hydration free energy network, saved as JSON files "
        "for the quickrun command."
    ),
)
@MOL_DIR.parameter(
    required=True, help=MOL_DIR.kwargs["help"] + " Any number of sdf paths."
)
@YAML_OPTIONS.parameter(
    multiple=False, required=False, default=None,
    help=YAML_OPTIONS.kwargs["help"],
)
@OUTPUT_DIR.parameter(
    help=OUTPUT_DIR.kwargs["help"] + " Defaults to `./alchemicalNetwork`.",
    default="alchemicalNetwork",
)
@print_duration
def plan_rhfe_network(molecules: List[str], yaml_settings: str, output_dir: str):
    """
    Plan a relative hydration free energy network, saved as JSON files for
    the quickrun command.

    This tool is an easy way to setup a RHFE-Calculation Campaign. This can
    be useful for testing our tools.  Plan-rhfe-network finds a reasonable
    network of transformations and adds the openfe rbfe protocol of year one
    to the transformations.  The output of the command can be used, in order
    to run the planned transformations with the quickrun tool.  For more
    sophisticated setups, please consider using the python layer of openfe.

    The tool will parse the input and run the rbfe network planner, which
    executes following steps:

        1. generate an atom mapping for all possible ligand pairs. (default:
           Lomap)

        2. score all atom mappings. (default: Lomap default score)

        3. build network form all atom mapping scores (default: minimal
           spanning graph)

    The generated Network will be stored in a folder containing for each
    transformation a JSON file, that can be run with quickrun (or other
    future tools).
    """

    from openfecli.plan_alchemical_networks_utils import plan_alchemical_network_output

    write("RHFE-NETWORK PLANNER")
    write("______________________")
    write("")

    write("Parsing in Files: ")

    # INPUT
    write("\tGot input: ")

    small_molecules = MOL_DIR.get(molecules)
    write(
        "\t\tSmall Molecules: "
        + " ".join([str(sm) for sm in small_molecules])
    )

    yaml_options = YAML_OPTIONS.get(yaml_settings)
    mapper_obj = yaml_options.mapper
    mapping_scorer = yaml_options.scorer
    ligand_network_planner = yaml_options.ligand_network_planner
    solvent = yaml_options.solvent

    write("\t\tSolvent: " + str(solvent))
    write("")

    write("Using Options:")
    write("\tMapper: " + str(mapper_obj))

    # TODO:  write nice parameter
    write("\tMapping Scorer: " + str(mapping_scorer))

    # TODO: write nice parameter
    write("\tNetworker: " + str(ligand_network_planner))
    write("")

    # DO
    write("Planning RHFE-Campaign:")
    alchemical_network, ligand_network = plan_rhfe_network_main(
        mapper=[mapper_obj],
        mapping_scorer=mapping_scorer,
        ligand_network_planner=ligand_network_planner,
        small_molecules=small_molecules,
        solvent=solvent,
    )
    write("\tDone")
    write("")

    # OUTPUT
    write("Output:")
    write("\tSaving to: " + output_dir)
    plan_alchemical_network_output(
        alchemical_network=alchemical_network,
        ligand_network=ligand_network,
        folder_path=OUTPUT_DIR.get(output_dir),
    )


PLUGIN = OFECommandPlugin(
    command=plan_rhfe_network, section="Network Planning", requires_ofe=(0, 3)
)
