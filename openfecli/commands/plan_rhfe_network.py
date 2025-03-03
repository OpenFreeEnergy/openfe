# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe


import click
from typing import List

from openfecli.utils import write, print_duration
from openfecli import OFECommandPlugin
from openfecli.parameters import (
    MOL_DIR, MAPPER, OUTPUT_DIR, YAML_OPTIONS, NCORES, OVERWRITE, N_PROTOCOL_REPEATS
)

def plan_rhfe_network_main(
    mapper, mapping_scorer, ligand_network_planner, small_molecules,
    solvent, n_protocol_repeats, partial_charge_settings, processors, overwrite_charges
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
    n_protocol_repeats: int
        number of completely independent repeats of the entire sampling process
    partial_charge_settings : OpenFFPartialChargeSettings
        how to assign partial charges to the input ligands
        (if they don't already have partial charges).
    processors: int
        The number of processors that should be used when generating the charges
    overwrite_charges: bool
        If any partial charges already present on the small molecules should be overwritten

    Returns
    -------
    Tuple[AlchemicalNetwork, LigandNetwork]
        Alchemical network with protocol for executing simulations, and the
        associated ligand network
    """
    from openfe.setup.alchemical_network_planner.relative_alchemical_network_planner import (
        RHFEAlchemicalNetworkPlanner
    )
    from openfe.setup.alchemical_network_planner.relative_alchemical_network_planner import RelativeHybridTopologyProtocol
    from openfe.protocols.openmm_utils.charge_generation import bulk_assign_partial_charges

    protocol_settings = RelativeHybridTopologyProtocol.default_settings()
    protocol_settings.protocol_repeats = n_protocol_repeats
    protocol = RelativeHybridTopologyProtocol(protocol_settings)

    write("assigning ligand partial charges -- this may be slow")

    charged_small_molecules = bulk_assign_partial_charges(
        molecules=small_molecules,
        overwrite=overwrite_charges,
        method=partial_charge_settings.partial_charge_method,
        toolkit_backend=partial_charge_settings.off_toolkit_backend,
        generate_n_conformers=partial_charge_settings.number_of_conformers,
        nagl_model=partial_charge_settings.nagl_model,
        processors=processors
    )

    network_planner = RHFEAlchemicalNetworkPlanner(
        mappers=mapper,
        mapping_scorer=mapping_scorer,
        ligand_network_planner=ligand_network_planner,
        protocol=protocol
    )
    alchemical_network = network_planner(
        ligands=charged_small_molecules, solvent=solvent
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
@N_PROTOCOL_REPEATS.parameter(multiple=False, required=False, default=3, help=N_PROTOCOL_REPEATS.kwargs["help"])

@NCORES.parameter(
    help=NCORES.kwargs["help"],
    default=1,
)
@OVERWRITE.parameter(
    help=OVERWRITE.kwargs["help"],
    default=OVERWRITE.kwargs["default"],
    is_flag=True
)
@print_duration
def plan_rhfe_network(molecules: List[str], yaml_settings: str, output_dir: str, n_cores: int, overwrite_charges: bool, n_protocol_repeats: int):
    """
    Plan a relative hydration free energy network, saved as JSON files for
    the quickrun command.

    This tool is an easy way to setup a RHFE-Calculation Campaign. This can
    be useful for testing our tools.  Plan-rhfe-network finds a reasonable
    network of transformations and adds the openfe rbfe protocol of year one
    to the transformations.  The output of the command can be used, in order
    to run the planned transformations with the quickrun tool.  For more
    sophisticated setups, please consider using the python layer of openfe.

    .. note::

       To ensure a consistent set of partial charges are used for each molecule across different transformations, this
       tool will automatically generate charges ahead of planning the network. ``am1bcc`` charges will be generated via
       ``ambertools``, this can also be customized using the settings yaml file.


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
    partial_charge = yaml_options.partial_charge

    write("\t\tSolvent: " + str(solvent))
    write("")

    write("Using Options:")
    write("\tMapper: " + str(mapper_obj))

    # TODO:  write nice parameter
    write("\tMapping Scorer: " + str(mapping_scorer))

    # TODO: write nice parameter
    write("\tNetworker: " + str(ligand_network_planner))

    write("\tPartial Charge Generation: " + str(partial_charge.partial_charge_method))
    if overwrite_charges:
        write("\tOverwriting partial charges")
    write("")
    write(f"\t{n_protocol_repeats=} ({n_protocol_repeats} simulation repeat(s) per transformation)\n")

    # DO
    write("Planning RHFE-Campaign:")
    alchemical_network, ligand_network = plan_rhfe_network_main(
        mapper=[mapper_obj],
        mapping_scorer=mapping_scorer,
        ligand_network_planner=ligand_network_planner,
        small_molecules=small_molecules,
        solvent=solvent,
        n_protocol_repeats=n_protocol_repeats,
        partial_charge_settings=partial_charge,
        processors=n_cores,
        overwrite_charges=overwrite_charges
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
