import pathlib

import click
from openfecli import OFECommandPlugin
from openfecli.parameters import MOL_DIR, YAML_OPTIONS, OUTPUT_FILE_AND_EXT, NCORES, OVERWRITE

YAML_HELP = """
Path to a YAML file specifying the method to use to charge the molecules
(any atom mapper or network generation options will be ignored).

Supported partial charge method choices are:
    - ``am1bcc``
    - ``am1bccelf10`` (only possible if ``off_toolkit_backend`` is ``openeye``)
    - ``nagl`` (must have openff-nagl installed)
    - ``espaloma`` (must have espaloma_charge installed)

``settings`` allows for passing in any keyword arguments of the method's corresponding Python API.

For example:
::

  partial_charge:
    method: am1bcc
    settings:
      off_toolkit_backend: ambertools

"""


@click.command("charge-molecules", short_help="Generate partial charges for a set of molecules.")
@MOL_DIR.parameter(required=True, help=MOL_DIR.kwargs["help"] + " Any number of sdf paths.")
@YAML_OPTIONS.parameter(multiple=False, required=False, default=None, help=YAML_HELP)
@OUTPUT_FILE_AND_EXT.parameter(
    help="The name of the SDF file the charged ligands should be written to.",
    required=True,
    type=click.Path(exists=False, path_type=pathlib.Path),
)
@NCORES.parameter(help=NCORES.kwargs["help"], default=1)
@OVERWRITE.parameter(help=OVERWRITE.kwargs["help"], default=OVERWRITE.kwargs["default"], is_flag=True)  # fmt: skip
def charge_molecules(molecules, yaml_settings, output, n_cores, overwrite_charges):
    """
    Generate partial charges for the set of input molecules and write them to file.
    """
    from openfecli.utils import write

    if output.exists():
        raise FileExistsError(
            f"The output file {output} already exists, choose a new file to write the charged ligands to"
        )

    write("SMALL MOLECULE PARTIAL CHARGE GENERATOR")
    write("_________________________________________")
    write("")

    from openfe.protocols.openmm_utils.charge_generation import bulk_assign_partial_charges

    write("Parsing in Files: ")

    # INPUT
    write("\tGot input: ")

    small_molecules = MOL_DIR.get(molecules)
    write("\t\tSmall Molecules: " + " ".join([str(sm) for sm in small_molecules]))

    yaml_options = YAML_OPTIONS.get(yaml_settings)
    partial_charge = yaml_options.partial_charge

    write("Using Options:")
    write("\tPartial Charge Generation: " + str(partial_charge.partial_charge_method))
    if overwrite_charges:
        write("\tOverwriting partial charges")
    write("")

    charged_molecules = bulk_assign_partial_charges(
        molecules=small_molecules,
        overwrite=overwrite_charges,
        method=partial_charge.partial_charge_method,
        toolkit_backend=partial_charge.off_toolkit_backend,
        generate_n_conformers=partial_charge.number_of_conformers,
        nagl_model=partial_charge.nagl_model,
        processors=n_cores,
    )

    write("\tDone")
    write("")

    # OUTPUT
    write("Output:")
    write(f"\tSaving to: {output}")

    with output.open(mode="w") as output:
        for mol in charged_molecules:
            output.write(mol.to_sdf())


PLUGIN = OFECommandPlugin(command=charge_molecules, section="Miscellaneous", requires_ofe=(0, 3))
