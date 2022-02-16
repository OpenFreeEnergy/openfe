# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
from openfecli import OFECommandPlugin
from openfecli.parameters import MOL, MAPPER

def allow_two_molecules(ctx, param, value):
    if len(value) != 2:
        raise click.BadParameter("Must specify --mol exactly twice.")
    return value


@click.command(
    "atommapping",
    short_help="Explore the alchemical mutations of a given mapping"
)
@MOL.parameter(multiple=True, callback=allow_two_molecules, required=True,
               help=MOL.kwargs['help'] + " Must be specified twice.")
@MAPPER.parameter(required=True)
def atommapping(mol, mapper):
    """
    This provides tools for looking at a specific atommapping.
    """
    mol1_str, mol2_str = mol
    mol1 = MOL.get(mol1_str)
    mol2 = MOL.get(mol2_str)
    mapper_cls = MAPPER.get(mapper)
    mapper_obj = mapper_cls()
    atom_mapping_print_dict_main(mapper_obj, mol1, mol2)


def generate_mapping(mapper, mol1, mol2):
    mappings = list(mapper.suggest_mappings(mol1.rdkit, mol2.rdkit))
    if len(mappings) != 1:
        raise click.UsageError(
            f"Found {len(mappings)} mappings; this command requires a mapper "
            "to provide exactly 1 mapping"
        )
    return mappings[0]


def atom_mapping_print_dict_main(mapper, mol1, mol2):
    mapping = generate_mapping(mapper, mol1, mol2)
    print(mapping.mol1_to_mol2)


PLUGIN = OFECommandPlugin(
    command=atommapping,
    section="Setup",
    requires_ofe=(0, 0, 1),
)
