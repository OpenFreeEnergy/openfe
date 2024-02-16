import pytest
from openff.units import unit
import gufe
from gufe import SolventComponent, ChemicalSystem
from gufe.tests.test_protocol import DummyProtocol


@pytest.fixture
def solv_comp():
    yield SolventComponent(positive_ion="K", negative_ion="Cl", ion_concentration=0.0 * unit.molar)


@pytest.fixture
def solvated_complex(T4_protein_component, benzene_transforms, solv_comp):
    return ChemicalSystem(
        {
            "ligand": benzene_transforms["toluene"],
            "protein": T4_protein_component,
            "solvent": solv_comp,
        },
    )


@pytest.fixture
def solvated_ligand(benzene_transforms, solv_comp):
    return ChemicalSystem(
        {
            "ligand": benzene_transforms["toluene"],
            "solvent": solv_comp,
        },
    )


@pytest.fixture
def absolute_transformation(solvated_ligand, solvated_complex):
    return gufe.Transformation(
        solvated_ligand,
        solvated_complex,
        protocol=DummyProtocol(settings=DummyProtocol.default_settings()),
        mapping=None,
    )


@pytest.fixture
def complex_equilibrium(solvated_complex):
    return gufe.NonTransformation(
        solvated_complex,
        protocol=DummyProtocol(settings=DummyProtocol.default_settings()),
    )


@pytest.fixture
def benzene_variants_star_map(benzene_transforms, solv_comp, T4_protein_component):
    variants = ["toluene", "phenol", "benzonitrile", "anisole", "benzaldehyde", "styrene"]

    # define the solvent chemical systems and transformations between
    # benzene and the others
    solvated_ligands = {}
    solvated_ligand_transformations = {}

    solvated_ligands["benzene"] = ChemicalSystem(
        {
            "solvent": solv_comp,
            "ligand": benzene_transforms["benzene"],
        },
        name="benzene-solvent",
    )

    for ligand in variants:
        solvated_ligands[ligand] = ChemicalSystem(
            {
                "solvent": solv_comp,
                "ligand": benzene_transforms[ligand],
            },
            name=f"{ligand}-solvent",
        )

        solvated_ligand_transformations[("benzene", ligand)] = gufe.Transformation(
            solvated_ligands["benzene"],
            solvated_ligands[ligand],
            protocol=DummyProtocol(settings=DummyProtocol.default_settings()),
            mapping=None,
        )

    # define the complex chemical systems and transformations between
    # benzene and the others
    solvated_complexes = {}
    solvated_complex_transformations = {}

    solvated_complexes["benzene"] = gufe.ChemicalSystem(
        {"protein": T4_protein_component, "solvent": solv_comp, "ligand": benzene_transforms["benzene"]},
        name="benzene-complex",
    )

    for ligand in variants:
        solvated_complexes[ligand] = gufe.ChemicalSystem(
            {"protein": T4_protein_component, "solvent": solv_comp, "ligand": benzene_transforms[ligand]},
            name=f"{ligand}-complex",
        )
        solvated_complex_transformations[("benzene", ligand)] = gufe.Transformation(
            solvated_complexes["benzene"],
            solvated_complexes[ligand],
            protocol=DummyProtocol(settings=DummyProtocol.default_settings()),
            mapping=None,
        )

    return gufe.AlchemicalNetwork(
        list(solvated_ligand_transformations.values()) + list(solvated_complex_transformations.values()),
    )
