import importlib
import pytest

@pytest.fixture
def rbfe_solvated_system():
    pass


@pytest.parametrize('sys_fixture, perses_xml, known_diffs', [
    (
        'rbfe_solvated_system', 'solvated-hybrid-system.xml',
        [
        ]
    ),
])
def test_serialized_diffs(request, sys_fixture, perses_xml, known_diffs):
    mm = pytest.importorskip('openmm')
    system = request.getfixturevalue(sys_fixture)
    ofe_lines = mm.XmlSerializer.serialize(system).split("\n")

    loc = "openfe.tests.data.perses_xml_file"
    per_lines = importlib.resources.read_text(loc, perses_xml).split("\n")

    assert len(per_lines) == len(ofe_lines)

    for per, ofe in zip(per_lines, ofe_list):
        if ofe in known_diffs:
            # skip lines that match known differenes
            continue

        assert per == ofe
