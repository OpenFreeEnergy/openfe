import pytest
from unittest import mock
from matplotlib import pyplot as plt
import matplotlib.figure

from openfe.utils.atommapping_network_plotting import (
    AtomMappingEdge, AtomMappingNetworkDrawing, plot_atommapping_network
)

from openfe.tests.utils.test_network_plotting import mock_event
from openfe.tests.utils.test_visualization import bound_args

from openfe.utils.network_plotting import Node


@pytest.fixture
def network_drawing(simple_network):
    nx_graph = simple_network.network.graph
    node_dict = {node.smiles: node for node in nx_graph.nodes}
    positions = {
        node_dict["CC"]: (0.0, 0.0),
        node_dict["CO"]: (0.5, 0.0),
        node_dict["CCO"]: (0.25, 0.25)
    }
    graph = AtomMappingNetworkDrawing(nx_graph, positions)
    graph.ax.set_xlim(0, 1)
    graph.ax.set_ylim(0, 1)
    yield graph
    plt.close(graph.fig)


@pytest.fixture
def default_edge(network_drawing):
    node_dict = {node.smiles: node for node in network_drawing.graph.nodes}
    yield network_drawing.edges[node_dict["CC"], node_dict["CO"]]


class TestAtomMappingEdge:
    def test_draw_mapped_molecule(self, default_edge):
        assert len(default_edge.artist.axes.images) == 0
        im = default_edge._draw_mapped_molecule(
            (0.05, 0.45, 0.5, 0.9),
            default_edge.node_artists[0].node,
            default_edge.node_artists[1].node,
            {0: 0}
        )
        # maybe add something about im itself? not sure what to test here
        assert len(default_edge.artist.axes.images) == 1
        assert default_edge.artist.axes.images[0] == im

    def test_get_image_extents(self, default_edge):
        left_extent, right_extent = default_edge._get_image_extents()
        assert left_extent == (0.05, 0.45, 0.5, 0.9)
        assert right_extent == (0.55, 0.95, 0.5, 0.9)

    def test_select(self, default_edge, network_drawing):
        assert not default_edge.picked
        assert len(default_edge.artist.axes.images) == 0

        event = mock_event('mouseup', 0.25, 0.0, network_drawing.fig)
        default_edge.select(event, network_drawing)

        assert default_edge.picked
        assert len(default_edge.artist.axes.images) == 2

    @pytest.mark.parametrize('edge_str,left_right,mol1_to_mol2', [
        (("CCO", "CC"), ("CC", "CCO"), {0: 0, 1: 1}),
        (("CC", "CO"), ("CC", "CO"), {0: 0}),
        (("CCO", "CO"), ("CCO", "CO"), {0: 0, 2: 1}),
    ])
    def test_select_mock_drawing(self, edge_str, left_right, mol1_to_mol2,
                                 network_drawing):
        # this tests that we call _draw_mapped_molecule with the correct
        # kwargs -- in particular, it ensures that we get the left and right
        # molecules correctly
        node_dict = {node.smiles: node
                     for node in network_drawing.graph.nodes}
        edge_tuple = tuple(node_dict[node] for node in edge_str)
        edge = network_drawing.edges[edge_tuple]
        left, right = [network_drawing.nodes[node_dict[node]]
                       for node in left_right]
        # ensure that we have them labelled correctly
        assert left.xy[0] < right.xy[0]
        func = edge._draw_mapped_molecule  # save for bound_args
        edge._draw_mapped_molecule = mock.Mock()

        event = mock_event('mouseup', 0.25, 0.0, network_drawing.fig)
        edge.select(event, network_drawing)

        arg_dicts = [
            bound_args(func, call.args, call.kwargs)
            for call in edge._draw_mapped_molecule.mock_calls
        ]
        expected_left = {
            'extent': (0.05, 0.45, 0.5, 0.9),
            'mol1': left.node,
            'mol2': right.node,
            'mol1_to_mol2': mol1_to_mol2,
        }
        expected_right = {
            'extent': (0.55, 0.95, 0.5, 0.9),
            'mol1': right.node,
            'mol2': left.node,
            'mol1_to_mol2': {v: k for k, v in mol1_to_mol2.items()},
        }
        assert len(arg_dicts) == 2
        assert expected_left in arg_dicts
        assert expected_right in arg_dicts

    def test_unselect(self, default_edge, network_drawing):
        # start by selecting; hard to be sure we mocked all the side effects
        # of select
        event = mock_event('mouseup', 0.25, 0.0, network_drawing.fig)
        default_edge.select(event, network_drawing)
        assert default_edge.picked
        assert len(default_edge.artist.axes.images) == 2
        assert default_edge.right_image is not None
        assert default_edge.left_image is not None

        default_edge.unselect()

        assert not default_edge.picked
        assert len(default_edge.artist.axes.images) == 0
        assert default_edge.right_image is None
        assert default_edge.left_image is None


def test_plot_atommapping_network(simple_network):
    fig = plot_atommapping_network(simple_network.network)
    assert isinstance(fig, matplotlib.figure.Figure)
