import inspect
import pytest
from unittest import mock
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.figure
import importlib.resources

from openfe.utils.atommapping_network_plotting import (
    AtomMappingNetworkDrawing, plot_atommapping_network,
    LigandNode, main
)

from openfe.tests.utils.test_network_plotting import mock_event


def bound_args(func, args, kwargs):
    """Return a dictionary mapping parameter name to value.

    Parameters
    ----------
    func : Callable
        this must be inspectable; mocks will require a spec
    args : List
        args list
    kwargs : Dict
        kwargs Dict

    Returns
    -------
    Dict[str, Any] :
        mapping of string name of function parameter to the value it would
        be bound to
    """
    sig = inspect.Signature.from_callable(func)
    bound = sig.bind(*args, **kwargs)
    return bound.arguments


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


@pytest.fixture
def default_node(network_drawing):
    node_dict = {node.smiles: node for node in network_drawing.graph.nodes}
    yield LigandNode(node_dict["CC"], 0.5, 0.5, 0.1, 0.1)



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

    @pytest.mark.parametrize('edge_str,left_right,molA_to_molB', [
        (("CCO", "CC"), ("CC", "CCO"), {0: 0, 1: 1}),
        (("CC", "CO"), ("CC", "CO"), {0: 0}),
        (("CCO", "CO"), ("CCO", "CO"), {0: 0, 2: 1}),
    ])
    def test_select_mock_drawing(self, edge_str, left_right, molA_to_molB,
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
            'molA': left.node,
            'molB': right.node,
            'molA_to_molB': molA_to_molB,
        }
        expected_right = {
            'extent': (0.55, 0.95, 0.5, 0.9),
            'molA': right.node,
            'molB': left.node,
            'molA_to_molB': {v: k for k, v in molA_to_molB.items()},
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


class TestLigandNode:
    def setup_method(self):
        self.fig, self.ax = plt.subplots()

    def teardown_method(self):
        plt.close(self.fig)

    def test_register_artist(self, default_node):
        assert len(self.ax.texts) == 0
        default_node.register_artist(self.ax)
        assert len(self.ax.texts) == 1
        assert self.ax.texts[0] == default_node.artist

    def test_extent(self, default_node):
        default_node.register_artist(self.ax)
        xmin, xmax, ymin, ymax = default_node.extent
        assert xmin == pytest.approx(0.5)
        assert ymin == pytest.approx(0.5)
        # can't do anything about upper bounds

    def test_xy(self, default_node):
        # default_node.register_artist(self.ax)
        x, y = default_node.xy
        assert x == pytest.approx(0.5)
        assert y == pytest.approx(0.5)


def test_plot_atommapping_network(simple_network):
    fig = plot_atommapping_network(simple_network.network)
    assert isinstance(fig, matplotlib.figure.Figure)


@pytest.mark.filterwarnings("ignore:.*non-GUI backend")
def test_main():
    # smoke test
    resource = importlib.resources.files('openfe.tests.data.serialization')
    ref = resource / "network_template.graphml"

    # prevent matplotlib from actually opening a window!
    backend = matplotlib.get_backend()
    matplotlib.use("ps")
    loc = "openfe.utils.atommapping_network_plotting.matplotlib.use"
    with mock.patch(loc, mock.Mock()):
        main(ref)

    matplotlib.use(backend)
