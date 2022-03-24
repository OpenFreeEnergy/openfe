import pytest
from unittest import mock
from numpy import testing as npt

from matplotlib import pyplot as plt
import networkx as nx
from openfe.utils.network_plotting import (
    Node, Edge, EventHandler, GraphDrawing
)

from matplotlib.backend_bases import MouseEvent, MouseButton


def mock_event(event_name, xdata, ydata, fig=None):
    if fig is None:
        fig, ax = plt.subplots()

    if len(fig.axes) != 1:
        raise RuntimeError("Error in test setup: figure must have exactly "
                           "one Axes object associated")

    name = {
        'mousedown': 'button_press_event',
        'mouseup': 'button_release_event',
        'drag': 'motion_notify_event',
    }[event_name]

    matplotlib_buttons = {
        'mousedown': MouseButton.LEFT,
        'mouseup': MouseButton.LEFT,
        'drag': MouseButton.LEFT,
    }
    button = matplotlib_buttons.get(event_name, None)
    x, y = fig.axes[0].transData.transform((xdata, ydata))
    return MouseEvent(name, fig.canvas, x, y, button)


@pytest.fixture
def nx_graph():
    return nx.MultiDiGraph([("A", "B"), ("B", "C"), ("B", "D")])


@pytest.fixture
def drawing_graph(nx_graph):
    return GraphDrawing(nx_graph, positions={
        "A": (0.0, 0.0), "B": (0.5, 0.0), "C": (0.5, 0.5), "D": (0.0, 0.5)
    })


class TestNode:
    def setup(self):
        self.node = Node("B", 0.5, 0.0)
        self.fig, self.ax = plt.subplots()
        self.node.register_artist(self.ax)

    def teardown(self):
        plt.close(self.fig)

    def test_register_artist(self):
        node = Node("B", 0.6, 0.0)
        fig, ax = plt.subplots()
        assert len(ax.patches) == 0
        node.register_artist(ax)
        assert len(ax.patches) == 1
        assert node.artist == ax.patches[0]

    def test_extent(self):
        assert self.node.extent == (0.5, 0.6, 0.0, 0.1)

    def test_xy(self):
        assert self.node.xy == (0.5, 0.0)

    def test_unselect(self):
        # initially blue; turn it red; unselect should switch it back
        assert self.node.artist.get_facecolor() == (0.0, 0.0, 1.0, 1.0)
        self.node.artist.set(color="red")
        assert self.node.artist.get_facecolor() != (0.0, 0.0, 1.0, 1.0)
        self.node.unselect()
        assert self.node.artist.get_facecolor() == (0.0, 0.0, 1.0, 1.0)

    def test_edge_select(self):
        # initially blue; edge_select should turn it red
        assert self.node.artist.get_facecolor() == (0.0, 0.0, 1.0, 1.0)
        edge = mock.Mock()  # unused in this method
        self.node.edge_select(edge)
        assert self.node.artist.get_facecolor() == (1.0, 0.0, 0.0, 1.0)

    def test_update_location(self):
        assert self.node.artist.xy == (0.5, 0.0)
        self.node.update_location(0.7, 0.5)
        assert self.node.artist.xy == (0.7, 0.5)
        assert self.node.xy == (0.7, 0.5)

    @pytest.mark.parametrize('point,expected', [
        ((0.55, 0.05), True),
        ((0.5, 0.5), False),
        ((-10, -10), False),
    ])
    def test_contains(self, point, expected):
        event = mock_event('drag', *point, fig=self.fig)
        assert self.node.contains(event) == expected

    def test_on_mousedown_in_rect(self, drawing_graph):
        event = mock_event('mousedown', 0.55, 0.05, self.fig)
        assert Node.lock is None
        assert self.node.press is None

        self.node.on_mousedown(event, drawing_graph)
        assert Node.lock == self.node
        assert self.node.press is not None
        Node.lock = None

    def test_on_mousedown_in_axes(self, drawing_graph):
        event = mock_event('mousedown', 0.25, 0.25, self.fig)

        assert Node.lock is None
        assert self.node.press is None
        self.node.on_mousedown(event, drawing_graph)
        assert Node.lock is None
        assert self.node.press is None

    def test_on_mousedown_out_axes(self, drawing_graph):
        node = Node("B", 0.5, 0.6)
        event = mock_event('mousedown', 0.55, 0.05, self.fig)

        fig2, ax2 = plt.subplots()
        node.register_artist(ax2)

        assert Node.lock is None
        assert node.press is None
        node.on_mousedown(event, drawing_graph)
        assert Node.lock is None
        assert node.press is None

    def test_on_drag(self, drawing_graph):
        event = mock_event('drag', 0.7, 0.7, self.fig)
        # set up things that should happen on mousedown
        Node.lock = self.node
        self.node.press = (0.5, 0.0), (0.55, 0.05)

        self.node.on_drag(event, drawing_graph)

        npt.assert_allclose(self.node.xy, (0.65, 0.65))

        # undo the lock; normally handled by mouseup
        Node.lock = None

    def test_on_drag_do_nothing(self, drawing_graph):
        event = mock_event('drag', 0.7, 0.7, self.fig)

        # don't set lock -- early exit
        original = self.node.xy
        self.node.on_drag(event, drawing_graph)
        assert self.node.xy == original

    def test_on_drag_no_mousedown(self, drawing_graph):
        event = mock_event('drag', 0.7, 0.7, self.fig)
        Node.lock = self.node

        with pytest.raises(RuntimeError, match="drag until mouse down"):
            self.node.on_drag(event, drawing_graph)

        Node.lock = None

    def test_on_mouseup(self, drawing_graph):
        event = mock_event('drag', 0.7, 0.7, self.fig)
        Node.lock = self.node
        self.node.press = (0.5, 0.0), (0.55, 0.05)

        self.node.on_mouseup(event, drawing_graph)
        assert Node.lock is None
        assert self.node.press is None

    def test_blitting(self):
        pytest.skip("Blitting hasn't been implemented yet")


class TestEdge:
    def setup(self):
        pass

    def test_register_artist(self):
        pass

    def test_contains(self):
        pass

    def test_edge_xs_ys(self):
        pass

    def test_set_standard(self):
        pass

    def test_select(self):
        pass

    def test_update_locations(self):
        pass


class TestEventHandler:
    def setup(self):
        pass

    def test_connect(self):
        pass

    def test_disconnect(self):
        pass

    def test_get_event_container(self):
        pass

    def test_on_mousedown(self):
        pass

    def test_on_drag(self):
        pass

    def test_on_mouseup(self):
        pass


class TestGraphDrawing:
    def setup(self):
        pass

    def test_edges_for_node(self):
        pass

    def test_get_nodes_extent(self):
        pass

    def test_reset_bounds(self):
        pass

    def test_draw(self):
        pass
