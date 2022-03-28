import pytest
from unittest import mock
from numpy import testing as npt

from matplotlib import pyplot as plt
import networkx as nx
from openfe.utils.network_plotting import (
    Node, Edge, EventHandler, GraphDrawing
)

from matplotlib.backend_bases import MouseEvent, MouseButton


def _get_fig_ax(fig):
    if fig is None:
        fig, _ = plt.subplots()

    if len(fig.axes) != 1:  # -no-cov-
        raise RuntimeError("Error in test setup: figure must have exactly "
                           "one Axes object associated")

    return fig, fig.axes[0]


def mock_event(event_name, xdata, ydata, fig=None):
    fig, ax = _get_fig_ax(fig)
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
    x, y = ax.transData.transform((xdata, ydata))
    return MouseEvent(name, fig.canvas, x, y, button)


def make_mock_graph(fig=None):
    fig, ax = _get_fig_ax(fig)

    def make_mock_node(node, x, y):
        return mock.Mock(node=node, x=x, y=y)

    def make_mock_edge(node1, node2, data):
        return mock.Mock(node_artists=[node1, node2], data=data)

    node_A = make_mock_node("A", 0.0, 0.0)
    node_B = make_mock_node("B", 0.5, 0.0)
    node_C = make_mock_node("C", 0.5, 0.5)
    node_D = make_mock_node("D", 0.0, 0.5)
    edge_AB = make_mock_edge(node_A, node_B, {'data': "AB"})
    edge_BC = make_mock_edge(node_B, node_C, {'data': "BC"})
    edge_BD = make_mock_edge(node_B, node_D, {'data': "BD"})

    mock_graph = mock.Mock(
        nodes={node.node: node for node in [node_A, node_B, node_C, node_D]},
        edges={tuple(edge.node_artists): edge
               for edge in [edge_AB, edge_BC, edge_BD]},
    )
    return mock_graph


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

    def test_on_mousedown_in_rect(self):
        event = mock_event('mousedown', 0.55, 0.05, self.fig)
        drawing_graph = make_mock_graph(self.fig)
        assert Node.lock is None
        assert self.node.press is None

        self.node.on_mousedown(event, drawing_graph)
        assert Node.lock == self.node
        assert self.node.press is not None
        Node.lock = None

    def test_on_mousedown_in_axes(self):
        event = mock_event('mousedown', 0.25, 0.25, self.fig)
        drawing_graph = make_mock_graph(self.fig)

        assert Node.lock is None
        assert self.node.press is None
        self.node.on_mousedown(event, drawing_graph)
        assert Node.lock is None
        assert self.node.press is None

    def test_on_mousedown_out_axes(self):
        node = Node("B", 0.5, 0.6)
        event = mock_event('mousedown', 0.55, 0.05, self.fig)
        drawing_graph = make_mock_graph(self.fig)

        fig2, ax2 = plt.subplots()
        node.register_artist(ax2)

        assert Node.lock is None
        assert node.press is None
        node.on_mousedown(event, drawing_graph)
        assert Node.lock is None
        assert node.press is None

    def test_on_drag(self):
        event = mock_event('drag', 0.7, 0.7, self.fig)
        # this test some integration, so we need more than a mock
        drawing_graph = GraphDrawing(
            nx.MultiDiGraph(([("A", "B"), ("B", "C"), ("B", "D")])),
            positions={"A": (0.0, 0.0), "B": (0.5, 0.0),
                       "C": (0.5, 0.5), "D": (0.0, 0.5)}
        )
        # set up things that should happen on mousedown
        Node.lock = self.node
        self.node.press = (0.5, 0.0), (0.55, 0.05)

        self.node.on_drag(event, drawing_graph)

        npt.assert_allclose(self.node.xy, (0.65, 0.65))

        # undo the lock; normally handled by mouseup
        Node.lock = None

    def test_on_drag_do_nothing(self):
        event = mock_event('drag', 0.7, 0.7, self.fig)
        drawing_graph = make_mock_graph(self.fig)

        # don't set lock -- early exit
        original = self.node.xy
        self.node.on_drag(event, drawing_graph)
        assert self.node.xy == original

    def test_on_drag_no_mousedown(self):
        event = mock_event('drag', 0.7, 0.7, self.fig)
        drawing_graph = make_mock_graph(self.fig)
        Node.lock = self.node

        with pytest.raises(RuntimeError, match="drag until mouse down"):
            self.node.on_drag(event, drawing_graph)

        Node.lock = None

    def test_on_mouseup(self):
        event = mock_event('drag', 0.7, 0.7, self.fig)
        drawing_graph = make_mock_graph(self.fig)
        Node.lock = self.node
        self.node.press = (0.5, 0.0), (0.55, 0.05)

        self.node.on_mouseup(event, drawing_graph)
        assert Node.lock is None
        assert self.node.press is None

    def test_blitting(self):
        pytest.skip("Blitting hasn't been implemented yet")


class TestEdge:
    def setup(self):
        self.nodes = [Node("A", 0.0, 0.0), Node("B", 0.5, 0.0)]
        self.data = {"data": "values"}
        self.edge = Edge(*self.nodes, self.data)
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.edge.register_artist(self.ax)

    def teardown(self):
        plt.close(self.fig)

    def test_register_artist(self):
        fig, ax = plt.subplots()
        edge = Edge(*self.nodes, self.data)
        assert len(ax.get_lines()) == 0
        edge.register_artist(ax)
        assert len(ax.get_lines()) == 1
        assert ax.get_lines()[0] == edge.artist

    @pytest.mark.parametrize('point,expected', [
        ((0.25, 0.05), True),
        ((0.6, 0.1), False),
    ])
    def test_contains(self, point, expected):
        event = mock_event('drag', *point, fig=self.fig)
        assert self.edge.contains(event) == expected

    def test_edge_xs_ys(self):
        npt.assert_allclose(self.edge._edge_xs_ys(*self.nodes),
                            ((0.05, 0.55), (0.05, 0.05)))

    def _get_colors(self):
        colors = {node: node.artist.get_facecolor()
                  for node in self.nodes}
        colors[self.edge] = self.edge.artist.get_color()
        return colors

    def test_unselect(self):
        original = self._get_colors()

        for node in self.nodes:
            node.artist.set(color='red')

        self.edge.artist.set(color='red')

        # ensure that we have changed from the original values
        changed = self._get_colors()
        for key in original:
            assert changed[key] != original[key]

        self.edge.unselect()
        after = self._get_colors()
        assert after == original

    def test_select(self):
        event = mock_event('mouseup', 0.25, 0.05, self.fig)
        drawing_graph = make_mock_graph(self.fig)
        original = self._get_colors()
        self.edge.select(event, drawing_graph)
        changed = self._get_colors()

        for key in self.nodes:
            assert changed[key] != original[key]
            assert changed[key] == (1.0, 0.0, 0.0, 1.0)  # red

        assert changed[self.edge] == "red"  # mpl doesn't convert to RGBA?!
        # it might be better in the future to pass that through some MPL
        # func that converts color string to RGBA; the fact that MPL keeps
        # color name in line2d seems like an implementation detail

    def test_update_locations(self):
        for node in self.nodes:
            x, y = node.xy
            node.update_location(x + 0.2, y + 0.2)

        self.edge.update_locations()
        npt.assert_allclose(self.edge.artist.get_xdata(), [0.25, 0.75])
        npt.assert_allclose(self.edge.artist.get_ydata(), [0.25, 0.25])


class TestEventHandler:
    def setup(self):
        self.event_handler = EventHandler(graph=make_mock_graph())
        graph = self.event_handler.graph
        node = graph.nodes["C"]
        edge = graph.edges[graph.nodes["B"], graph.nodes["C"]]
        self.setup_contains = {
            "node": (node, [node]),
            "edge": (edge, [edge]),
            "node+edge": (node, [node, edge]),
            "miss": (None, []),
        }

    def _mock_for_connections(self):
        self.event_handler.on_mousedown = mock.Mock()
        self.event_handler.on_mouseup = mock.Mock()
        self.event_handler.on_drag = mock.Mock()

    @pytest.mark.parametrize('event_type', ['mousedown', 'mouseup', 'drag'])
    def test_connect(self, event_type):
        self._mock_for_connections()
        fig, _ = plt.subplots()
        event = mock_event(event_type, 0.2, 0.2, fig)

        methods = {
            'mousedown': self.event_handler.on_mousedown,
            'mouseup': self.event_handler.on_mouseup,
            'drag': self.event_handler.on_drag,
        }
        should_call = methods[event_type]
        should_not_call = set(methods.values()) - {should_call}
        assert len(self.event_handler.connections) == 0

        self.event_handler.connect(fig.canvas)
        assert len(self.event_handler.connections) == 3

        # check that the event is processed
        fig.canvas.callbacks.process(event.name, event)
        should_call.assert_called_once()
        for method in should_not_call:
            assert not method.called

        plt.close(fig)

    @pytest.mark.parametrize('event_type', ['mousedown', 'mouseup', 'drag'])
    def test_disconnect(self, event_type):
        self._mock_for_connections()
        fig, _ = plt.subplots()
        event = mock_event(event_type, 0.2, 0.2, fig)

        self.event_handler.connect(fig.canvas)  # not quite full isolation
        assert len(self.event_handler.connections) == 3

        self.event_handler.disconnect(fig.canvas)
        assert len(self.event_handler.connections) == 0
        methods = [self.event_handler.on_mousedown,
                   self.event_handler.on_mousedown,
                   self.event_handler.on_drag]

        fig.canvas.callbacks.process(event.name, event)
        for method in methods:
            assert not method.called

        plt.close(fig)

    def _mock_contains(self, mock_objs):
        graph = self.event_handler.graph
        objs = list(graph.nodes.values()) + list(graph.edges.values())
        for obj in objs:
            if obj in mock_objs:
                obj.contains = mock.Mock(return_value=True)
            else:
                obj.contains = mock.Mock(return_value=False)

    @pytest.mark.parametrize('hit', ['node', 'edge', 'node+edge', 'miss'])
    def test_get_event_container_select_node(self, hit):
        expected, contains_event = self.setup_contains[hit]
        expected_count = {
            "node": 3,  # nodes A, B, C
            "edge": 6,  # nodes A, B, C, D; edges AB, BC
            "node+edge": 3,  # nodes A, B, C
            "miss": 7,  # nodes A, B, C, D; edges AB BC, BD
        }[hit]
        self._mock_contains(contains_event)
        event = mock.Mock()
        found = self.event_handler._get_event_container(event)
        assert found is expected
        for container in contains_event:
            if container is not expected:
                assert not container.called

        graph = self.event_handler.graph
        all_objs = list(graph.nodes.values()) + list(graph.edges.values())
        contains_count = sum(obj.contains.called for obj in all_objs)
        assert contains_count == expected_count

    @pytest.mark.parametrize('hit', ['node', 'edge', 'node+edge', 'miss'])
    def test_on_mousedown(self, hit):
        expected, contains_event = self.setup_contains[hit]
        self._mock_contains(contains_event)
        event = mock_event('mousedown', 0.5, 0.5)

        assert self.event_handler.click_location is None
        assert self.event_handler.active is None
        self.event_handler.on_mousedown(event)
        npt.assert_allclose(self.event_handler.click_location, (0.5, 0.5))
        assert self.event_handler.active is expected
        if expected is not None:
            expected.on_mousedown.assert_called_once()

        plt.close(event.canvas.figure)

    @pytest.mark.parametrize('is_active', [True, False])
    def test_on_drag(self, is_active):
        fig, ax = plt.subplots()
        node = self.event_handler.graph.nodes["C"]
        node.artist.axes = ax
        event = mock_event('drag', 0.25, 0.25, fig)
        if is_active:
            self.event_handler.active = node

        self.event_handler.on_drag(event)

        if is_active:
            node.on_drag.assert_called_once()
        else:
            assert not node.on_drag.called

        plt.close(fig)

    @pytest.mark.parametrize('has_selected', [True, False])
    def test_on_mouseup_click_select(self, has_selected):
        # start: mouse hasn't moved, and something is active
        graph = self.event_handler.graph
        edge = graph.edges[graph.nodes["B"], graph.nodes["C"]]
        if has_selected:
            old_selected = graph.edges[graph.nodes["A"], graph.nodes["B"]]
            self.event_handler.selected = old_selected

        self._mock_contains([edge])
        event = mock_event('mouseup', 0.25, 0.25)
        self.event_handler.click_location = (event.xdata, event.ydata)
        self.event_handler.active = edge

        # this should select the active object
        self.event_handler.on_mouseup(event)

        if has_selected:
            old_selected.unselect.assert_called_once()

        edge.select.assert_called_once()
        edge.on_mouseup.assert_called_once()
        assert self.event_handler.selected is edge
        assert self.event_handler.active is None
        assert self.event_handler.click_location is None
        graph.draw.assert_called_once()

        plt.close(event.canvas.figure)

    @pytest.mark.parametrize('has_selected', [True, False])
    def test_on_mouseup_click_not_select(self, has_selected):
        # start: mouse hasn't moved, nothing is active
        graph = self.event_handler.graph
        if has_selected:
            old_selected = graph.edges[graph.nodes["A"], graph.nodes["B"]]
            self.event_handler.selected = old_selected

        event = mock_event('mouseup', 0.25, 0.25)
        self.event_handler.click_location = (event.xdata, event.ydata)

        self.event_handler.on_mouseup(event)

        if has_selected:
            old_selected.unselect.assert_called_once()

        assert self.event_handler.selected is None
        assert self.event_handler.active is None
        assert self.event_handler.click_location is None
        graph.draw.assert_called_once()
        plt.close(event.canvas.figure)

    @pytest.mark.parametrize('has_selected', [True, False])
    def test_on_mouseup_drag(self, has_selected):
        # start: mouse has moved, something is active
        graph = self.event_handler.graph
        edge = graph.edges[graph.nodes["B"], graph.nodes["C"]]
        if has_selected:
            old_selected = graph.edges[graph.nodes["A"], graph.nodes["B"]]
            self.event_handler.selected = old_selected

        event = mock_event('mouseup', 0.25, 0.25)
        self.event_handler.click_location = (0.5, 0.5)
        self.event_handler.active = edge

        self.event_handler.on_mouseup(event)

        if has_selected:
            assert not old_selected.unselect.called

        assert not edge.selected.called
        edge.on_mouseup.assert_called_once()
        assert self.event_handler.active is None
        assert self.event_handler.click_location is None
        graph.draw.assert_called_once()
        plt.close(event.canvas.figure)


class TestGraphDrawing:
    def setup(self):
        self.nx_graph = nx.MultiDiGraph()
        self.nx_graph.add_edges_from([
            ("A", "B", {'data': "AB"}),
            ("B", "C", {'data': "BC"}),
            ("B", "D", {'data': "BD"}),
        ])
        self.graph = GraphDrawing(self.nx_graph, positions={
            "A": (0.0, 0.0), "B": (0.5, 0.0), "C": (0.5, 0.5),
            "D": (-0.1, 0.6)
        })

    def test_init(self):
        # this also tests _register_node, _register_edge
        assert len(self.graph.nodes) == 4
        assert len(self.graph.edges) == 3
        assert len(self.graph.fig.axes) == 1
        assert self.graph.fig.axes[0] is self.graph.ax
        assert len(self.graph.ax.patches) == 4
        assert len(self.graph.ax.lines) == 3

    @pytest.mark.parametrize('node,edges', [
        ("A", [("A", "B")]),
        ("B", [("A", "B"), ("B", "C"), ("B", "D")]),
        ("C", [("B", "C")]),
    ])
    def test_edges_for_node(self, node, edges):
        expected_edges = {self.graph.edges[n1, n2] for n1, n2 in edges}
        assert set(self.graph.edges_for_node(node)) == expected_edges

    def test_get_nodes_extent(self):
        assert self.graph._get_nodes_extent() == (-0.1, 0.6, 0.0, 0.7)

    def test_reset_bounds(self):
        old_xlim = self.graph.ax.get_xlim()
        old_ylim = self.graph.ax.get_ylim()
        self.graph.ax.set_xlim(old_xlim[0] * 2, old_xlim[1] * 2)
        self.graph.ax.set_ylim(old_ylim[0] * 2, old_ylim[1] * 2)
        self.graph.reset_bounds()
        assert self.graph.ax.get_xlim() == old_xlim
        assert self.graph.ax.get_ylim() == old_ylim

    def test_draw(self):
        # just a smoke test; there's really nothing that we can test here
        # other that integration
        self.graph.draw()

