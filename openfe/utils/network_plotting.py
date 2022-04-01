# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""
Generic tools for plotting networks. Interfaces NetworkX and matplotlib.

Create subclasses of ``Node``, ``Edge``, and ``GraphDrawing`` to customize
behavior how the graph is visualized or what happens on interactive events.
"""

from __future__ import annotations

import itertools
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from typing import Dict, List, Tuple, Optional, Any, Union, cast
from openfe.utils.custom_typing import (
    MPL_MouseEvent, MPL_FigureCanvasBase, MPL_Axes, TypeAlias
)

ClickLocation: TypeAlias = Tuple[Tuple[float, float], Tuple[Any, Any]]


class Node:
    """Node in the GraphDrawing network.

    This connects a node in the NetworkX graph to the matplotlib artist.
    This is the only object that should directly use the matplotlib artist
    for this node. This acts as an adapter class, allowing different artists
    to be used, as well as enabling different functionalities.
    """
    # TODO: someday it might be good to separate the artist adapter from the
    # functionality on select, etc.
    draggable = True
    pickable = False
    lock = None  # lock used while dragging; only one Node dragged at a time

    def __init__(self, node, x: float, y: float, dx=0.1, dy=0.1):
        self.node = node
        self.dx = dx
        self.dy = dx
        self.artist = self._make_artist(x, y, dx, dy)
        self.picked = False
        self.press: Optional[ClickLocation] = None

    def _make_artist(self, x, y, dx, dy):
        return Rectangle((x, y), dx, dy, color='blue')

    def register_artist(self, ax: MPL_Axes):
        """Register this node's artist with the matplotlib Axes"""
        ax.add_patch(self.artist)

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """extent of this node in matplotlib data coordinates"""
        bounds = self.artist.get_bbox().bounds
        return (bounds[0], bounds[0] + bounds[2],
                bounds[1], bounds[1] + bounds[3])

    @property
    def xy(self) -> Tuple[float, float]:
        """lower left (matplotlib data coordinates) position of this node"""
        return self.artist.xy

    def select(self, event: MPL_MouseEvent, graph: GraphDrawing):  # -no-cov-
        """Set this node to its state when it is selected (clicked on)"""
        return

    def unselect(self):
        """Reset this node to its standard, unselected visualization"""
        self.artist.set(color='blue')

    def edge_select(self, edge: Edge):
        """Change node visualization when one of its edges is selected"""
        self.artist.set(color='red')

    def update_location(self, x: float, y: float):
        """Update the location of the underlying artist"""
        self.artist.set(x=x, y=y)

    # note: much the stuff below is based on the "Draggable rectangle"
    # exercise at:
    # https://matplotlib.org/stable/users/explain/event_handling.html#draggable-rectangle-exercise
    def contains(self, event: MPL_MouseEvent) -> bool:
        """Report whether this object contains the given event"""
        return self.artist.contains(event)[0]

    def on_mousedown(self, event: MPL_MouseEvent, graph: GraphDrawing):
        """Handle mousedown event (button_press_event)"""
        # these early returns probably won't be called in practice, since
        # the event handler should only call this method when those
        # conditions are met; still, defensive programming!
        if event.inaxes != self.artist.axes:
            return

        if not self.contains(event):
            return

        # record the original click location; lock that we're the only
        # object being dragged
        self.press = self.xy, (event.xdata, event.ydata)
        Node.lock = self
        # TODO: blitting

    def on_drag(self, event: MPL_MouseEvent, graph: GraphDrawing):
        """Handle dragging this node"""
        if event.inaxes != self.artist.axes or Node.lock is not self:
            return

        if self.press:
            (x0, y0), (xpress, ypress) = self.press
        else:
            # this should be impossible in practice, but mypy needed the
            # explicit check so it didn't unpack None
            raise RuntimeError("Can't drag until mouse down!")

        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.update_location(x0 + dx, y0 + dy)

        # TODO: this might be cached on mousedown
        edges = graph.edges_for_node(self.node)
        for edge in edges:
            edge.update_locations()

        # TODO: blitting
        self.artist.figure.canvas.draw()

    def on_mouseup(self, event: MPL_MouseEvent, graph: GraphDrawing):
        """Handle mouseup event (button_release_event)"""
        self.press = None
        Node.lock = None
        # TODO: blitting
        self.artist.figure.canvas.draw()


class Edge:
    """Edge in the GraphDrawing network.

    This connects an edge in the NetworkX graph to the matplotlib artist. In
    addition to the edge data, this needs to know the two GraphDrawing
    ``Node`` instances associated with this edge.

    Parameters
    ----------
    node_artist1, node_artist2 : :class:`.Node`
        GraphDrawing nodes for this edge
    data : Dict
        data dictionary for this edge
    """
    pickable = True

    def __init__(self, node_artist1: Node, node_artist2: Node, data: Dict):
        self.data = data
        self.node_artists = [node_artist1, node_artist2]
        self.artist = self._make_artist(node_artist1, node_artist2, data)
        self.picked = False

    def _make_artist(self, node_artist1: Node, node_artist2: Node,
                     data: Dict) -> Any:
        xs, ys = self._edge_xs_ys(node_artist1, node_artist2)
        return Line2D(xs, ys, color='black', picker=True, zorder=-1)

    def register_artist(self, ax: MPL_Axes):
        """Register this edge's artist with the matplotlib Axes"""
        ax.add_line(self.artist)

    def contains(self, event: MPL_MouseEvent) -> bool:
        """Report whether this object contains the given event"""
        return self.artist.contains(event)[0]

    @staticmethod
    def _edge_xs_ys(node1: Node, node2: Node):
        def get_midpoint(node):
            x0, x1, y0, y1 = node.extent
            return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

        midpt1 = get_midpoint(node1)
        midpt2 = get_midpoint(node2)

        xs, ys = list(zip(*[midpt1, midpt2]))
        return xs, ys

    def on_mousedown(self, event: MPL_MouseEvent, graph: GraphDrawing):
        """Handle mousedown event (button_press_event)"""
        return  # -no-cov-

    def on_drag(self, event: MPL_MouseEvent, graph: GraphDrawing):
        """Handle drag event"""
        return  # -no-cov-

    def on_mouseup(self, event: MPL_MouseEvent, graph: GraphDrawing):
        """Handle mouseup event (button_release_event)"""
        return  # -no-cov-

    def unselect(self):
        """Reset this edge to its standard, unselected visualization"""
        self.artist.set(color='black')
        for node_artist in self.node_artists:
            node_artist.unselect()
        self.picked = False

    def select(self, event: MPL_MouseEvent, graph: GraphDrawing):
        """Mark this edge as selected, update visualization"""
        self.artist.set(color='red')
        for artist in self.node_artists:
            artist.edge_select(self)
        self.picked = True
        return True

    def update_locations(self):
        """Update the location of this edge based on node locations"""
        xs, ys = self._edge_xs_ys(*self.node_artists)
        self.artist.set(xdata=xs, ydata=ys)


class EventHandler:
    """Pass event information to nodes/edges.

    This is the single place where we connect to the matplotlib event
    system. This object receives matplotlib events and delegates to the
    appropriate node or edge.

    Parameters
    ----------
    graph : GraphDrawing
        the graph drawing that we're handling events for

    Attributes
    ----------
    active : Optional[Union[Node, Edge]]
        Object activated by a mousedown event, or None if either no object
        activated by mousedown or if mouse is not currently pressed. This is
        primarily used to handle drag events.
    selected : Optional[Union[Node, Edge]]
        Object selected by a mouse click (after mouse is up), or None if no
        object has been selected in the graph.
    click_location : Optional[Tuple[int, int]]
        Cached location of the mousedown event, or None if mouse is up
    connections : List[int]
        list of IDs for connections to matplotlib canvas
    """
    def __init__(self, graph: GraphDrawing):
        self.graph = graph
        self.active: Optional[Union[Node, Edge]] = None
        self.selected: Optional[Union[Node, Edge]] = None
        self.click_location: Optional[Tuple[int, int]] = None
        self.connections: List[int] = []

    def connect(self, canvas: MPL_FigureCanvasBase):
        """Connect our methods to events in the matplotlib canvas"""
        self.connections.extend([
            canvas.mpl_connect('button_press_event', self.on_mousedown),
            canvas.mpl_connect('motion_notify_event', self.on_drag),
            canvas.mpl_connect('button_release_event', self.on_mouseup),
        ])

    def disconnect(self, canvas: MPL_FigureCanvasBase):
        """Disconnect all connections to the canvas."""
        for cid in self.connections:
            canvas.mpl_disconnect(cid)
        self.connections = []

    def _get_event_container(self, event: MPL_MouseEvent):
        """Identify which object should process an event.

        Note that we prefer nodes to edges: If you click somewhere that
        could be a node or an edge, it is interpreted as clicking on the
        node.
        """
        containers = itertools.chain(self.graph.nodes.values(),
                                     self.graph.edges.values())
        for container in containers:
            if container.contains(event):
                break
        else:
            container = None

        return container

    def on_mousedown(self, event: MPL_MouseEvent):
        """Handle mousedown event (button_press_event)"""
        self.click_location = event.xdata, event.ydata
        container = self._get_event_container(event)
        if container is None:
            return

        # cast because mypy can't tell that we did early return if None
        self.active = cast(Union[Node, Edge], container)
        self.active.on_mousedown(event, self.graph)

    def on_drag(self, event: MPL_MouseEvent):
        """Handle dragging"""
        if not self.active or event.inaxes != self.active.artist.axes:
            return

        self.active.on_drag(event, self.graph)

    def on_mouseup(self, event: MPL_MouseEvent):
        """Handle mouseup event (button_release_event)"""
        if self.click_location == (event.xdata, event.ydata):
            # mouse hasn't moved; call it a click
            # first unselect whatever was previously selected
            if self.selected:
                self.selected.unselect()

            # if it is a click and the active object contains it, select it;
            # otherwise unset selection
            if self.active and self.active.contains(event):
                self.active.select(event, self.graph)
                self.selected = self.active
            else:
                self.selected = None

        if self.active:
            self.active.on_mouseup(event, self.graph)

        self.active = None
        self.click_location = None
        self.graph.draw()


class GraphDrawing:
    """
    Base class for drawing networks with matplotlib.

    Connects to the matplotlib figure and to the underlying NetworkX graph.

    Typical use will require a subclass with custom values of ``NodeCls``
    and ``EdgeCls`` to handle the specific visualization.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        NetworkX graph with information in nodes and edges to be drawn
    positions : Optional[Dict[Any, Tuple[float, float]]]
        mapping of node to position
    """
    NodeCls = Node
    EdgeCls = Edge

    def __init__(self, graph: nx.Graph, positions=None, ax=None):
        # TODO: use scale to scale up the positions?
        self.event_handler = EventHandler(self)
        self.graph = graph
        self.nodes: Dict[Node, Any] = {}
        self.edges: Dict[Tuple[Node, Node], Any] = {}

        if positions is None:
            positions = nx.spring_layout(self.graph)

        was_interactive = plt.isinteractive()
        plt.ioff()
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
        else:
            self.fig, self.ax = ax.figure, ax

        for node, pos in positions.items():
            self._register_node(node, pos)

        self.fig.canvas.draw()  # required to get renderer
        for edge in graph.edges(data=True):
            self._register_edge(edge)

        self.reset_bounds()
        self.ax.set_aspect(1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        if was_interactive:
            plt.ion()  # -no-cov-

        self.event_handler.connect(self.fig.canvas)

    def _ipython_display_(self):  # -no-cov-
        return self.fig

    def edges_for_node(self, node: Node) -> List[Edge]:
        """List of edges for the given node"""
        edges = (list(self.graph.in_edges(node))
                 + list(self.graph.out_edges(node)))
        return [self.edges[edge] for edge in edges]

    def _get_nodes_extent(self):
        """Find the extent of all nodes (used in setting bounds)"""
        min_xs, max_xs, min_ys, max_ys = zip(*(
            node.extent for node in self.nodes.values()
        ))
        return min(min_xs), max(max_xs), min(min_ys), max(max_ys)

    def reset_bounds(self):
        """Set the bounds of the matplotlib Axes to include all nodes"""
        min_x, max_x, min_y, max_y = self._get_nodes_extent()
        self.ax.set_xlim(min_x, max_x)
        self.ax.set_ylim(min_y, max_y)

    def draw(self):
        """Draw the current canvas"""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _register_node(self, node: Any, position: Tuple[float, float]):
        """Create and register ``Node`` from NetworkX node and position"""
        if node in self.nodes:
            raise RuntimeError("node provided multiple times")

        draw_node = self.NodeCls(node, *position)
        self.nodes[node] = draw_node
        draw_node.register_artist(self.ax)

    def _register_edge(self, edge: Tuple[Node, Node, Dict]):
        """Create and register ``Edge`` from NetworkX edge information"""
        node1, node2, data = edge
        draw_edge = self.EdgeCls(self.nodes[node1], self.nodes[node2], data)
        self.edges[(node1, node2)] = draw_edge
        draw_edge.register_artist(self.ax)
