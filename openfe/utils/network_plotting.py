import functools
import itertools
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


class Node:
    draggable = True
    pickable = False
    lock = None  # lock used while dragging; only one Node dragged at a time

    def __init__(self, node, x, y, dx=0.1, dy=0.1):
        self.node = node
        self.dx = dx
        self.dy = dx
        self.artist = self._make_artist(x, y, dx, dy)
        self.picked = False
        self.press = None

    def _make_artist(self, x, y, dx, dy):
        # TODO: this should be abstract
        return Rectangle((x, y), dx, dy, color='blue')

    def register_artist(self, ax):
        # TODO: this might be abstract
        ax.add_patch(self.artist)

    @property
    def extent(self):
        bounds = self.artist.get_bbox().bounds
        return [[bounds[0], bounds[0] + bounds[2]],
                [bounds[1], bounds[1] + bounds[3]]]

    @property
    def xy(self):
        return self.artist.xy

    def set_standard(self):
        # TODO: this should be abstract
        self.artist.set(color='blue')

    def update_location(self, x, y):
        self.artist.set(x=x, y=y)

    def select(self, event, graph):
        return

    # note: much the stuff below is based on the "Draggable rectangle"
    # exercise at:
    # https://matplotlib.org/stable/users/explain/event_handling.html#draggable-rectangle-exercise
    def contains(self, event):
        return self.artist.contains(event)[0]

    def on_mousedown(self, event, graph):
        if event.inaxes != self.artist.axes:
            return

        if not self.artist.contains(event):
            return

        self.press = self.xy, (event.xdata, event.ydata)
        Node.lock = self
        # TODO: blitting

    def on_drag(self, event, graph):
        if event.inaxes != self.artist.axes or Node.lock is not self:
            return
        (x0, y0), (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.update_location(x0 + dx, y0 + dy)

        edges = graph.edges_for_node(self.node)
        for edge in edges:
            edge.update_locations()

        # TODO: blitting
        self.artist.figure.canvas.draw()

    def on_mouseup(self, event, graph):
        self.press = None
        # TODO: blitting
        self.artist.figure.canvas.draw()


class Edge:
    pickable = True
    def __init__(self, node_artist1, node_artist2, data):
        self.data = data
        self.node_artists = [node_artist1, node_artist2]
        self.artist = self._make_artist(node_artist1, node_artist2, data)
        self.picked = False

    def _make_artist(self, node_artist1, node_artist2, data):
        xs, ys = self._edge_xs_ys(node_artist1, node_artist2)
        return Line2D(xs, ys, color='black', picker=True, zorder=-1)

    def register_artist(self, ax):
        ax.add_line(self.artist)

    def contains(self, event):
        return self.artist.contains(event)[0]

    @staticmethod
    def _edge_xs_ys(node1, node2):
        def get_midpoint(node):
            (x0, x1), (y0, y1) = node.extent
            return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

        midpt1 = get_midpoint(node1)
        midpt2 = get_midpoint(node2)

        xs, ys = list(zip(*[midpt1, midpt2]))
        return xs, ys

    def on_mousedown(self, event, graph):
        return

    def on_drag(self, event, graph):
        return

    def on_mouseup(self, event, graph):
        return

    def set_standard(self):
        # TODO: this should be abstract
        self.artist.set(color='black')
        for node_artist in self.node_artists:
            node_artist.set_standard()
        self.picked = False

    def select(self, event, graph):
        # TODO; this should be abstract
        node_artists = [n.artist for n in self.node_artists]
        for artist in node_artists + [self.artist]:
            artist.set(color='red')
        self.picked = True
        return True

    def update_locations(self):
        xs, ys = self._edge_xs_ys(*self.node_artists)
        self.artist.set(xdata=xs, ydata=ys)


class EventHandler:
    """
    Logic for handling object selection and dragging. Connects to the
    matplotlib event system.
    """
    def __init__(self, graph):
        self.graph = graph
        self.active = None
        self.selected = None
        self.click_location = None
        self.connections = []

    def connect(self, canvas):
        self.connections.extend([
            canvas.mpl_connect('button_press_event', self.on_mousedown),
            canvas.mpl_connect('motion_notify_event', self.on_drag),
            canvas.mpl_connect('button_release_event', self.on_mouseup),
        ])

    def disconnect(self, canvas):
        for cid in self.connections:
            canvas.mpl_disconnect(cid)

    def _get_event_container(self, event):
        container = None
        containers = itertools.chain(self.graph.nodes.values(),
                                     self.graph.edges.values())
        for container in containers:
            if container.contains(event):
                break

        return container

    def on_mousedown(self, event):
        self.click_location = event.xdata, event.ydata
        container = self._get_event_container(event)
        if not container:
            return

        self.active = container
        self.active.on_mousedown(event, self.graph)

    def on_drag(self, event):
        if not self.active or event.inaxes != self.active.artist.axes:
            return

        self.active.on_drag(event, self.graph)

    def on_mouseup(self, event):
        if self.click_location == (event.xdata, event.ydata):
            # mouse hasn't moved; call it a click
            # first unselect whatever was previously selected
            if self.selected:
                self.selected.set_standard()

            # if it is a click and the active object contains it, select it;
            # otherwise unset selection
            if self.active.contains(event):
                self.active.select(event, self.graph)
                self.selected = self.active
            else:
                self.selected = None

        self.active.on_mouseup(event, self.graph)
        self.active = None
        self.graph.draw()


class GraphDrawing:
    """
    Central drawing class. Connects to the matplotlib figure and to the
    underlying NetworkX graph.
    """
    NodeCls = Node
    EdgeCls = Edge
    def __init__(self, graph, positions=None):
        # TODO: use scale to scale up the positions?
        self.event_handler = EventHandler(self)
        self.graph = graph
        self.nodes = {}
        self.edges = {}

        if positions is None:
            positions = nx.spring_layout(self.graph)

        was_interactive = plt.isinteractive()
        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(8,8))
        for node, pos in positions.items():
            self._register_node(node, pos)

        self.fig.canvas.draw()
        for edge in graph.edges(data=True):
            self._register_edge(edge)

        self.reset_bounds()
        self.ax.set_aspect(1)
        if was_interactive:
            plt.ion()

        self.event_handler.connect(self.fig.canvas)

    def _ipython_display_(self):
        return self.fig

    def edges_for_node(self, node):
        # return list(self.graph.edges(node))
        edges = (list(self.graph.in_edges(node))
                 + list(self.graph.out_edges(node)))
        return [self.edges[edge] for edge in edges]

    def _get_nodes_extent(self):
        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")
        for node in self.nodes.values():
            (lo_x, hi_x), (lo_y, hi_y) = node.extent
            min_x = min([lo_x, min_x])
            max_x = max([hi_x, max_x])
            min_y = min([lo_y, min_y])
            max_y = max([hi_y, max_y])
        return [min_x, max_x], [min_y, max_y]

    def reset_bounds(self):
        (min_x, max_x), (min_y, max_y) = self._get_nodes_extent()
        self.ax.set_xlim(min_x, max_x)
        self.ax.set_ylim(min_y, max_y)

    def draw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _register_node(self, node, position):
        if node in self.nodes:
            raise RuntimeError("node provided multiple times")

        draw_node = self.NodeCls(node, *position)
        self.nodes[node] = draw_node
        draw_node.register_artist(self.ax)

    def _register_edge(self, edge):
        node1, node2, data = edge
        draw_edge = self.EdgeCls(self.nodes[node1], self.nodes[node2], data)
        self.edges[(node1, node2)] = draw_edge
        draw_edge.register_artist(self.ax)
