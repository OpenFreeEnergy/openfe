import io
import matplotlib
from rdkit import Chem

from network_plotting import GraphDrawing, Node, Edge
from openfe.utils.visualization import draw_one_molecule_mapping


class LigandEdge(Edge):
    def __init__(self, node_artist1, node_artist2, data):
        super().__init__(node_artist1, node_artist2, data)
        self.left_image = None
        self.right_image = None

    def _draw_mapped_molecule(self, extent, mol1, mol2, mol1_to_mol2):
        # create the image in a format matplotlib can handle
        # TODO: use a custom draw2d object; figure size from transforms
        img_bytes = draw_one_molecule_mapping(mol1_to_mol2,
                                              mol1.to_rdkit(),
                                              mol2.to_rdkit())
        # from openfe.utils.visualization import draw_unhighlighted_molecule
        # img_bytes = draw_unhighlighted_molecule(mol1.to_rdkit())
        img_filelike = io.BytesIO(img_bytes)
        img_data = matplotlib.pyplot.imread(img_filelike)


        # create BboxImage
        ax = self.artist.axes
        x0, x1, y0, y1 = extent
        # bounds = (x0, y0, x1 - x0, y1 - y0)
        # bounds = (0.2, 0.2, 0.3, 0.3)
        # bbox0 = matplotlib.transforms.Bbox.from_bounds(*bounds)
        # bbox = matplotlib.transforms.TransformedBbox(bbox0, ax.transAxes)
        # im = matplotlib.image.BboxImage(bbox)

        im = matplotlib.image.AxesImage(ax, extent=extent, zorder=10)

        # set image data and register
        im.set_data(img_data)
        ax.add_artist(im)
        return im

    def select(self, event, graph):
        super().select(event, graph)
        mapping = self.data['object']

        # figure out which node is to the left and which to the right
        xs = [node.xy[0] for node in self.node_artists]
        if xs[0] <= xs[1]:
            left = mapping.mol1
            right =  mapping.mol2
            left_to_right = mapping.mol1_to_mol2
        else:
            left = mapping.mol2
            right = mapping.mol1
            left_to_right = {v: k for k, v in mapping.mol1_to_mol2.items()}

        right_to_left = {v: k for k, v in left_to_right.items()}

        # figure out the extent for left and right
        x0, x1 = self.artist.axes.get_xlim()
        dx = x1 - x0
        left_x0, left_x1 = 0.05 * dx + x0, 0.45 * dx + x0
        right_x0, right_x1 = 0.55 * dx + x0, 0.95 * dx + x0
        y0, y1 = self.artist.axes.get_ylim()
        dy = y1 - y0
        y_bottom, y_top = 0.5 * dx + y0, 0.9 * dx + y0

        left_extent = (left_x0, left_x1, y_bottom, y_top)
        right_extent = (right_x0, right_x1, y_bottom, y_top)

        # add the images
        self.left_image = self._draw_mapped_molecule(left_extent,
                                                     left,
                                                     right,
                                                     left_to_right)
        self.right_image = self._draw_mapped_molecule(right_extent,
                                                      right,
                                                      left,
                                                      right_to_left)
        graph.fig.canvas.draw()

    def set_standard(self):
        super().set_standard()
        for artist in [self.left_image, self.right_image]:
            if artist is not None:
                artist.remove()

        self.left_image = None
        self.right_image = None


class LigandNetworkDrawing(GraphDrawing):
    NodeCls = Node
    EdgeCls = LigandEdge


def plot_network(network):
    return LigandNetworkDrawing(network.graph).fig


if __name__ == "__main__":
    import sys
    import openfe.setup
    matplotlib.use("TkAgg")  # MacOS only works Python <3.8 ?!
    with open(sys.argv[1], mode='r') as f:
        graphml = f.read()

    network = openfe.setup.Network.from_graphml(graphml)
    fig = plot_network(network)
    matplotlib.pyplot.show()

