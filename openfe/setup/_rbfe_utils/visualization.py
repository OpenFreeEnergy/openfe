"""
Utility functions for visualisation
"""
import networkx as nx
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
from urllib import parse
from networkx.readwrite import cytoscape_data
import cyjupyter
from cyjupyter import Cytoscape


def draw_radial_network(network):
    """
    This is based on iwatobipen's awesome demo: https://iwatobipen.wordpress.com/2020/03/30/draw-scaffold-tree-as-network-with-molecular-image-rdkit-cytoscape/

    TODO
    ----
    This is pretty much a pseudo-hardcoded radial network viz, I'll need some
    input on fixing the edge setting at some point.
    """

    def image(rdmol):
        drawer = rdMolDraw2D.MolDraw2DSVG(690, 400)
        AllChem.Compute2DCoords(rdmol)
        drawer.DrawMolecule(rdmol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace("svg:", "")
        impath = 'data:image/svg+xml;charset=utf-8,' + parse.quote(svg,
                                                                   safe="")
        return impath

    # get index of the benzene
    molecules = {}
    for edge in network.edges:
        def add_to_dict(dicter, name):
            if name in molecules:
                molecules[name] += 1
            else:
                molecules[name] = 1
        add_to_dict(molecules, edge.molA.name)
        add_to_dict(molecules, edge.molB.name)

    central_ligand = max(molecules, key=molecules.get)
    print("central ligand is: ", central_ligand)

    for i, node in enumerate(network.nodes):
        if node.name == central_ligand:
            central_index = i

    # create a new graph based on the input network
    g = nx.graph.Graph()
    for idx, node in enumerate(network.nodes):
        g.add_node(idx, smiles=node.smiles, img=image(node.to_rdkit()),
                   hac=node.to_rdkit().GetNumAtoms())
    for i, edge in enumerate(g.nodes):
        if i != central_ligand:
            g.add_edge(central_index, i)

    cy_g = cytoscape_data(g)
    stobj = [
      {'style': [{'css': {
          'background-color': 'white',
          'shape': 'circle',
          'width': 600,
          'height': 400,
          'border-color': 'rgb(0,0,0)',
          'border-opacity': 1.0,
          'border-width': 0.0,
          'color': '#4579e8',
          'background-image': 'data(img)',
          'background-fit': 'contain'
                        },
       'selector': 'node'},
           {'css': {'width': 20.0,}, 'selector': 'edge'}],}]
    cyobj = Cytoscape(data=cy_g, visual_style=stobj[0]['style'],
                      layout_name='breadthfirst')
    return cyobj
