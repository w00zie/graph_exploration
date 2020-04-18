import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import itertools


def draw_graph(path_to_map):
    """This method draws a graph with the use of pyplot.
    The graph has to be in the form of a 3D tensor of shape (2, n, n)
    where n is the number of nodes. The first element of this
    tensor has to be the adjacency matrix and the second one a
    matrix filled with ones on obstacle nodes.
    """
    # Get the adjacency matrix and the obstacles list
    state = np.load(path_to_map)
    adj, obstacles = state[0], state[1].nonzero()[0]

    n_nodes = adj.shape[0]
    G = nx.from_numpy_matrix(adj)
    grid_size = int(np.sqrt(n_nodes))
    pos = np.array(list(itertools.product(range(grid_size),
                                          range(grid_size))))

    valid_nodes = [n for n in range(n_nodes) if n not in obstacles]
    pos = [(y, 9-x) for (x, y) in pos]

    fig = plt.figure(figsize=(7, 7))
    nx.draw_networkx_edges(G, pos=pos, width=3.0)
    nx.draw_networkx_nodes(G, pos=pos,
                           nodelist=valid_nodes,
                           node_color='black',
                           node_size=0)
    nx.draw_networkx_nodes(G, pos=pos,
                           nodelist=obstacles,
                           node_color='r',
                           node_size=800)

    # Other things you might be interested in plotting
    # nx.draw_networkx_labels(G, pos=pos, font_size=10,
    #                        font_family='sans-serif')
    # weights = nx.get_edge_attributes(G,'weight')
    # nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=weights)

    fname = path_to_map.split('/')[-1]
    fname = fname.split('.')[0]
    plt.title(fname)
    plt.show()


if __name__ == '__main__':

    env_dir = 'mazes/5x5/'
    envs = [f for f in os.listdir(env_dir) if f.endswith('.npy')]
    for env in envs:
        draw_graph(env_dir+env)
