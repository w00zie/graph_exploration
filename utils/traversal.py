class GraphTraversal():
    """
    This class handles the animation of the exploration.
    At every frame the agent's action is plotted over the graph.
    """
    def __init__(self, g, pos, valid_nodes, obstacles, traversal):
        """
        Parameters
        ----------
        g : nx.Graph
            The environment.
        pos : List
            List of tuples indicating the space positioning of the nodes.
        valid_nodes : List
        obstacles : List
        traversal : List
            A sequence of nodes to be plotted.
        """
        self.g = g
        self.pos = [(y,9-x) for (x,y) in pos]
        self.valid_nodes = valid_nodes
        self.obstacles = obstacles
        self.traversal = traversal

    def animate(self):
        for i, n in enumerate(self.traversal):
            #clear_output()
            fig, ax = plt.subplots(figsize=(8,4.5))
            nx.draw(self.g, pos=self.pos, node_color='b', node_size=0)
            nx.draw_networkx_nodes(self.g, self.pos, nodelist=self.obstacles,
                                   node_color='r',
                                   node_size=600)
            nx.draw_networkx_nodes(self.g, pos=self.pos,
                                   nodelist=self.traversal[:i],
                                   node_color='g',
                                   node_size=600)
            nx.draw_networkx_nodes(self.g, pos=self.pos, nodelist=[n],
                                   node_color='black', node_size=600)
            nx.draw_networkx_labels(self.g, self.pos)
            plt.show()
            sleep(.1)
