import itertools
import gym
from gym import spaces
import networkx as nx
import torch
from torch_geometric.data import Data
import numpy as np


class Environment(gym.Env):
    """
    The environment (the gym subclassing is not really needed).
    Registers the action from the agent and returns a scalar reward.
    Handles the state management (builds and updates the features).
    """
    #metadata = {'render.modes':[]}
    def __init__(self, path_to_map, fixed_start_node=False):

        # Get the adjacency matrix and the obstacles list
        state = np.load(path_to_map)
        self.adj, self.obstacles = state[0], state[1].nonzero()[0]

        # Construct networkx graph (may be useful)
        self.G = nx.from_numpy_matrix(self.adj)

        self.n_nodes = self.adj.shape[0]
        self.valid_nodes = [n for n in range(self.n_nodes) 
                            if n not in self.obstacles]

        # Construct data (required by torch_geometric)
        self.edge_index = torch.tensor(self.adj.nonzero(), dtype=torch.long)
        self.grid_size = int(np.sqrt(self.n_nodes))
        self.pos = torch.tensor(list(itertools.product(range(self.grid_size), 
                                                       range(self.grid_size))))
        self.data = self.build_data()

        self.fixed_start_node = fixed_start_node
        # Observation space is the discrete space of all nodes
        self.observation_space = spaces.Discrete(self.n_nodes)

        # Action space is (0:Left, 1:Up, 2:Right, 3:Down)
        self.action_space = spaces.Discrete(4)

        # History of explored nodes.
        self.history = []
        return

    def build_data(self):
        node_features = self.init_features()
        edge_features = torch.ones(len(self.G.edges))
        return Data(x=node_features, 
                    edge_index=self.edge_index, 
                    edge_attr=edge_features, 
                    pos=self.pos)

    def get_n_neighs_to_visit(self, X):
        list_of_neighs = [self.get_neigh(node) for node in self.G.nodes]
        visited = [torch.where(X[n, 3] > 0)[0] for n in list_of_neighs]
        n_to_visit = [len(list_of_neighs[i]) - len(visited[i]) for i in 
                      range(self.n_nodes)]
        return torch.tensor(n_to_visit)

    def init_features(self):
        """
        This method initializes the graph features.
        Every node has 7 features:
        (x, y, is_obstacle, n_visits, is_wall, n_neighs_to_visit, curr_occup) where
        0,1 - `x,y,` are x,y coordinates of the node
        2 - `is_obstacle` is a boolean flag indicating whether the node is an obstacle
        3 - `n_visits` the number of visits for the node
        4 - `is_wall` is a boolan flag indicating wheter the node is part of the perimetral wall
        5 - `n_neighs_to_visit` is the number of neighbors yet to be visited
        6 - `curr_occup` is a boolean flag indicating whether the node is occupied by an agent

        Returns
        -------
        X : tensor
            The `num_nodes` times `num_feat` feature matrix.
        """
        num_feat = 7
        X = torch.zeros((self.n_nodes, num_feat), dtype=torch.float)
        # First two features are `(x,y)` coordinates
        X[:,:2] = torch.tensor([x for x in itertools.product(range(self.grid_size), 
                                                             range(self.grid_size))])
        # Third feature is `is_obstacle`
        X[self.obstacles, 2] = 1
        # Fifth feature is `is_wall`
        frontier = [i for i in range(self.n_nodes) if X[i,0] == 0 
                    or X[i,0] == 9 
                    or X[i,1] == 0 
                    or X[i,1] == 9]
        X[frontier, 4] = 1
        # Sixth feature is `n_neighs_to_visit`
        X[:, 5] = self.get_n_neighs_to_visit(X)
        return X

    def get_visited_nodes(self):
        return torch.where(self.data.x[:, 3] > 0)[0].tolist()

    def get_neigh(self, node_idx):
        return list(self.G.neighbors(node_idx))

    def get_history(self):
        return self.history.copy()

    def map_neigh_action(self):
        """
        Returns a dictionary {action: next state}
        that maps actions to future nodes.
        0: Left, 1:Up, 2:Right, 3:Down
        """
        neighs = self.get_neigh(self.current_node)
        action_state = {}
        for n in neighs:
            if n + self.grid_size == self.current_node:
                action_state[1] = n
            elif n - self.grid_size == self.current_node:
                action_state[3] = n
            elif n + 1 == self.current_node:
                action_state[0] = n
            elif n - 1 == self.current_node:
                action_state[2] = n
            else:
                print('Something wrong')
                exit()
        return action_state

    def step(self, action):
        """
        This method executes the action and returns the next state and reward. 
        It also checks if the agent reached its goal, i.e. visit every node.
        """
        action_state = self.map_neigh_action()
        n_nodes_vis = len(self.get_visited_nodes())
        info = {}
        # If action is valid (agent does not want to exit the map)
        if action in action_state.keys():
            next_node = action_state[action]
            if next_node in self.valid_nodes:
                # Execute the transition to the next state
                self.current_node = next_node
                # Update currently occupied
                self.data.x[:, 6] = 0
                self.data.x[self.current_node, 6] = 1
                # Reward function
                if self.current_node in self.get_visited_nodes():
                    reward = -0.1*self.data.x[self.current_node, 3]
                else:
                    reward = 0.1
            else:
                reward = -0.5 - 0.1*self.data.x[self.current_node, 3]
        else:
            reward = -0.5 - 0.1*self.data.x[self.current_node, 3]

        # Update visit count
        self.data.x[self.current_node, 3] += 1

        # Update number of neighbors to be visited
        self.data.x[:, 5] = self.get_n_neighs_to_visit(self.data.x)

        # First r_t function, seems not to produce any result during training
        #reward = (len(self.get_visited_nodes()) - n_nodes_vis)/len(self.valid_nodes)

        # Log into history
        self.history.append(self.current_node)

        # Check if agent visited every node
        done = all(self.data.x[self.valid_nodes, 3] > 0)
        if done:
            reward += 1.

        return self.data.clone(), reward, done, info

    def reset(self):
        """
        Randomly initializes the state to a valid node
        """
        # Re-init torch_geometric `Data` object
        self.data = self.build_data()

        # Randomly select the first state from set of valid nodes
        self.current_node = np.random.choice(self.valid_nodes)
        if self.fixed_start_node:
            self.current_node = 0

        # Log into history
        self.history = []
        self.history.append(self.current_node)

        # Update visit count
        self.data.x[self.current_node, 3] += 1

        # Set currently occupied
        self.data.x[self.current_node, 6] = 1

        return self.data.clone()
