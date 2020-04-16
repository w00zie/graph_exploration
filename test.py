from env import Environment
from model import GNN
from agent import Agent
import numpy as np
import torch
from torch_geometric.data import Batch
import os
from traversal import GraphTraversal


def purely_random(path_to_map, num_episodes=50, num_steps=25):
    """
    This method tests an environment in a purely random fashion.
    Returns a list, whose len is num_episodes, with percentages
    of visited nodes.

    Parameters
    ----------
    path_to_map : string
        absolute path to the environment.
    num_episodes : int
        Maximum number of episodes of the test (default 50)
    num_steps : int
        Number of steps for every episode (default 25)

    Returns
    -------
    pctg_vis_nodes : List
        List of percentages of visited node in every one of the
        `num_episodes` episodes.
    """
    env = Environment(path_to_map=path_to_map)

    val_nodes = len(env.valid_nodes)
    vis_nodes = []

    for _ in range(num_episodes):
        state = env.reset()
        for _ in range(num_steps):
            action = np.random.choice(range(4))
            next_state, reward, done, _ = env.step(action)
            state = next_state.clone()
        vis_nodes.append(env.get_history())

    pctg_vis_nodes = [100*len(set(v))/val_nodes for v in vis_nodes]

    return pctg_vis_nodes


@torch.no_grad()
def single_graph_test(path_to_map, path_to_model, 
                      num_episodes=50, num_steps=25):
    """
    This method tests an environment following the action
    proposed by the network only.

    Parameters
    ----------
    path_to_map : string
        absolute path to the environment.
    num_episodes : int
        Maximum number of episodes of the test (default 50)
    num_steps : int
        Number of steps for every episode (default 25)

    Returns
    -------
    pctg_vis_nodes : List
        List of percentages of visited node in every one of the
        `num_episodes` episodes.
    """
    test_env = Environment(path_to_map=path_to_map)
    model = GNN(num_features=7, num_actions=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path_to_model, map_location=device))
    model.eval()

    val_nodes = len(test_env.valid_nodes)
    vis_nodes = []

    for ep in range(num_episodes):
        state = test_env.reset()
        for step in range(num_steps):
            state = Batch.from_data_list([state])
            action = torch.argmax(model(state)).item()
            next_state, reward, done, _ = test_env.step(action)
            state = next_state.clone()
        vis_nodes.append(test_env.get_history())

    pctg_vis_nodes = [100*len(set(v))/val_nodes for v in vis_nodes]
    #gt = GraphTraversal(test_env.G, test_env.pos, test_env.valid_nodes, 
    #                    test_env.obstacles, vis_nodes[best_run])
    #gt.animate()

    return pctg_vis_nodes


def test_random_directory(test_dir='/content/test_set', 
                          num_episodes=50, num_steps=25):
    """
    This method tests a whole `test_dir` directory going purely random.
    Calculates mean and std for every file inside the `test_dir` and 
    calculates a final mean and std for the whole dir.

    Parameters
    ----------
    path_to_model : string
        Path to the trained model (usually a .pt).
    test_dir : string
        Path to the directory containing the testing environments.
    """
    verb = True

    means, variances = [], []
    test_files = sorted([filename for filename in os.listdir(test_dir) if 
                  filename.endswith('.npy')])
    print('\n|      Map     | Mean |  Std | Best Run |\n|:------------:|:----:|:----:|:--------:|')
    for fnam in test_files:
        pctg_vis_nodes = purely_random(test_dir+'/'+fnam, 
                                       num_episodes=num_episodes, 
                                       num_steps=num_steps)

        mean, var = np.mean(pctg_vis_nodes), np.var(pctg_vis_nodes)

        # This is the environment on which I've trained my models. 
        # Change this if you did not do that.
        if '5x5_0' not in fnam:
            means.append(mean)
            variances.append(var)

        if verb:
            nam = fnam.split('/')[-1].split('.')[0]
            best_run = pctg_vis_nodes.index(max(pctg_vis_nodes))
            print('| `{}` | {} | {} |   {}   |'.
                  format(nam,
                         round(mean,1), 
                         round(np.sqrt(var),1),
                         round(pctg_vis_nodes[best_run],1)))
    mean = np.mean(means)
    var = np.sum(variances) / (len(variances)**2)
    print('\n\tFINAL : MEAN = {}  STD = {}'.
          format(round(mean, 2), round(np.sqrt(var),2)))


def test_model_directory(path_to_model, test_dir='/content/test_set', 
                         num_episodes=50, num_steps=25):
    """
    This method tests a whole `test_dir` directory with the use of the
    trained model. Calculates mean and std for every file inside the
    `test_dir` and calculates a final mean and std for the whole dir.

    Parameters
    ----------
    path_to_model : string
        Path to the trained model (usually a .pt).
    test_dir : string
        Path to the directory containing the testing environments.
    """
    verb = True

    means, variances = [], []
    test_files = sorted([filename for filename in os.listdir(test_dir) if 
                  filename.endswith('.npy')])
    print('\n|      Map     | Mean |  Std | Best Run |\n|:------------:|:----:|:----:|:--------:|')
    for fnam in test_files:
        pctg_vis_nodes = single_graph_test(test_dir+'/'+fnam, path_to_model, 
                                           num_episodes=num_episodes, 
                                           num_steps=num_steps)

        mean, var = np.mean(pctg_vis_nodes), np.var(pctg_vis_nodes)

        # This is the environment on which I've trained my models. 
        # Change this if you did not do that.
        if '5x5_0' not in fnam:
            means.append(mean)
            variances.append(var)

        if verb:
            nam = fnam.split('/')[-1].split('.')[0]
            best_run = pctg_vis_nodes.index(max(pctg_vis_nodes))
            print('| `{}` | {} | {} |   {}   |'.
                  format(nam,
                         round(mean,1), 
                         round(np.sqrt(var),1),
                         round(pctg_vis_nodes[best_run],1)))

    mean = np.mean(means)
    var = np.sum(variances) / (len(variances)**2)

    print('\n\tFINAL : MEAN = {}  STD = {}'.
          format(round(mean, 2), round(np.sqrt(var),2)))


if __name__ == '__main__':
    """
    This main test a `test_dir` directory, containing several environments,
    in two manners: the first one going purely random, the second one
    using only the trained model. If several models are present inside the
    `models/` directory this test will go through everyone of them.
    """

    num_episodes = 30
    num_steps = 25
    test_dir = 'mazes/5x5'
    path_to_models = 'models/'

    print('Testing {} with {} episodes of {} steps'.format(test_dir,
                                                           num_episodes,
                                                           num_steps))

    print('\nRANDOM')
    test_random_directory(test_dir=test_dir,
                        num_episodes=num_episodes, num_steps=num_steps)
    print(40*'-')

    models = sorted([path_to_models+filename for filename in
                    os.listdir(path_to_models) if filename.endswith('.pt')])
    for model_fname in models:
        print('\nMODEL {}'.format(model_fname))
        test_model_directory(model_fname, test_dir=test_dir,
                            num_episodes=num_episodes, num_steps=num_steps)
        print(40*'-')
