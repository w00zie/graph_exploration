from datetime import datetime
import torch
from agent import Agent
from env import Environment
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    """
    This method handles the training phase.

    Parameters
    ----------
    env : Environment
        The graph that the agent wants to explore.
    agent : Agent
    max_episodes : int
        The number of epochs (episodes) for the training phase.
    max_spets : int
        The number of steps for every episode.
    batch_size : int
        Batch size for training with SGD.
    """
    writer = SummaryWriter('runs/{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    # Decaying epsilon for epsilon-greedy action selection
    epsilon = np.exp(-np.linspace(0.1, 3, num=max_episodes))
    # Other epsilon strategies
    #epsilon = 1/(1+np.exp(np.linspace(-2.5, 4, num=max_episodes)))
    #epsilon = np.linspace(0.95, 0.05, num=max_episodes)
    #epsilon = max_episodes*[0.2]
    wins = 0
    LOG_EVERY = 10

    for episode in range(max_episodes):
        # Reset state and reward for every episode
        reward_exploit = []
        state = env.reset()
        episode_reward = 0
        n_nodes_visited = []
        exploit_actions = [0, 0, 0, 0]
        for step in range(max_steps):

            # Select the action
            action, exploit_flag = agent.get_action(state, eps=epsilon[episode])

            # Get the reward and the next state
            next_state, reward, done, _ = env.step(action)

            # Push experience into buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Increment the current episode reward
            episode_reward += reward

            # Log number of nodes visited
            n_nodes_visited.append(len(env.get_visited_nodes()))

            # Log reward when agent is exploiting
            if exploit_flag:
                reward_exploit.append(reward)
                exploit_actions[action] += 1

            # Take a SGD step if buffer contains at least one batch
            if len(agent.replay_buffer) > batch_size:
                loss = agent.update(batch_size)

            # Log a few metrics into tensorboard every `LOG_EVERY`
            if step == max_steps-1 and episode > 0 and episode % LOG_EVERY == 0:
                episode_reward = torch.as_tensor(episode_reward)
                print('EPISODE: {}\tREWARD = {}\tLOSS = {}\t \
                    NODES = {}\tEXPLOITED {}% TIMES\tEPS = {}\tACTIONS = {}'.
                    format(episode, round(episode_reward.item(), 7),
                           round(loss.item(), 7), len(env.get_visited_nodes()),
                           round(100*len(reward_exploit)/max_steps, 1),
                           round(epsilon[episode], 4), exploit_actions))

                writer.add_scalar('loss', loss, episode)
                writer.add_scalar('reward', episode_reward, episode)
                writer.add_scalar('nodes', len(env.get_visited_nodes()) , episode)
                writer.add_scalar('eps', epsilon[episode], episode)
                writer.add_scalar('reward exploit', np.mean(reward_exploit), episode)

                try:
                    for tag, parm in agent.Q_eval.named_parameters():
                        writer.add_histogram(tag, parm.grad.data.cpu().numpy(), episode)
                except Exception as e:
                    print(episode, e)
                break

            if done:
                print('Done! Episode = {}\tStep = {}\tReward = {}'.
                    format(episode, step, episode_reward))
                wins += 1
                break
            # Take the transition and update current state
            state = next_state.clone()

    # Print out the final number of wins
    print('{} WINS'.format(wins))
    return


if __name__ == '__main__':

    logs_base_dir = "runs"
    os.makedirs(logs_base_dir, exist_ok=True)

    MAX_EPISODES = 3
    MAX_STEPS = 25
    BATCH_SIZE = 32
    LR = 1e-3
    GAMMA = 0.99
    WEIGHT_DEC = 0
    BETA = 0.99
    MAX_MEMORY = int(1e5)

    FIXED_START = False
    PLOT = False

    env = Environment(path_to_map='mazes/5x5/maze_5x5_0.npy', 
                      fixed_start_node=FIXED_START)
    print('There are {} visitable nodes in this graph'.
        format(len(env.valid_nodes)))
    agent = Agent(env, learning_rate=LR, gamma=GAMMA, beta=BETA,
                  weight_dec=WEIGHT_DEC, buffer_size=MAX_MEMORY)
    mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)
    FNAME = 'models/model_{}.pt'.format(datetime.now().strftime("%d_%m_%Y-%H%M%S"))
    torch.save(agent.Q_eval.state_dict(), FNAME)
