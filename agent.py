import torch
from replay import ReplayMemory
from model import GNN
from torch_geometric.data import Batch
import numpy as np


class Agent:
    def __init__(self,
                 env,
                 learning_rate=5e-4,
                 gamma=0.9,
                 beta=1e-3,
                 weight_dec=5e-4,
                 buffer_size=10000,
                 use_ddqn=False):

        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.beta = beta
        self.weight_dec = weight_dec
        self.replay_buffer = ReplayMemory(max_size=buffer_size)
        self.use_ddqn = use_ddqn

        # Select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Q_eval = GNN(num_features=env.data.num_features, 
                          num_actions=env.action_space.n).to(self.device)
        self.Q_target = GNN(num_features=env.data.num_features, 
                            num_actions=env.action_space.n).to(self.device)

        self.soft_update_target()

        # Use Adam
        self.optimizer = torch.optim.Adam(self.Q_eval.parameters(), 
                                          lr=self.learning_rate,
                                          weight_decay=self.weight_dec)

        # Specify loss
        self.loss_fn = torch.nn.MSELoss()
        #self.loss_fn = torch.nn.SmoothL1Loss()

        # Printing model & params
        print(self.Q_eval)
        trainable_params = sum(p.numel() for p in self.Q_eval.parameters() if p.requires_grad)
        print('Trainable params = {}'.format(trainable_params))
        for name, p in self.Q_eval.named_parameters():
            if p.requires_grad:
                print('\t{} : {} trainable params'.format(name, p.numel()))

    def get_action(self, state, eps=0.20):
        '''
        Epsilon-greedy strategy for action selection.
        '''
        # Select next action based on an epsilon-greedy strategy
        if np.random.rand() < eps:
            exploit_flag = False
            # Randomly sample from action space
            action = self.env.action_space.sample()
        else:
            exploit_flag = True
            # Required by torch_geometric
            state = Batch.from_data_list([state]).to(self.device)

            # Calculate the Q(s,a) approximation
            self.Q_eval.eval()
            with torch.no_grad():
                qvals = self.Q_eval(state)
            self.Q_eval.train()
            action = np.argmax(qvals.cpu().detach().numpy())
        return action, exploit_flag

    def compute_loss(self, batch):
        '''
        Compute loss for batch.
        '''
        # De-compose batch
        states, actions, rewards, next_states, dones = batch
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        states, next_states = Batch.from_data_list(states), Batch.from_data_list(next_states)
        states, next_states = states.to(self.device), next_states.to(self.device)

        self.Q_eval.train()
        self.Q_target.eval()

        # Calculate current Q
        curr_Q = self.Q_eval(states).gather(1, actions.unsqueeze(1))

        # Calculate next Q and its max
        with torch.no_grad():
            # Use DDQN algorithm
            if self.use_ddqn:
                _, max_actions = self.Q_eval(next_states).max(1, keepdim=True)
                max_next_Q = self.Q_target(next_states).gather(1, max_actions)
            # Use DQN algorithm
            else:
                next_Q = self.Q_target(next_states)
                max_next_Q = next_Q.max(1)[0].unsqueeze(1)

        dones = dones.unsqueeze(1)

        # Take expectation
        expected_Q = rewards + (1-dones)*self.gamma*max_next_Q

        # Compute loss
        loss = self.loss_fn(curr_Q, expected_Q)
        return loss

    def update(self, batch_size):
        '''
        Update network parameters via SGD.
        '''
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.Q_eval.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.soft_update_target()
        return loss

    def soft_update_target(self):
        '''
        θ_target = β*θ_online + (1 - β)*θ_target
        '''
        for target_param, local_param in zip(self.Q_target.parameters(),
                                             self.Q_eval.parameters()):
            target_param.data.copy_(self.beta*local_param.data + (1-self.beta)*target_param.data)
