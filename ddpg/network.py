import os
import pathlib

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorNetwork(nn.Module):

    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions, name, ckpt_dir='ckpt'):
        super().__init__()
        self.learning_rate = learning_rate
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_file = os.path.join(ckpt_dir, self.name + 'actor_ddpg_ckpt')

        # first dense layer
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)

        # initialize the layer as supplementary information says.
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        # It uses LayerNorm instaed of BatchNorm since we are scaling for samples, not for features
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        # second dense layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # last layer
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        f3 = 0.003  # actor uses 0.003
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        if not os.path.exists(ckpt_dir):
            pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.mu(x)
        x = T.tanh(x)  # section 7
        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        print("Model saved")

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        print("Model loaded")


class CriticNetwork(nn.Module):

    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions, name, ckpt_dir='ckpt'):
        super().__init__()
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_file = os.path.join(ckpt_dir, self.name + 'critic_ddpg_ckpt')

        # first dense layer
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)

        # initialize the layer as supplementary information says.
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        # It uses LayerNorm instaed of BatchNorm since we are scaling for samples, not for features
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        # second dense layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # last layer
        self.action_value = nn.Linear(self.n_actions, fc2_dims)

        f3 = 0.0003  # critic uses 0.0003
        self.q = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))

        state_action_value = F.relu(T.add(state_value, action_value))

        state_action_value = self.q(state_action_value)
        return state_action_value

    def save_checkpoint(self):
        print("Saving Checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("Loading Checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file))
