import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ddpg.network import ActorNetwork, CriticNetwork
from ddpg.noise_injector import OrnsteinUhlenbeckActionNoise
from ddpg.replaybuffer import ReplayBuffer


class Agent:

    def __init__(self, lr_actor, lr_critic, input_dims, tau, gamma=0.99, n_actions=2, max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau

        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(n_actions))
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.actor = ActorNetwork(
            lr_actor, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='Actor')
        self.target_actor = ActorNetwork(
            lr_actor, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='TargetActor')

        self.critic = CriticNetwork(
            lr_critic, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='Critic')
        self.target_critic = CriticNetwork(
            lr_critic, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='TargetCritic')

        self.update_network_parameters(tau=1)

        self.batch_size = batch_size

    def choose_action(self, observation):
        self.actor.eval()  # Turn eval mode on. This is just for inference.
        observation = T.tensor(
            observation, dtype=T.float).to(self.actor.device)
        mu = self.actor(observation).to(self.actor.device)
        mu_w_noise = mu + T.tensor(self.noise(),
                                   dtype=T.float).to(self.actor.device)
        self.actor.train()  # Recover model mode
        return mu_w_noise.cpu().detach().numpy()

    def learn(self):

        # If memory is not big enough to train, skip it.
        if self.memory.total_count < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.get_minibatch(
            self.batch_size)
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        # Freeze following network
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(next_state)
        critic_value_next = self.target_critic.forward(
            next_state, target_actions)
        # previous value, to be updated.
        critic_value = self.critic.forward(state, action)

        critic_target = []
        for j in range(self.batch_size):
            critic_target.append(
                reward[j] + self.gamma * critic_value_next[j] * done[j])

        critic_target = T.tensor(critic_target).to(self.critic.device)
        critic_target = critic_target.view(self.batch_size, 1)

        # now update critic network
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(critic_target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()  # now, freeze it for actor update

        # now update actor network
        self.actor.train()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        # output of critic is Q value, which needs to be maximized. Therefore put negative sign to this to minimize.
        actor_q = self.critic.forward(state, mu)
        actor_loss = T.mean(-actor_q)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters(self.tau)

    def update_network_parameters(self, tau):

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                (1-tau) * target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                (1-tau) * target_actor_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
