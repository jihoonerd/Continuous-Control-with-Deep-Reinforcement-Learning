import numpy as np


class ReplayBuffer:

    def __init__(self, capacity, input_shape, n_actions):
        self.capacity = capacity
        self.total_count = 0
        self.state_memory = np.zeros((self.capacity, *input_shape))
        self.new_state_memory = np.zeros((self.capacity, *input_shape))
        # action is a real-valued number
        self.action_memory = np.zeros((self.capacity, n_actions))
        self.reward_memory = np.zeros(self.capacity)
        # cast it to float instead of bool
        self.terminal_memory = np.zeros(self.capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        index = self.total_count % self.capacity
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_state
        self.terminal_memory[index] = 1 - done
        self.total_count += 1

    def get_minibatch(self, batch_size):
        max_mem = min(self.total_count, self.capacity)
        minibatch_mem_id = np.random.choice(max_mem, batch_size)

        states = self.state_memory[minibatch_mem_id]
        new_states = self.new_state_memory[minibatch_mem_id]
        rewards = self.reward_memory[minibatch_mem_id]
        actions = self.action_memory[minibatch_mem_id]
        terminal = self.terminal_memory[minibatch_mem_id]

        return states, actions, rewards, new_states, terminal
