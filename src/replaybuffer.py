import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, obs_dim, action_dim):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer = []
        self.ptr = 0

    def store_transition(self, obs, action, reward, next_obs, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.ptr] = (obs, action, reward, next_obs, done)
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        obs_batch = np.array([transition[0] for transition in batch])
        action_batch = np.array([transition[1] for transition in batch])
        reward_batch = np.array([transition[2] for transition in batch])
        next_obs_batch = np.array([transition[3] for transition in batch])
        done_batch = np.array([transition[4] for transition in batch])
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def size(self):
        return len(self.buffer)