import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, obs_dim, action_dim, rehearsal_size=0):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer = []
        self.ptr = 0
        self.rehearsal_size = max(0, int(rehearsal_size))
        self.rehearsal_buffer = []
        self.total_seen = 0

    def store_transition(self, obs, action, reward, next_obs, done):
        transition = (
            np.asarray(obs, dtype=np.float32).copy(),
            np.asarray(action, dtype=np.float32).copy(),
            float(reward),
            np.asarray(next_obs, dtype=np.float32).copy(),
            bool(done),
        )
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.max_size
        self.total_seen += 1

        if self.rehearsal_size <= 0:
            return

        if len(self.rehearsal_buffer) < self.rehearsal_size:
            self.rehearsal_buffer.append(transition)
            return

        reservoir_index = np.random.randint(0, self.total_seen)
        if reservoir_index < self.rehearsal_size:
            self.rehearsal_buffer[reservoir_index] = transition

    def _format_batch(self, batch):
        obs_batch = np.array([transition[0] for transition in batch])
        action_batch = np.array([transition[1] for transition in batch])
        reward_batch = np.array([transition[2] for transition in batch])
        next_obs_batch = np.array([transition[3] for transition in batch])
        done_batch = np.array([transition[4] for transition in batch])
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def sample(self, batch_size, rehearsal_fraction=0.0):
        total_available = len(self.buffer) + len(self.rehearsal_buffer)
        batch_size = min(batch_size, total_available)
        if batch_size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        rehearsal_fraction = float(np.clip(rehearsal_fraction, 0.0, 1.0))
        rehearsal_count = min(
            len(self.rehearsal_buffer),
            int(round(batch_size * rehearsal_fraction)),
        )
        main_count = min(len(self.buffer), batch_size - rehearsal_count)

        remaining = batch_size - (main_count + rehearsal_count)
        if remaining > 0 and len(self.buffer) - main_count > 0:
            extra_main = min(len(self.buffer) - main_count, remaining)
            main_count += extra_main
            remaining -= extra_main
        if remaining > 0 and len(self.rehearsal_buffer) - rehearsal_count > 0:
            rehearsal_count += min(len(self.rehearsal_buffer) - rehearsal_count, remaining)

        batch = []
        if main_count > 0:
            main_indices = np.random.choice(len(self.buffer), main_count, replace=False)
            batch.extend(self.buffer[i] for i in main_indices)
        if rehearsal_count > 0:
            rehearsal_indices = np.random.choice(
                len(self.rehearsal_buffer),
                rehearsal_count,
                replace=False,
            )
            batch.extend(self.rehearsal_buffer[i] for i in rehearsal_indices)

        np.random.shuffle(batch)
        return self._format_batch(batch)

    def size(self):
        return len(self.buffer)
