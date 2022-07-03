import random
import numpy as np
import torch

class ReplayBuffer:
    """Replay Buffer."""

    def __init__(self, buffer_size):
        """Replay Buffer Initialization."""
        self.buffer = []
        self.buffer_size = buffer_size

    def append(self, item):
        """Append to existing Replay Buffer."""
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(item)

    def sample(self, batch_size):
        """Sampling."""
        batch = random.sample(self.buffer, batch_size)

        batch_state = torch.cat(tuple(b[0] for b in batch))
        batch_action = torch.cat(tuple(b[1] for b in batch))
        batch_reward = torch.cat(tuple(b[2] for b in batch))
        batch_state_new = torch.cat(tuple(b[3] for b in batch))
        batch_terminal = torch.from_numpy(np.stack(tuple(float(b[4]) for b in batch)))
        
        return batch_state, batch_action, batch_reward, batch_state_new, batch_terminal