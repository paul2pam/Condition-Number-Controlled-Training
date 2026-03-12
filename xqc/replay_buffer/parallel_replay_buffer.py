import collections

import gymnasium as gym
import numpy as np

Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "masks", "next_observations", "discount"],
)


class ParallelReplayBuffer:
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        capacity: int,
        num_seeds: int,
        n_steps: int,
        gamma: float,
    ):
        self.observations = np.empty(
            (num_seeds, capacity, observation_space.shape[-1]),
            dtype=observation_space.dtype,
        )
        self.next_observations = np.empty(
            (num_seeds, capacity, observation_space.shape[-1]),
            dtype=observation_space.dtype,
        )
        self.actions = np.empty(
            (num_seeds, capacity, action_space.shape[-1]), dtype=np.float32
        )
        self.rewards = np.empty((num_seeds, capacity), dtype=np.float32)
        self.masks = np.empty((num_seeds, capacity), dtype=np.float32)
        self.timeouts = np.empty((num_seeds, capacity), dtype=np.float32)

        self.num_seeds = num_seeds
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma
        self.size = 0
        self.insert_index = 0

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        truncs: bool,
        next_observation: np.ndarray,
    ):
        self.observations[:, self.insert_index] = observation
        self.next_observations[:, self.insert_index] = next_observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.timeouts[:, self.insert_index] = truncs

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_parallel_multibatch(self, batch_size: int, num_batches: int) -> Batch:
        indxs = np.random.randint(self.size, size=(num_batches, batch_size))

        last_valid_idx = self.insert_index - 1
        original_timeout_values = self.timeouts[:, last_valid_idx].copy()
        self.timeouts[:, last_valid_idx] = np.logical_or(
            original_timeout_values, self.masks[:, last_valid_idx]
        )

        steps = np.arange(self.n_steps).reshape(1, -1)
        indices = (indxs[..., None] + steps) % self.capacity

        rewards_seq = self.rewards[:, indices]
        done_seq = np.logical_not(self.masks[:, indices])
        truncated_seq = self.timeouts[:, indices]

        done_or_trunc = np.logical_or(done_seq, truncated_seq)
        done_idx = done_or_trunc.argmax(axis=-1)
        has_done_or_truncated = done_or_trunc.any(axis=-1)
        done_idx = np.where(has_done_or_truncated, done_idx, self.n_steps - 1)

        mask = np.arange(self.n_steps).reshape(1, 1, 1, -1) <= done_idx[..., None]
        target_q_discounts = self.gamma ** mask.sum(axis=-1, keepdims=True).astype(
            np.float32
        )

        discounts = self.gamma ** np.arange(self.n_steps, dtype=np.float32).reshape(
            1, 1, 1, -1
        )
        discounted_rewards = rewards_seq * discounts * mask
        n_step_returns = discounted_rewards.sum(axis=-1)  # [batch, 1]

        # Compute indices of next_obs/done at the final point of the n-step transition
        last_indices = (indxs[None] + done_idx) % self.capacity
        next_obs = self.next_observations[
            np.arange(self.next_observations.shape[0])[:, None, None], last_indices
        ]
        next_masks = self.masks[np.arange(self.num_seeds)[:, None, None], last_indices]
        self.timeouts[:, last_valid_idx] = original_timeout_values

        return Batch(
            observations=self.observations[:, indxs],
            actions=self.actions[:, indxs],
            rewards=n_step_returns,
            masks=next_masks,
            next_observations=next_obs,
            discount=target_q_discounts,
        )
