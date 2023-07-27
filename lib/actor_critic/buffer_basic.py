"""
Do not support different experience length of different agents.
"""
import copy
import scipy.signal
import torch
import numpy as np


class Buffer:

    def __init__(self, buffer_size, n_agents, dim_observation_actor, dim_observation_critic, configuration):

        self.buffer_size = buffer_size

        self.n_agents = n_agents

        self.device = configuration.device

        self.gamma = configuration.gamma

        self.gae_lam = configuration.gae_lam

        self.dim_observation_actor = \
            (dim_observation_actor, ) if np.isscalar(dim_observation_actor) else dim_observation_actor

        self.dim_observation_critic = \
            (dim_observation_critic,) if np.isscalar(dim_observation_critic) else dim_observation_critic

        # Initialization.

        self.buffer_observation_actor = \
            torch.zeros((buffer_size, n_agents, *self.dim_observation_actor), dtype=torch.double, device=self.device)

        self.buffer_observation_critic = \
            torch.zeros((buffer_size, n_agents, *self.dim_observation_critic), dtype=torch.double, device=self.device)

        self.buffer_action = torch.zeros((buffer_size, n_agents), dtype=torch.long, device=self.device)

        self.buffer_action_log_probability = torch.zeros((buffer_size, n_agents), dtype=torch.long, device=self.device)

        # For compute return.

        self.buffer_reward = torch.zeros((buffer_size, n_agents), dtype=torch.double, device=self.device)

        # For compute advantage.

        self.buffer_value = torch.zeros((buffer_size, n_agents), dtype=torch.double, device=self.device)

        self.buffer_advantage = torch.zeros((buffer_size, n_agents), dtype=torch.double, device=self.device)

        self.buffer_return = torch.zeros((buffer_size, n_agents), dtype=torch.double, device=self.device)

        # Buffer tracking.

        self.ptr, self.path_start_idx, self.max_buffer_size = 0, 0, buffer_size

        pass

    def store(self, swarm_observations_actor, swarm_observations_critic,
              swarm_actions, swarm_rewards, swarm_values, swarm_actions_log_probability,
              swarm_agent_active_index=None):

        # Buffer has to have room so you can store.
        assert self.ptr < self.max_buffer_size

        self.buffer_observation_actor[self.ptr] = swarm_observations_actor
        self.buffer_observation_critic[self.ptr] = swarm_observations_critic
        self.buffer_action[self.ptr] = swarm_actions
        self.buffer_reward[self.ptr] = swarm_rewards
        self.buffer_value[self.ptr] = swarm_values
        self.buffer_action_log_probability[self.ptr] = swarm_actions_log_probability

        self.ptr += 1

    def finish_path(self, swarm_last_values):

        path_slice = slice(self.path_start_idx, self.ptr)

        rewards = torch.vstack((self.buffer_reward[path_slice, :], swarm_last_values))
        values = torch.vstack((self.buffer_value[path_slice, :], swarm_last_values))

        # The next two lines implement GAE-Lambda advantage calculation.

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.buffer_advantage[path_slice] = \
            torch.from_numpy(self.discount_cumulative_sum(deltas.tolist(), self.gamma * self.gae_lam, axis=0).copy())

        # The next line computes rewards-to-go, to be targets for the value function.

        self.buffer_return[path_slice] = \
            torch.from_numpy(self.discount_cumulative_sum(rewards.tolist(), self.gamma, axis=0)[:-1].copy())

        self.path_start_idx = self.ptr

        pass

    def get(self):

        # Buffer has to be full before you can get. Why?
        # Ensure one epoch has enough experiences to train the model.
        # One epoch can have the experiences of more than one or less than one episode.

        assert self.ptr == self.max_buffer_size

        # The next two lines implement the advantage normalization trick.

        advantage_mean = torch.mean(self.buffer_advantage, dtype=torch.double)
        advantage_std = torch.std(self.buffer_advantage, unbiased=False)

        self.buffer_advantage = (self.buffer_advantage - advantage_mean) / (advantage_std + 1e-5)

        # (batch, ...)
        data = dict(obs_actor=self.buffer_observation_actor.reshape(self.buffer_size * self.n_agents,
                                                                    *self.dim_observation_actor),
                    obs_critic=self.buffer_observation_critic.reshape(self.buffer_size * self.n_agents,
                                                                      *self.dim_observation_critic),
                    act=self.buffer_action.reshape(self.buffer_size * self.n_agents),
                    ret=self.buffer_return.reshape(self.buffer_size * self.n_agents),
                    adv=self.buffer_advantage.reshape(self.buffer_size * self.n_agents),
                    val=self.buffer_value.reshape(self.buffer_size * self.n_agents),
                    logp=self.buffer_action_log_probability.reshape(self.buffer_size * self.n_agents)
                    )

        self.ptr, self.path_start_idx = 0, 0

        return copy.deepcopy(data)

    def discount_cumulative_sum(self, x, discount, axis=0):
        """
        Magic from rllab for computing discounted cumulative sums of vectors.
        Input:
            Vector x =
            [x0,
             x1,
             x2]
        Output:
            [x0 + discount * x1 + discount^2 * x2,
             x1 + discount * x2,
             x2]
        """
        # scipy.signal.lfilter(b, a, x, axis=-1, zi=None)
        # scipy.signal.lfilter([1], [1, -0.9], [1, 2, 3])
        # array([1.  , 2.9 , 5.61])

        # return scipy.signal.lfilter([1], [1, float(-discount)], x.flip(dims=(0,)), axis=axis)[::-1]

        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=axis)[::-1]


def test_2d_discount_cumulative_sum():
    # array([1.  , 2.9 , 5.61])
    scipy.signal.lfilter([1], [1, -0.9], [1, 2, 3], axis=0)

    # array([[1.  , 1.  ],
    #        [2.9 , 2.9 ],
    #        [5.61, 5.61]])
    scipy.signal.lfilter([1], [1, -0.9], [[1, 1], [2, 2], [3, 3]], axis=0)
    pass

