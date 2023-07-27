"""
Support different experience length of different agents.
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

        self.buffer_action_log_probability = torch.zeros((buffer_size, n_agents), dtype=torch.double,
                                                         device=self.device)

        # For compute return.

        self.buffer_reward = torch.zeros((buffer_size, n_agents), dtype=torch.double, device=self.device)

        # For compute advantage.

        self.buffer_value = torch.zeros((buffer_size, n_agents), dtype=torch.double, device=self.device)

        self.buffer_advantage = torch.zeros((buffer_size, n_agents), dtype=torch.double, device=self.device)

        self.buffer_return = torch.zeros((buffer_size, n_agents), dtype=torch.double, device=self.device)

        # Buffer tracking.

        self.ptr = torch.zeros(n_agents, dtype=torch.long, device=self.device)
        self.path_start_idx = torch.zeros(n_agents, dtype=torch.long, device=self.device)
        self.max_buffer_size = buffer_size
        self.swarm_agent_active_index = torch.arange(0, n_agents, 1, device=self.device)
        pass

    def store(self, swarm_observations_actor, swarm_observations_critic,
              swarm_actions, swarm_rewards, swarm_values, swarm_actions_log_probability,
              swarm_agent_active_index=None):

        if swarm_agent_active_index is None:
            swarm_agent_active_index = self.swarm_agent_active_index

        # Buffer has to have room so you can store.

        assert torch.max(self.ptr) < self.max_buffer_size

        self.buffer_observation_actor[self.ptr[swarm_agent_active_index], swarm_agent_active_index] = \
            swarm_observations_actor
        self.buffer_observation_critic[self.ptr[swarm_agent_active_index], swarm_agent_active_index] = \
            swarm_observations_critic
        self.buffer_action[self.ptr[swarm_agent_active_index], swarm_agent_active_index] = swarm_actions
        self.buffer_reward[self.ptr[swarm_agent_active_index], swarm_agent_active_index] = swarm_rewards
        self.buffer_value[self.ptr[swarm_agent_active_index], swarm_agent_active_index] = swarm_values
        self.buffer_action_log_probability[self.ptr[swarm_agent_active_index], swarm_agent_active_index] = \
            swarm_actions_log_probability

        self.ptr[swarm_agent_active_index] += 1

    def finish_path_by_index(self, last_value, agent_index):

        path_slice = slice(self.path_start_idx[agent_index], self.ptr[agent_index])

        rewards = torch.hstack((self.buffer_reward[path_slice, agent_index], last_value))
        values = torch.hstack((self.buffer_value[path_slice, agent_index], last_value))

        # The next two lines implement GAE-Lambda advantage calculation.

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.buffer_advantage[path_slice, agent_index] = \
            torch.from_numpy(self.discount_cumulative_sum(deltas.tolist(), self.gamma * self.gae_lam, axis=0).copy())

        # The next line computes rewards-to-go, to be targets for the value function.

        self.buffer_return[path_slice, agent_index] = \
            torch.from_numpy(self.discount_cumulative_sum(rewards.tolist(), self.gamma, axis=0)[:-1].copy())

        self.path_start_idx[agent_index] = copy.deepcopy(self.ptr[agent_index])

        pass

    def get(self):

        # Buffer has to be full before you can get. Why?
        # Ensure one epoch has enough experiences to train the model.
        # One epoch can have the experiences of more than one or less than one episode.

        # The following line is useful in single-agent case, not necessary in multi-agent case.
        # In multi-agent case, especially when # of agents change from time to time in an episode,
        # it cannot assure an agent of a specific index collect all epoch timestep experience.
        # E.g., one epoch has two episode, with the max epoch timestep 40.
        # In episode 1, agent 1 exit at timestep 10, agent 2 exist the longest to  timestep 20.
        # In episode 2, agent 2 exist the longest to timestep 20, agent 2 exit early at timestep 15.
        # When the epoch ends, agent 1 has 30 timesteps experience, agent 2 has 35 timesteps, neither reach 40.

        # assert torch.max(self.ptr) == self.max_buffer_size

        # Merge data.

        buffer_observation_actor = []
        buffer_observation_critic = []
        buffer_action = []
        buffer_return = []
        buffer_advantage = []
        buffer_value = []
        buffer_action_log_probability = []

        for idx_agent in range(self.n_agents):

            path_slice = slice(0, self.ptr[idx_agent])

            buffer_observation_actor.append(self.buffer_observation_actor[path_slice, idx_agent])
            buffer_observation_critic.append(self.buffer_observation_critic[path_slice, idx_agent])
            buffer_action.append(self.buffer_action[path_slice, idx_agent])
            buffer_return.append(self.buffer_return[path_slice, idx_agent])
            buffer_advantage.append(self.buffer_advantage[path_slice, idx_agent])
            buffer_value.append(self.buffer_value[path_slice, idx_agent])
            buffer_action_log_probability.append(self.buffer_action_log_probability[path_slice, idx_agent])

        buffer_observation_actor = torch.cat(buffer_observation_actor, dim=0)
        buffer_observation_critic = torch.cat(buffer_observation_critic, dim=0)
        buffer_action = torch.cat(buffer_action, dim=0)
        buffer_return = torch.cat(buffer_return, dim=0)
        buffer_advantage = torch.cat(buffer_advantage, dim=0)
        buffer_value = torch.cat(buffer_value, dim=0)
        buffer_action_log_probability = torch.cat(buffer_action_log_probability, dim=0)

        # The next two lines implement the advantage normalization trick.

        advantage_mean = torch.mean(buffer_advantage, dtype=torch.double)
        advantage_std = torch.std(buffer_advantage, unbiased=False)

        buffer_advantage = (buffer_advantage - advantage_mean) / (advantage_std + 1e-5)

        # (batch, ...)
        data = dict(obs_actor=buffer_observation_actor,
                    obs_critic=buffer_observation_critic,
                    act=buffer_action,
                    ret=buffer_return,
                    adv=buffer_advantage,
                    val=buffer_value,
                    logp=buffer_action_log_probability
                    )

        self.ptr = torch.zeros(self.n_agents, dtype=torch.long, device=self.device)
        self.path_start_idx = torch.zeros(self.n_agents, dtype=torch.long, device=self.device)

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

