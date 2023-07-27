"""
Random select an action,
regardless of no others, such as other legality of actions, collisions, other agents.

Author: Lijun SUN.
"""
import torch
import torch.nn as nn


class RandomAgent(nn.Module):

    def __init__(self, n_agents=1, n_actions=5):

        super().__init__()

        self.n_agents = n_agents

        self.n_actions = n_actions

    def forward(self, observations=None, *args):

        n_agents = self.n_agents if observations is None else observations.shape[0]

        action = torch.randint(low=0, high=self.n_actions, size=(n_agents,))
        # action = torch.randint(low=0, high=self.n_actions, size=(self.n_agents,)).squeeze()

        return action


