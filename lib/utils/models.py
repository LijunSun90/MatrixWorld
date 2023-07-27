import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import numpy as np


class ModelMLP(nn.Module):

    def __init__(self, dim_input, dim_output, hidden_sizes=(400, 300),
                 with_layer_normalization=True, use_initialization=False):

        super().__init__()

        # Input check.

        dim_input = np.prod(dim_input) if not np.isscalar(dim_input) else dim_input

        self.with_layer_normalization = with_layer_normalization

        self.fc1 = nn.Linear(dim_input, hidden_sizes[0], dtype=torch.double)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1], dtype=torch.double)
        self.fc3 = nn.Linear(hidden_sizes[1], dim_output, dtype=torch.double)

        # Network initialization.

        if use_initialization:

            self.fc1 = self.initialize_linear_layer(self.fc1)
            self.fc2 = self.initialize_linear_layer(self.fc2)
            self.fc3 = self.initialize_linear_layer(self.fc3)

        self.layer_normalize_1 = nn.LayerNorm(hidden_sizes[0], dtype=torch.double)
        self.layer_normalize_2 = nn.LayerNorm(hidden_sizes[1], dtype=torch.double)
        self.layer_normalize_3 = nn.LayerNorm(dim_output, dtype=torch.double)

        self.relu = nn.ReLU()

    def forward(self, x):

        # If no layer normalization, even with BCE loss clamp, loss will be nan after some time in some experiments.

        if self.with_layer_normalization:

            x = self.relu(self.layer_normalize_1(self.fc1(x)))
            x = self.relu(self.layer_normalize_2(self.fc2(x)))

            logits = self.layer_normalize_3(self.fc3(x))

        else:

            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))

            logits = self.fc3(x)

        return logits.squeeze(dim=-1)

    @staticmethod
    def initialize_linear_layer(module, gain=None):
        gain = nn.init.calculate_gain('relu') if gain is None else gain
        nn.init.orthogonal_(module.weight.data, gain=gain)
        nn.init.constant_(module.bias.data, 0)
        return module


class ModelPolicyBase(nn.Module):

    def __init__(self):

        super().__init__()

    def compute_action_distribution(self, observation):
        raise NotImplementedError

    def forward(self, observation):

        action_distribution = self.compute_action_distribution(observation)

        action = action_distribution.sample()

        action_log_probability = action_distribution.log_prob(action)

        return action, action_log_probability

    def evaluate_action(self, observation, action):

        action_distribution = self.compute_action_distribution(observation)

        action_log_probability = action_distribution.log_prob(action)

        distribution_entropy = action_distribution.entropy().mean()

        return action_log_probability, distribution_entropy


class ModelPolicy(ModelPolicyBase):

    def __init__(self, dim_input, dim_output, hidden_sizes=(400, 300),
                 with_layer_normalization=True, use_initialization=False):

        super().__init__()

        self.model = ModelMLP(dim_input=dim_input,
                              dim_output=dim_output,
                              hidden_sizes=hidden_sizes,
                              with_layer_normalization=with_layer_normalization,
                              use_initialization=use_initialization)

    def compute_action_distribution(self, observation):

        # Neural network policy.

        # (batch, -1)

        # Flatten the last three dimensions.

        flatten_dim = observation.shape[-3] * observation.shape[-2] * observation.shape[-1]

        n_dim = observation.dim()

        if n_dim > 4:

            observation = observation.reshape(*observation.shape[:-3], flatten_dim)

        elif n_dim == 4:

            observation = observation.reshape(observation.shape[0], flatten_dim)

        else:

            observation = observation.reshape(flatten_dim)

        logits = self.model(observation)

        action_distribution = Categorical(logits=logits)

        return action_distribution


class ModelPolicySafe(ModelPolicyBase):
    """
    Safe in obstacle avoidance.
    """

    def __init__(self, dim_input, dim_output, hidden_sizes=(400, 300),
                 with_layer_normalization=True, use_initialization=False,
                 device=torch.device('cpu'), obstacle_channel=[2]):

        super().__init__()

        self.model_policy = ModelMLP(dim_input=dim_input,
                                     dim_output=dim_output,
                                     hidden_sizes=hidden_sizes,
                                     with_layer_normalization=with_layer_normalization,
                                     use_initialization=use_initialization)

        self.model_obstacle_avoidance = ModelRuledObstacleAvoidance(device=device, obstacle_channel=obstacle_channel)

    def compute_action_distribution(self, observation):
        """
        :param observation: (batch, fov, fov, 3).
        :return:
        """
        # Ruled obstacle avoidance.

        obstacle_sigmoid_actions = self.model_obstacle_avoidance(observation)

        # Neural network policy.

        # (batch, -1)

        # Flatten the last three dimensions.

        flatten_dim = observation.shape[-3] * observation.shape[-2] * observation.shape[-1]

        n_dim = observation.dim()

        if n_dim > 4:

            observation = observation.reshape(*observation.shape[:-3], flatten_dim)

        elif n_dim == 4:

            observation = observation.reshape(observation.shape[0], flatten_dim)

        else:

            observation = observation.reshape(flatten_dim)

        logits = self.model_policy(observation)

        action_distribution = Categorical(logits=logits)

        # Fuse actions.

        fused_action_distribution = self.fuse_actions(probability_actions=action_distribution.probs,
                                                      obstacle_sigmoid_actions=obstacle_sigmoid_actions)

        action_distribution = Categorical(probs=fused_action_distribution)

        return action_distribution

    @staticmethod
    def fuse_actions(probability_actions, obstacle_sigmoid_actions):

        fused_actions = torch.minimum(probability_actions, obstacle_sigmoid_actions)

        return fused_actions


class ModelRuledObstacleAvoidance(nn.Module):

    def __init__(self, device, obstacle_channel=[2]):

        super().__init__()

        self.device = device

        self.obstacle_channel = obstacle_channel

    def forward(self, observation):

        # Input preprocessing: any input shape to (batch, fov, fov, 3).

        n_dim = observation.dim()

        if n_dim > 4:

            batch_size, set_size = observation.shape[:2]

            batch_observation = observation.reshape(batch_size * set_size, *observation.shape[2:])

            batch_sigmoid_actions = self.compute_obstacle_avoidance_actions(batch_observation)

            batch_set_sigmoid_actions = batch_sigmoid_actions.reshape(batch_size, set_size,
                                                                      batch_sigmoid_actions.shape[-1])

            return batch_set_sigmoid_actions

        elif n_dim == 4:

            batch_sigmoid_actions = self.compute_obstacle_avoidance_actions(observation)

            return batch_sigmoid_actions

        elif n_dim == 3:

            batch_observation = observation.unsqueeze(dim=0)

            batch_sigmoid_actions = self.compute_obstacle_avoidance_actions(batch_observation)

            sigmoid_actions = batch_sigmoid_actions.squeeze(dim=0)

            return sigmoid_actions

        else:

            raise ValueError('Incorrect input shape! Required shape: (batch, fov, fov, 3).')

    def compute_obstacle_avoidance_actions(self, batch_observation):
        """
        :param batch_observation: (batch_size, fov, fov, 3).
        :return: (batch_size, dim_action).
        """

        batch_size, fov_scope = batch_observation.shape[:2]

        agent_position = torch.tensor([round((fov_scope - 1) / 2), round((fov_scope - 1) / 2)],
                                      dtype=torch.long, device=self.device)

        # (batch_size, n_rows, n_columns)

        batch_observation_obstacles = batch_observation[:, :, :, self.obstacle_channel].sum(dim=-1)

        sigmoid_actions = self.legality_of_actions(agent_position, batch_observation_obstacles)

        return sigmoid_actions

    def legality_of_actions(self, agent_position, batch_observation_obstacles):
        """
        :param agent_position: (2,).
        :param batch_observation_obstacles:
        :return:
        """

        # Rule.

        # (5, 2).

        action_directions = torch.tensor([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]],
                                         dtype=torch.long, device=self.device)

        # (5, 2).

        to_positions = action_directions + agent_position

        # (batch_size, dim_action=5).

        batch_collide_obstacle = torch.stack(list(map(
            lambda to_position: batch_observation_obstacles[:, to_position[0], to_position[1]] > 0,
            to_positions)), dim=-1)

        # (batch_size,).

        action_legality = ~ batch_collide_obstacle

        action_legality = action_legality.type(torch.double)

        return action_legality


class ModelActorCritic(nn.Module):

    def __init__(self, model_policy, model_value):

        super().__init__()

        self.model_policy = model_policy

        self.model_value = model_value

        pass

    def forward(self, observation_actor, observation_critic=None):

        observation_critic = observation_actor if observation_critic is None else observation_critic

        with torch.no_grad():

            action, action_log_probability = self.model_policy(observation_actor)

            value = self.model_value(observation_critic).squeeze(dim=-1)

        return action, value, action_log_probability

